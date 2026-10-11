"""FastAPI backend for the speech-mimic webapp.

Receives a recorded utterance (WAV), extracts pyin pitch the same way
training did, runs the controller checkpoint, and returns the Pink Trombone
parameter trajectories in native JS units for the browser synth to play.

Also serves the webapp frontend: on startup it builds the Next.js static
export (``webapp/out``) and mounts it at ``/``, so a single process serves
both the UI and the ``/api/*`` endpoints on the same origin — just open the
server's URL, no separate ``pnpm dev`` needed.

By default it serves the published checkpoint from the Hugging Face Hub
(``vvolhejn/samuel``), downloaded and cached on first start, so a fresh clone
runs without access to the training cluster. ``--checkpoint`` (or
``SAMUEL_CHECKPOINT``) points it at something else.

Run (serves UI + API on http://127.0.0.1:8471):
    uv run --extra server python -m samuel.server
    uv run --extra server python -m samuel.server --checkpoint runs/<run>/checkpoints/last.pt
    # or, equivalently, with explicit uvicorn (env only, no CLI flags):
    uv run --extra server uvicorn samuel.server:app --port 8471

The flags of ``python -m samuel.server --help`` each set the matching env var
below, which is what the app itself reads.

Env:
    SAMUEL_PORT / SAMUEL_HOST   override the default 127.0.0.1:8471
                                (python -m samuel.server entrypoint only)
    SAMUEL_CHECKPOINT   Hugging Face repo ref, local .pt path, wandb run
                        URL/path, or wandb artifact ref
                        (default: hf:vvolhejn/samuel)
                        A Hub ref is ``hf:<repo-id>[@<revision>]``; the repo
                        holds checkpoints/last.pt next to config.json, so it
                        needs no SAMUEL_RUN_CONFIG either.
                        A run URL (https://wandb.ai/<entity>/<project>/runs/<id>,
                        or just <entity>/<project>/runs/<id>) is self-contained:
                        the run's model artifact and its config both come from
                        wandb, so no SAMUEL_RUN_CONFIG is needed.
    SAMUEL_RUN_CONFIG   run config.json to build the model from. Optional:
                        by default it is found next to the checkpoint's run
                        dir (<run_dir>/config.json), or taken from the wandb run
                        when SAMUEL_CHECKPOINT is a run URL. Required only for a
                        bare wandb-artifact checkpoint, which has no config beside it.
    SAMUEL_SERVE_FRONTEND    "0" to serve only the API (no build/mount).
                             Default builds and serves the frontend.
    SAMUEL_ALLOW_ORIGINS     comma-separated origins allowed to call /api/*
                             cross-origin, for a separately hosted frontend
                             (e.g. https://samuel.vvolhejn.com). Localhost is
                             always allowed.
    SAMUEL_FRONTEND_SKIP_BUILD  "1" to serve an existing webapp/out without
                             rebuilding (fast restarts during backend work).
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from samuel.data import fill_unvoiced
from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola

logger = logging.getLogger("samuel.server")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The published checkpoint, served unless overridden: a public Hub repo holding
# checkpoints/last.pt plus the run's config.json (staged by deploy/stage-model.sh,
# see deploy/README.md). Default so a clone without runs/ still works.
_DEFAULT_CHECKPOINT = "hf:vvolhejn/samuel"

# The text-to-speech model (Pocket TTS generating Samuel's trajectories, see
# scripts/tts/say.py): hf:<repo>[@rev] or a local copy of one, served unless
# SAMUEL_TTS is set to "". Optional: when it can't be loaded the rest of the
# server runs as usual and /api/tts answers 503.
_DEFAULT_TTS = "hf:vvolhejn/samuel-tts"
# Longest text /api/tts takes, and the longest it lets the model talk for.
TTS_MAX_CHARS = 300
TTS_MAX_SEC = 20.0
# Sampling settings the model was evaluated with (see its model card).
TTS_TEMP = 0.3
TTS_CFG = 2.0

# Next.js frontend: `pnpm build` (output: "export") emits the static site here.
_WEBAPP_DIR = _REPO_ROOT / "webapp"
_FRONTEND_DIST = _WEBAPP_DIR / "out"

# pyin settings matching scripts/precompute_pitch.py defaults (the settings
# the pitch cache for this run was built with).
PYIN_FMIN = 70.0
PYIN_FMAX = 500.0
PYIN_FRAME_LENGTH = 4096

# Matches the run's synth.ir_length (read raw — the stored synth block has
# keys the current SynthConfig rejects).
IR_LENGTH = 256


# https://wandb.ai/<entity>/<project>/runs/<id>[/...][?query], or the bare
# <entity>/<project>/runs/<id> path.
_WANDB_RUN_RE = re.compile(
    r"^(?:https?://[^/]*wandb\.ai/)?(?P<entity>[^/?#]+)/(?P<project>[^/?#]+)"
    r"/runs/(?P<run_id>[^/?#]+)"
)

# hf:<owner>/<name>[@<revision>], also spelled hf://... — deliberately an
# explicit scheme, so a bare <a>/<b> string stays unambiguous next to the wandb
# forms above.
_HF_REPO_RE = re.compile(
    r"^hf:(?://)?(?P<repo_id>[^/@?#]+/[^/@?#]+)(?:@(?P<revision>[^/?#]+))?$"
)


def _resolve_hf_repo(ref: str) -> Path:
    """Download a Hub model repo; return the path to its checkpoint.

    The repo mirrors a run dir (``checkpoints/last.pt`` beside ``config.json``),
    so :func:`_resolve_config_path` finds the config without help.
    """
    from huggingface_hub import snapshot_download

    match = _HF_REPO_RE.match(ref)
    assert match is not None, ref
    repo_id, revision = match.group("repo_id"), match.group("revision")
    local_dir = Path(
        snapshot_download(repo_id, revision=revision, allow_patterns=["*.json", "*.pt"])
    )
    logger.info(
        "hugging face repo %s (revision %s): downloaded to %s",
        repo_id,
        revision or "main",
        local_dir,
    )
    return local_dir / "checkpoints" / "last.pt"


def _resolve_wandb_run(ref: str) -> tuple[Path, dict]:
    """Download a wandb run's checkpoint artifact; return (path, run config).

    The run config comes from wandb (``train.py`` logs the full config dump as
    ``wandb.init(config=...)``), so a run URL needs no local run dir.
    """
    import wandb

    match = _WANDB_RUN_RE.match(ref)
    assert match is not None, ref
    run_path = "{entity}/{project}/{run_id}".format(**match.groupdict())
    run = wandb.Api().run(run_path)

    models = [a for a in run.logged_artifacts() if a.type == "model"]
    if not models:
        raise FileNotFoundError(
            f"wandb run {run_path} ({run.name}) logged no model artifact; "
            "point SAMUEL_CHECKPOINT at a local .pt instead (only runs with "
            "log.ckpt_wandb_artifact upload one)"
        )
    # Runs mirror each checkpoint to the same artifact name, so pick the highest
    # version (older versions are usually deleted, but not always).
    artifact = max(models, key=lambda a: int(a.version.lstrip("v")))
    logger.info(
        "wandb run %s (%s): using artifact %s", run_path, run.name, artifact.name
    )
    return Path(artifact.download()) / "last.pt", dict(run.config)


def _resolve_checkpoint(ref: str) -> tuple[Path, dict | None]:
    """Resolve a checkpoint ref to a local path, plus its run config if known.

    ``ref`` is a Hub repo ref, a local ``.pt`` path, a wandb run URL/path, or a
    wandb artifact ref. Only the wandb-run form carries a config; the others
    return None and leave config resolution to :func:`_resolve_config_path`.
    """
    if _HF_REPO_RE.match(ref):
        return _resolve_hf_repo(ref), None

    if Path(ref).exists():
        return Path(ref), None

    if _WANDB_RUN_RE.match(ref):
        return _resolve_wandb_run(ref)

    import wandb

    artifact = wandb.Api().artifact(ref, type="model")
    return Path(artifact.download()) / "last.pt", None


def _resolve_config_path(checkpoint_path: Path) -> Path:
    """The run config.json for a checkpoint.

    Explicit ``SAMUEL_RUN_CONFIG`` wins; otherwise it is found relative to the
    checkpoint. Runs are laid out as ``<run_dir>/checkpoints/<step>.pt`` with
    ``<run_dir>/config.json`` alongside, so we look one and two levels up.
    A wandb-artifact checkpoint has no config.json beside it — set
    ``SAMUEL_RUN_CONFIG`` explicitly in that case.
    """
    override = os.environ.get("SAMUEL_RUN_CONFIG")
    if override:
        return Path(override)

    for candidate in (
        checkpoint_path.parent.parent / "config.json",  # <run_dir>/config.json
        checkpoint_path.parent / "config.json",  # checkpoint sits in run dir
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"no config.json found near checkpoint {checkpoint_path}; "
        "set SAMUEL_RUN_CONFIG (required when the checkpoint is a wandb "
        "artifact, whose download dir contains only last.pt)"
    )


# Run config for the loaded checkpoint, populated by _load_model at startup so
# request handlers can reuse it without re-resolving paths.
_run_cfg: dict | None = None
# What the webapp shows for the loaded checkpoint: a wandb run URL, or the
# resolved local .pt path (also set by _load_model).
_checkpoint: str | None = None
# Content hash of the loaded model, see _model_fingerprint (also set there).
_fingerprint: str | None = None


def _model_fingerprint(model: PinkTromboneController) -> str:
    """A short content hash of the loaded model: same weights, same string.

    Unlike ``_checkpoint`` this does not depend on where the weights came from,
    so it can be committed and compared across machines. The webapp's
    precomputed clip responses carry it, which is how a checkpoint swap is
    noticed instead of being served stale forever — see
    ``scripts/precompute_clip_responses.py``.
    """
    digest = hashlib.sha256(model.config.model_dump_json().encode())
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to("cpu").numpy().tobytes())
    return digest.hexdigest()[:16]


def _run_config() -> dict:
    assert _run_cfg is not None, "run config not loaded (call _load_model first)"
    return _run_cfg


def _load_model() -> PinkTromboneController:
    global _run_cfg, _checkpoint, _fingerprint
    checkpoint_ref = os.environ.get("SAMUEL_CHECKPOINT") or _DEFAULT_CHECKPOINT
    checkpoint_path, run_cfg = _resolve_checkpoint(checkpoint_ref)
    override = os.environ.get("SAMUEL_RUN_CONFIG")
    if run_cfg is not None and not override:
        config_source = f"wandb run config ({checkpoint_ref})"
        _run_cfg = run_cfg
    else:
        config_path = _resolve_config_path(checkpoint_path)
        config_source = str(config_path)
        _run_cfg = json.loads(config_path.read_text())

    # Only the "model" block is read: the run's full config may contain keys
    # from other branches that the current pydantic configs reject.
    model_config = PinkTromboneControllerConfig.model_validate(_run_cfg["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PinkTromboneController(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # Same precedence as _resolve_checkpoint: hub ref, then local path, then run.
    hf_match = _HF_REPO_RE.match(checkpoint_ref)
    run_match = (
        None
        if hf_match is not None or Path(checkpoint_ref).exists()
        else _WANDB_RUN_RE.match(checkpoint_ref)
    )
    if hf_match is not None:
        # A URL, so the webapp renders it as a link (see app/page.tsx).
        _checkpoint = "https://huggingface.co/" + hf_match.group("repo_id")
        if hf_match.group("revision"):
            _checkpoint += "/tree/" + hf_match.group("revision")
    elif run_match is not None:
        # Normalized so the webapp can link it even for the bare-path form.
        _checkpoint = "https://wandb.ai/{entity}/{project}/runs/{run_id}".format(
            **run_match.groupdict()
        )
    else:
        # last.pt is a symlink to <step>.pt; resolve so the real file is visible.
        _checkpoint = str(checkpoint_path.resolve())
    _fingerprint = _model_fingerprint(model)
    logger.info(
        "loaded checkpoint %s (config %s) on %s (frame_rate=%.3f, fingerprint %s)",
        checkpoint_path,
        config_source,
        device,
        model_config.frame_rate,
        _fingerprint,
    )
    return model


def _pitch_track(
    audio: np.ndarray, samples_per_frame: int, t_ctrl: int
) -> tuple[np.ndarray, np.ndarray]:
    """pyin f0 at hop=samples_per_frame, filled and trimmed/padded to t_ctrl."""
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=SAMPLE_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=samples_per_frame,
    )
    voiced = voiced_flag & np.isfinite(f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)

    if len(f0) < t_ctrl:
        pad = t_ctrl - len(f0)
        f0 = np.pad(f0, (0, pad))
        voiced = np.pad(voiced, (0, pad))
    f0, voiced = f0[:t_ctrl], voiced[:t_ctrl]
    return fill_unvoiced(f0, voiced, PYIN_FMIN, PYIN_FMAX), voiced


def _target_rms() -> float:
    """``data.target_rms`` for the loaded checkpoint, so inference matches training."""
    return float(_run_config()["data"]["target_rms"])


def _rms_normalize(wav: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale ``wav`` to ``target_rms`` (train._rms_normalize)."""
    rms = float(np.sqrt(np.clip((wav.astype(np.float64) ** 2).mean(), 1e-12, None)))
    return (wav * (target_rms / rms)).astype(np.float32)


def _encode_wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    peak = np.abs(audio).max()
    if peak > 1.0:  # PCM_16 clips beyond [-1, 1]
        audio = audio / peak
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Localhost is always allowed: it covers both the single-process setup and the
# dev workflow (Next on :3000, uvicorn on :8471).
_LOCALHOST_ORIGIN_RE = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"


def _allow_origin_regex() -> str:
    """Regex of origins allowed to call /api/* cross-origin.

    Adds each comma-separated origin in ``SAMUEL_ALLOW_ORIGINS`` to localhost,
    for a frontend hosted apart from this backend (e.g. the static export on
    https://samuel.vvolhejn.com against a Hugging Face Space).
    """
    extra = os.environ.get("SAMUEL_ALLOW_ORIGINS", "")
    origins = [o.strip().rstrip("/") for o in extra.split(",") if o.strip()]
    if origins:
        logger.info("allowing cross-origin requests from %s", ", ".join(origins))
    return "|".join([_LOCALHOST_ORIGIN_RE, *(re.escape(o) for o in origins)])


app = FastAPI(title="samuel speech-mimic backend")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_allow_origin_regex(),
    allow_methods=["*"],
    allow_headers=["*"],
    # /api/synthesize sends Content-Type: audio/wav, which is not CORS-safelisted,
    # so every utterance preflights. Let the browser cache that for an hour.
    max_age=3600,
)

_model: PinkTromboneController | None = None
# (FlowLM, SamuelCodec) of the text-to-speech model, or None if it isn't served.
_tts: tuple | None = None
# Generation keeps per-call state on the model; one request at a time.
_tts_lock = threading.Lock()


def _build_frontend() -> Path | None:
    """Build the Next.js static export and return its output dir (or None).

    Honors SAMUEL_SERVE_FRONTEND ("0" disables) and SAMUEL_FRONTEND_SKIP_BUILD
    ("1" serves an existing build without rebuilding). Any failure downgrades
    to API-only rather than taking the whole server down.
    """
    if os.environ.get("SAMUEL_SERVE_FRONTEND", "1") == "0":
        logger.info("SAMUEL_SERVE_FRONTEND=0; serving API only")
        return None
    if not _WEBAPP_DIR.is_dir():
        logger.warning("webapp dir %s not found; serving API only", _WEBAPP_DIR)
        return None

    if os.environ.get("SAMUEL_FRONTEND_SKIP_BUILD") == "1":
        if _FRONTEND_DIST.is_dir():
            logger.info("SKIP_BUILD set; serving existing %s", _FRONTEND_DIST)
            return _FRONTEND_DIST
        logger.warning("SKIP_BUILD set but %s missing; building", _FRONTEND_DIST)

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        logger.warning("pnpm not on PATH; cannot build frontend")
        return _FRONTEND_DIST if _FRONTEND_DIST.is_dir() else None

    try:
        if not (_WEBAPP_DIR / "node_modules").is_dir():
            logger.info("installing frontend deps (pnpm install)...")
            subprocess.run([pnpm, "install"], cwd=_WEBAPP_DIR, check=True)
        logger.info("building frontend (pnpm build)... this can take a minute")
        subprocess.run([pnpm, "build"], cwd=_WEBAPP_DIR, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("frontend build failed (%s); serving API only", e)
        return _FRONTEND_DIST if _FRONTEND_DIST.is_dir() else None

    if not _FRONTEND_DIST.is_dir():
        logger.warning(
            "build finished but %s missing; serving API only", _FRONTEND_DIST
        )
        return None
    return _FRONTEND_DIST


def _warm_up() -> None:
    """Run one throwaway request so the first real one is not the slow one.

    librosa's pyin is numba-jitted, and compiling it costs several seconds the
    first time it runs — long enough to look broken to whoever speaks first
    after the server starts. Doing it here moves the cost into startup.
    """
    t = np.arange(SAMPLE_RATE // 2, dtype=np.float32) / SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 140.0 * t, dtype=np.float32)
    try:
        _mimic(tone)
    except Exception as e:  # noqa: BLE001 - an optimization; never block startup
        logger.warning("warm-up pass failed (%s); first request will be slow", e)
    else:
        logger.info("warm-up pass done")


def _load_tts() -> tuple | None:
    """The text-to-speech model, or None if it is disabled or can't be loaded."""
    ref = os.environ.get("SAMUEL_TTS", _DEFAULT_TTS)
    if not ref:
        logger.info("text-to-speech disabled")
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        # The vendored pocket-tts training package.
        from training.load import load_run

        # The codec gets the controller this server already runs rather than
        # loading its own copy (and downloading it, in the container). Turning
        # trajectories into sound only reads its config.
        model, codec, _ = load_run(
            ref,
            device,
            codec_kwargs={"checkpoint": _model, "target_rms": _target_rms()},
        )
    except Exception as e:  # noqa: BLE001 - an extra; never block startup
        logger.warning("text-to-speech unavailable (%s: %s)", type(e).__name__, e)
        return None
    logger.info("loaded text-to-speech model %s on %s", ref, device)
    return model, codec


@app.on_event("startup")
def _startup() -> None:
    global _model, _tts
    _model = _load_model()
    _warm_up()
    _tts = _load_tts()

    # Mount the built frontend last, so the explicit /api/* routes (registered
    # at import time) still take precedence over the catch-all mount at "/".
    dist = _build_frontend()
    if dist is not None:
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
        logger.info("serving frontend from %s", dist)


@app.get("/api/health")
def health() -> dict:
    assert _model is not None
    assert _checkpoint is not None
    return {
        "status": "ok",
        "frame_rate": _model.config.frame_rate,
        "device": str(next(_model.parameters()).device),
        "checkpoint": _checkpoint,
        "model_fingerprint": _fingerprint,
        "tts": _tts is not None,
    }


def _mimic(audio: np.ndarray) -> dict:
    """Run the controller on ``audio`` (float32 at SAMPLE_RATE) and build the
    response payload shared by /api/synthesize and /api/dataset_clip."""
    assert _model is not None
    if len(audio) < _model.samples_per_frame:
        raise HTTPException(status_code=400, detail="audio too short")

    t_ctrl = _model.t_ctrl_for(len(audio))
    f0, voiced = _pitch_track(audio, _model.samples_per_frame, t_ctrl)

    # The model was trained on RMS-normalised input, so normalise here too.
    audio_in = _rms_normalize(audio, _target_rms())

    device = next(_model.parameters()).device
    wav = torch.from_numpy(audio_in).to(device)[None, None, :]
    f0_t = torch.from_numpy(f0).to(device)[None, :]
    with torch.no_grad():
        # Reference resynthesis with the Python synth (for A/B debugging),
        # mirroring the eval loop.
        params = _model(wav, f0_t)  # [1, T_ctrl, N_PARAMS]
        synth = (
            pink_trombone_ola(
                params,
                seed=0,
                ir_length=IR_LENGTH,
                control_rate=_model.config.frame_rate,
            )[0]
            .cpu()
            .numpy()
        )

    # The synth output is used as-is: the model was scored on its own output
    # level, so it carries the energy envelope itself (via intensity).
    params = params[0].cpu().numpy()

    return {
        "frame_rate": _model.config.frame_rate,
        "n_frames": t_ctrl,
        "duration_s": len(audio) / SAMPLE_RATE,
        "params": {name: params[:, i].tolist() for i, name in enumerate(PARAM_NAMES)},
        "voiced": voiced.tolist(),
        "synth_audio_b64": _encode_wav_b64(synth.astype(np.float32)),
    }


@app.post("/api/synthesize")
async def synthesize(request: Request) -> dict:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body")
    try:
        audio, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    except Exception as e:  # noqa: BLE001 - malformed uploads
        raise HTTPException(status_code=400, detail=f"could not decode audio: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return _mimic(audio.astype(np.float32, copy=False))


class TtsRequest(BaseModel):
    text: str


@app.post("/api/tts")
def tts(request: TtsRequest) -> dict:
    """Speak ``text``: the same response as /api/synthesize, but the trajectories
    come from the text-to-speech model instead of from imitating audio."""
    if _tts is None:
        raise HTTPException(
            status_code=503, detail="text-to-speech is not available on this server"
        )
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty text")
    if len(text) > TTS_MAX_CHARS:
        raise HTTPException(
            status_code=400, detail=f"text longer than {TTS_MAX_CHARS} characters"
        )
    from pocket_tts.models.text_chunking import prepare_text_prompt

    model, codec = _tts
    # Same text clean-up as Pocket TTS inference, with this model's settings
    # (config.yaml on the Hub): capitalised, terminal punctuation added.
    prompt, _ = prepare_text_prompt(
        text, pad_with_spaces_for_short_inputs=False, remove_semicolons=False
    )
    tokens = torch.tensor(
        model.flow_lm.conditioner.tokenizer.sp.encode(prompt), dtype=torch.long
    )
    device = model.flow_lm.bos_emb.device
    with _tts_lock, torch.no_grad():
        # No voice prompt: there is only one voice anyway.
        (latents,) = model.generate(
            [tokens],
            [torch.zeros(0, codec.latent_dim, device=device)],
            max_frames=int(TTS_MAX_SEC * codec.frame_rate),
            temp=TTS_TEMP,
            cfg_coef=TTS_CFG,
        )
        if latents.shape[0] < codec.min_decode_frames:
            raise HTTPException(
                status_code=500, detail="the model said nothing; try again"
            )
        latents = latents[None].to(device)
        params = codec.latents_to_params(latents)[0].cpu().numpy()
        synth = codec.decode_to_audio(latents)[0, 0].cpu().numpy()

    frame_rate = SAMPLE_RATE / codec.samples_per_frame
    voiceness = params[:, PARAM_NAMES.index("voiceness")]
    return {
        "frame_rate": frame_rate,
        "n_frames": len(params),
        "duration_s": len(params) / frame_rate,
        "params": {name: params[:, i].tolist() for i, name in enumerate(PARAM_NAMES)},
        # There is no pyin track for generated speech; voiceness stands in.
        "voiced": (voiceness >= 0.5).tolist(),
        "synth_audio_b64": _encode_wav_b64(synth.astype(np.float32)),
    }


# Uncommon default port (avoids clashing with the usual 8000/8080/3000). The
# dev proxy in webapp/next.config.ts points here too — keep them in sync.
DEFAULT_PORT = 8471


def _parse_args() -> None:
    """Turn CLI flags into the env vars the app reads.

    Env stays the single source of truth, since ``uvicorn samuel.server:app``
    imports the app directly and never sees these flags. Each flag defaults to
    the matching env var, so both spellings work and the flag wins.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m samuel.server",
        description="Serve the samuel webapp (UI + /api/*).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAMUEL_CHECKPOINT") or _DEFAULT_CHECKPOINT,
        help="hf:<repo-id>[@<revision>], a local .pt, a wandb run URL/path, "
        "or a wandb artifact ref",
    )
    parser.add_argument(
        "--run-config",
        default=os.environ.get("SAMUEL_RUN_CONFIG"),
        help="run config.json to build the model from; by default found next to "
        "the checkpoint (only needed for a bare wandb artifact)",
    )
    parser.add_argument(
        "--tts",
        default=os.environ.get("SAMUEL_TTS", _DEFAULT_TTS),
        help='text-to-speech model: hf:<repo-id>[@<revision>] or a local run dir; "" '
        "disables it",
    )
    parser.add_argument("--host", default=os.environ.get("SAMUEL_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("SAMUEL_PORT", DEFAULT_PORT))
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        default=os.environ.get("SAMUEL_SERVE_FRONTEND", "1") == "0",
        help="serve only the API, without building or mounting the frontend",
    )
    parser.add_argument(
        "--skip-frontend-build",
        action="store_true",
        default=os.environ.get("SAMUEL_FRONTEND_SKIP_BUILD") == "1",
        help="serve an existing webapp/out without rebuilding it",
    )
    args = parser.parse_args()

    os.environ["SAMUEL_CHECKPOINT"] = args.checkpoint
    if args.run_config:
        os.environ["SAMUEL_RUN_CONFIG"] = args.run_config
    os.environ["SAMUEL_TTS"] = args.tts
    os.environ["SAMUEL_HOST"] = args.host
    os.environ["SAMUEL_PORT"] = str(args.port)
    os.environ["SAMUEL_SERVE_FRONTEND"] = "0" if args.no_frontend else "1"
    os.environ["SAMUEL_FRONTEND_SKIP_BUILD"] = "1" if args.skip_frontend_build else "0"


if __name__ == "__main__":
    import uvicorn

    _parse_args()
    uvicorn.run(
        app,
        host=os.environ["SAMUEL_HOST"],
        port=int(os.environ["SAMUEL_PORT"]),
    )
