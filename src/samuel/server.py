"""FastAPI backend for the speech-mimic webapp.

Receives a recorded utterance (WAV), extracts pyin pitch the same way
training did, runs the controller checkpoint, and returns the Pink Trombone
parameter trajectories in native JS units for the browser synth to play.

Also serves the webapp frontend: on startup it builds the Next.js static
export (``webapp/out``) and mounts it at ``/``, so a single process serves
both the UI and the ``/api/*`` endpoints on the same origin — just open the
server's URL, no separate ``pnpm dev`` needed.

Run (serves UI + API on http://127.0.0.1:8471):
    uv run --extra server python -m samuel.server
    # or, equivalently, with explicit uvicorn:
    uv run --extra server uvicorn samuel.server:app --port 8471

Env:
    SAMUEL_PORT / SAMUEL_HOST   override the default 127.0.0.1:8471
                                (python -m samuel.server entrypoint only)
    SAMUEL_CHECKPOINT   local .pt path or wandb artifact ref
                        (default: runs/reg-f1.0-s0.1_20260719-120003/checkpoints/last.pt)
    SAMUEL_RUN_CONFIG   run config.json to build the model from. Optional:
                        by default it is found next to the checkpoint's run
                        dir (<run_dir>/config.json). Required only for a
                        wandb-artifact checkpoint, which has no config beside it.
    SAMUEL_MANIFEST     dataset manifest (jsonl) for /api/dataset_clip
                        (default: the run config's data.manifest_path)
    SAMUEL_SERVE_FRONTEND    "0" to serve only the API (no build/mount).
                             Default builds and serves the frontend.
    SAMUEL_FRONTEND_SKIP_BUILD  "1" to serve an existing webapp/out without
                             rebuilding (fast restarts during backend work).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import shutil
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from samuel.data import fill_unvoiced
from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola

logger = logging.getLogger("samuel.server")

_REPO_ROOT = Path(__file__).resolve().parents[2]
# wandb: moboehle-kyutai/samuel/xl45k4i7
_DEFAULT_RUN_DIR = _REPO_ROOT / "runs" / "reg-f1.0-s0.1_20260719-120003"

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

# Length of the clips /api/dataset_clip serves (matches the eval chunking).
CLIP_SECONDS = 10.0

# Cap on the per-frame volume-match gain. Training's _volume_match is
# unclamped, but near-silent synth frames against noisy mic input can
# produce absurd gains; the JS synth's absolute level differs from the
# Python synth's anyway, so the envelope shape is what matters.
MAX_GAIN = 30.0


def _resolve_checkpoint(ref: str) -> Path:
    """Local ``.pt`` path, or a wandb artifact ref downloaded on demand."""
    if Path(ref).exists():
        return Path(ref)

    import wandb

    artifact = wandb.Api().artifact(ref, type="model")
    return Path(artifact.download()) / "last.pt"


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
# request handlers (e.g. _get_clips) can reuse it without re-resolving paths.
_run_cfg: dict | None = None


def _run_config() -> dict:
    assert _run_cfg is not None, "run config not loaded (call _load_model first)"
    return _run_cfg


def _load_model() -> PinkTromboneController:
    global _run_cfg
    checkpoint_ref = os.environ.get(
        "SAMUEL_CHECKPOINT", str(_DEFAULT_RUN_DIR / "checkpoints" / "last.pt")
    )
    checkpoint_path = _resolve_checkpoint(checkpoint_ref)
    config_path = _resolve_config_path(checkpoint_path)
    _run_cfg = json.loads(config_path.read_text())

    # Only the "model" block is read: the run's full config may contain keys
    # from other branches that the current pydantic configs reject.
    model_config = PinkTromboneControllerConfig.model_validate(_run_cfg["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PinkTromboneController(model_config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        "loaded checkpoint %s (config %s) on %s (frame_rate=%.3f)",
        checkpoint_path,
        config_path,
        device,
        model_config.frame_rate,
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


def _volume_match(
    synth: np.ndarray, target: np.ndarray, hop: int, t_ctrl: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame RMS gain matching ``synth`` to ``target`` (train._volume_match).

    Returns (gain [t_ctrl], synth_normalized). The gain vector is padded with
    its last value up to ``t_ctrl`` so it aligns with the parameter frames.
    """
    S = min(len(synth), len(target))
    T = S // hop
    synth_f = synth[: T * hop].reshape(T, hop)
    tgt_f = target[: T * hop].reshape(T, hop)
    synth_rms = np.sqrt(np.clip((synth_f**2).mean(-1), 1e-12, None))
    tgt_rms = np.sqrt(np.clip((tgt_f**2).mean(-1), 1e-12, None))
    gain = np.clip(tgt_rms / synth_rms, 0.0, MAX_GAIN).astype(np.float32)
    synth_norm = (synth_f * gain[:, None]).reshape(-1).astype(np.float32)
    if len(gain) < t_ctrl:
        pad_value = gain[-1] if len(gain) else 1.0
        gain = np.pad(gain, (0, t_ctrl - len(gain)), constant_values=pad_value)
    return gain[:t_ctrl], synth_norm


def _encode_wav_b64(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    peak = np.abs(audio).max()
    if peak > 1.0:  # PCM_16 clips beyond [-1, 1]
        audio = audio / peak
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


app = FastAPI(title="samuel speech-mimic backend")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: PinkTromboneController | None = None
_clips: list[dict] | None = None


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


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = _load_model()

    # Mount the built frontend last, so the explicit /api/* routes (registered
    # at import time) still take precedence over the catch-all mount at "/".
    dist = _build_frontend()
    if dist is not None:
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
        logger.info("serving frontend from %s", dist)


def _get_clips() -> list[dict]:
    """Manifest entries long enough for CLIP_SECONDS, loaded on first use
    (the manifest may live on a filesystem that isn't always mounted)."""
    global _clips
    if _clips is None:
        manifest_path = Path(
            os.environ.get("SAMUEL_MANIFEST") or _run_config()["data"]["manifest_path"]
        )
        entries = [
            json.loads(line)
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        ]
        _clips = [e for e in entries if e["duration"] >= CLIP_SECONDS]
        logger.info(
            "manifest %s: %d/%d clips >= %.0fs",
            manifest_path,
            len(_clips),
            len(entries),
            CLIP_SECONDS,
        )
    return _clips


@app.get("/api/health")
def health() -> dict:
    assert _model is not None
    return {
        "status": "ok",
        "frame_rate": _model.config.frame_rate,
        "device": str(next(_model.parameters()).device),
    }


def _mimic(audio: np.ndarray) -> dict:
    """Run the controller on ``audio`` (float32 at SAMPLE_RATE) and build the
    response payload shared by /api/synthesize and /api/dataset_clip."""
    assert _model is not None
    if len(audio) < _model.samples_per_frame:
        raise HTTPException(status_code=400, detail="audio too short")

    t_ctrl = _model.t_ctrl_for(len(audio))
    f0, voiced = _pitch_track(audio, _model.samples_per_frame, t_ctrl)

    device = next(_model.parameters()).device
    wav = torch.from_numpy(audio).to(device)[None, None, :]
    f0_t = torch.from_numpy(f0).to(device)[None, :]
    with torch.no_grad():
        params = _model(wav, f0_t)  # [1, T_ctrl, N_PARAMS]
        # Reference resynthesis with the Python synth (for A/B debugging),
        # mirroring the eval loop: OLA synth + per-frame RMS match to target.
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

    gain, synth_norm = _volume_match(synth, audio, _model.samples_per_frame, t_ctrl)
    params = params[0].cpu().numpy()

    return {
        "frame_rate": _model.config.frame_rate,
        "n_frames": t_ctrl,
        "duration_s": len(audio) / SAMPLE_RATE,
        "params": {name: params[:, i].tolist() for i, name in enumerate(PARAM_NAMES)},
        "voiced": voiced.tolist(),
        # Per-frame gain the browser synth should apply (training's loss only
        # ever saw volume-matched output, so the raw model output has no
        # meaningful envelope — silence would hum without this).
        "gain": gain.tolist(),
        "synth_audio_b64": _encode_wav_b64(synth_norm),
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


@app.post("/api/dataset_clip")
def dataset_clip() -> dict:
    """Mimic a random CLIP_SECONDS window from the training manifest.

    The response is /api/synthesize's payload plus the source clip itself
    (``clip_audio_b64``) so the browser can play it before the imitation.
    """
    try:
        clips = _get_clips()
    except OSError as e:
        raise HTTPException(status_code=503, detail=f"manifest unavailable: {e}")
    if not clips:
        raise HTTPException(status_code=503, detail="no manifest clip is long enough")

    entry = random.choice(clips)
    try:
        info = sf.info(entry["path"])
        n_clip = int(CLIP_SECONDS * info.samplerate)
        start = random.randint(0, max(0, info.frames - n_clip))
        audio, sr = sf.read(
            entry["path"],
            start=start,
            frames=n_clip,
            dtype="float32",
            always_2d=False,
        )
    except OSError as e:
        raise HTTPException(status_code=503, detail=f"could not read clip: {e}")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio.astype(np.float32, copy=False)

    result = _mimic(audio)
    result["clip_path"] = entry["path"]
    result["clip_offset_s"] = start / sr
    result["clip_audio_b64"] = _encode_wav_b64(audio)
    return result


# Uncommon default port (avoids clashing with the usual 8000/8080/3000). The
# dev proxy in webapp/next.config.ts points here too — keep them in sync.
DEFAULT_PORT = 8471

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("SAMUEL_HOST", "127.0.0.1"),
        port=int(os.environ.get("SAMUEL_PORT", DEFAULT_PORT)),
    )
