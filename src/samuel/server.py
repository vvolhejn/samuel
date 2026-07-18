"""FastAPI backend for the speech-mimic webapp.

Receives a recorded utterance (WAV), extracts pyin pitch the same way
training did, runs the controller checkpoint, and returns the Pink Trombone
parameter trajectories in native JS units for the browser synth to play.

Run:
    uv run --extra server uvicorn samuel.server:app --port 8000

Env:
    SAMUEL_CHECKPOINT   local .pt path or wandb artifact ref
                        (default: runs/onset-off_20260527-193518/checkpoints/last.pt)
    SAMUEL_RUN_CONFIG   run config.json to build the model from
                        (default: the same run's config.json)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from samuel.data import fill_unvoiced
from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola

logger = logging.getLogger("samuel.server")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUN_DIR = _REPO_ROOT / "runs" / "onset-off_20260527-193518"

# pyin settings matching scripts/precompute_pitch.py defaults (the settings
# the pitch cache for this run was built with).
PYIN_FMIN = 70.0
PYIN_FMAX = 500.0
PYIN_FRAME_LENGTH = 4096

# Matches the run's synth.ir_length (read raw — the stored synth block has
# keys the current SynthConfig rejects).
IR_LENGTH = 256

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


def _load_model() -> PinkTromboneController:
    config_path = Path(
        os.environ.get("SAMUEL_RUN_CONFIG", _DEFAULT_RUN_DIR / "config.json")
    )
    checkpoint_ref = os.environ.get(
        "SAMUEL_CHECKPOINT", str(_DEFAULT_RUN_DIR / "checkpoints" / "last.pt")
    )

    # Only the "model" block is read: the run's full config may contain keys
    # from other branches that the current pydantic configs reject.
    run_config = json.loads(config_path.read_text())
    model_config = PinkTromboneControllerConfig.model_validate(run_config["model"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PinkTromboneController(model_config).to(device)
    ckpt = torch.load(
        _resolve_checkpoint(checkpoint_ref), map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(
        "loaded checkpoint %s on %s (frame_rate=%.3f)",
        checkpoint_ref,
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


@app.on_event("startup")
def _startup() -> None:
    global _model
    _model = _load_model()


@app.get("/api/health")
def health() -> dict:
    assert _model is not None
    return {
        "status": "ok",
        "frame_rate": _model.config.frame_rate,
        "device": str(next(_model.parameters()).device),
    }


@app.post("/api/synthesize")
async def synthesize(request: Request) -> dict:
    assert _model is not None
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
    audio = audio.astype(np.float32, copy=False)
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
