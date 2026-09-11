"""Samuel as the latent codec for Pocket TTS training.

Encode runs the speech-to-speech controller on audio and returns its Pink
Trombone parameter trajectories; decode runs the Python synth on them. The
Pocket TTS FlowLM then learns text -> trajectories instead of text -> Mimi
latents, and never touches audio.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from multiprocessing import get_context
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from training.codec import LatentCodec

from samuel.data import fill_unvoiced
from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola

# pyin settings the published checkpoint was trained with (scripts/precompute_pitch.py).
PYIN_FMIN = 70.0
PYIN_FMAX = 500.0
PYIN_FRAME_LENGTH = 4096

_HF_REPO_RE = re.compile(
    r"^hf:(?://)?(?P<repo_id>[^/@?#]+/[^/@?#]+)(?:@(?P<revision>[^/?#]+))?$"
)


def resolve_checkpoint(ref: str) -> tuple[Path, dict]:
    """``hf:<repo>[@rev]`` or a local ``.pt`` path -> (checkpoint path, run config)."""
    match = _HF_REPO_RE.match(ref)
    if match is not None:
        from huggingface_hub import snapshot_download

        local = Path(
            snapshot_download(
                match.group("repo_id"),
                revision=match.group("revision"),
                allow_patterns=["*.json", "*.pt"],
            )
        )
        ckpt = local / "checkpoints" / "last.pt"
    else:
        ckpt = Path(ref)
    for cfg in (ckpt.parent.parent / "config.json", ckpt.parent / "config.json"):
        if cfg.exists():
            return ckpt, json.loads(cfg.read_text())
    raise FileNotFoundError(f"no config.json next to {ckpt}")


def load_controller(ref: str) -> tuple[PinkTromboneController, dict]:
    ckpt_path, run_cfg = resolve_checkpoint(ref)
    config = PinkTromboneControllerConfig.model_validate(run_cfg["model"])
    model = PinkTromboneController(config)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval(), run_cfg


def pyin_f0(audio: np.ndarray, samples_per_frame: int, n_frames: int) -> np.ndarray:
    """Filled pyin f0 track (Hz) with exactly ``n_frames`` frames."""
    import librosa

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=PYIN_FMIN,
        fmax=PYIN_FMAX,
        sr=SAMPLE_RATE,
        frame_length=PYIN_FRAME_LENGTH,
        hop_length=samples_per_frame,
    )
    voiced = voiced_flag & np.isfinite(f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    if len(f0) < n_frames:
        pad = n_frames - len(f0)
        f0 = np.pad(f0, (0, pad))
        voiced = np.pad(voiced, (0, pad))
    f0, voiced = f0[:n_frames], voiced[:n_frames]
    return fill_unvoiced(f0, voiced, PYIN_FMIN, PYIN_FMAX)


def _pyin_job(args: tuple[np.ndarray, int, int]) -> np.ndarray:
    return pyin_f0(*args)


class SamuelCodec(LatentCodec):
    """Audio <-> Pink Trombone parameter trajectories via a frozen Samuel checkpoint.

    Latent frame = the controller's trainable parameters scaled to [-1, 1] by
    their ranges, followed by log2(f0 / f0_ref). ``stack`` consecutive control
    frames are concatenated into one latent, lowering the frame rate.
    """

    stitch_frames = 0  # the controller is causal: latents after a cut are exact

    def __init__(
        self,
        checkpoint: str | PinkTromboneController = "hf:vvolhejn/samuel",
        stack: int = 1,
        f0_ref: float | None = None,
        pitch_workers: int | None = None,
        ir_length: int | None = None,
        target_rms: float | None = None,
    ):
        super().__init__()
        if stack < 1:
            raise ValueError(f"stack must be >= 1, got {stack}")
        if isinstance(checkpoint, PinkTromboneController):
            self.controller, run_cfg = checkpoint.eval(), {}
            if target_rms is None:
                raise ValueError("target_rms is required with an in-memory controller")
        else:
            self.controller, run_cfg = load_controller(checkpoint)
        for p in self.controller.parameters():
            p.requires_grad_(False)
        cfg = self.controller.config
        self.stack = stack
        self.samples_per_frame = cfg.samples_per_frame
        self.sample_rate = SAMPLE_RATE
        self.frame_rate = SAMPLE_RATE / (self.samples_per_frame * stack)
        self.target_rms = float(
            target_rms if target_rms is not None else run_cfg["data"]["target_rms"]
        )
        self.ir_length = ir_length or int(
            run_cfg.get("synth", {}).get("ir_length", 256)
        )
        self.f0_ref = f0_ref or math.sqrt(PYIN_FMIN * PYIN_FMAX)
        self.pitch_workers = pitch_workers or max(1, len(os.sched_getaffinity(0)) - 2)
        self._pool: Pool | None = None

        self.trainable_names = cfg.trainable_names()
        self.n_trainable = len(self.trainable_names)
        self.latent_dim = (self.n_trainable + 1) * stack
        lo = torch.tensor([cfg.param_spec[n][0] for n in self.trainable_names])
        hi = torch.tensor([cfg.param_spec[n][1] for n in self.trainable_names])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        self.register_buffer(
            "trainable_idx",
            torch.tensor([PARAM_NAMES.index(n) for n in self.trainable_names]),
        )
        frozen = sorted(cfg.frozen_values.items())
        self.register_buffer(
            "frozen_idx", torch.tensor([PARAM_NAMES.index(n) for n, _ in frozen])
        )
        self.register_buffer("frozen_vals", torch.tensor([v for _, v in frozen]))
        self.freq_idx = PARAM_NAMES.index("frequency")

    # -- pitch ------------------------------------------------------------

    def _pitch_pool(self) -> Pool:
        # multiprocessing.Pool workers are daemonic, so they die with the
        # interpreter; ProcessPoolExecutor workers kept finished runs alive.
        if self._pool is None:
            self._pool = get_context("spawn").Pool(processes=self.pitch_workers)
        return self._pool

    def _f0(self, wav: Tensor, lengths: Tensor, n_frames: int) -> Tensor:
        """[B, S] audio -> [B, n_frames] filled f0 in Hz (pyin on CPU workers)."""
        rows = wav.detach().float().cpu().numpy()
        jobs = [
            (rows[b, : int(lengths[b])], self.samples_per_frame, n_frames)
            for b in range(rows.shape[0])
        ]
        if len(jobs) == 1:
            tracks = [_pyin_job(jobs[0])]
        else:
            tracks = self._pitch_pool().map(_pyin_job, jobs)
        return torch.from_numpy(np.stack(tracks)).to(wav.device)

    # -- codec interface --------------------------------------------------

    @staticmethod
    def _valid_lengths(wav: Tensor) -> Tensor:
        """Samples up to the last non-zero one per row (batches are zero-padded)."""
        nonzero = wav != 0
        last = wav.shape[-1] - nonzero.flip(-1).float().argmax(-1)
        return torch.where(nonzero.any(-1), last, torch.ones_like(last))

    @torch.no_grad()
    def encode_to_latent(self, audio: Tensor) -> Tensor:
        S = audio.shape[-1]
        wav = audio[:, 0].float()
        lengths = self._valid_lengths(wav)
        valid = torch.arange(S, device=wav.device)[None, :] < lengths[:, None]
        rms = ((wav.pow(2) * valid).sum(-1) / lengths).clamp_min(1e-12).sqrt()
        wav = wav * (self.target_rms / rms)[:, None]

        n_frames = self.controller.t_ctrl_for(S)
        f0 = self._f0(wav, lengths, n_frames)
        params = self.controller(wav[:, None], f0)  # [B, T, N_PARAMS]
        scaled = (params[..., self.trainable_idx] - self.lo) / (
            self.hi - self.lo
        ) * 2 - 1
        log_f0 = torch.log2(f0 / self.f0_ref)
        lat = torch.cat([scaled, log_f0[..., None]], dim=-1)  # [B, T, n_trainable + 1]
        if self.stack > 1:
            pad = (-lat.shape[1]) % self.stack
            if pad:
                lat = torch.cat([lat, lat[:, -1:].expand(-1, pad, -1)], dim=1)
            lat = rearrange(lat, "b (t k) c -> b t (k c)", k=self.stack)
        return lat

    def latents_to_params(self, latents: Tensor) -> Tensor:
        """[B, T, C] latents -> [B, T * stack, N_PARAMS] Pink Trombone parameters."""
        if self.stack > 1:
            latents = rearrange(latents, "b t (k c) -> b (t k) c", k=self.stack)
        B, T, _ = latents.shape
        scaled = latents[..., : self.n_trainable].clamp(-1, 1)
        values = self.lo + (scaled + 1) / 2 * (self.hi - self.lo)
        f0 = (self.f0_ref * torch.exp2(latents[..., self.n_trainable])).clamp(
            PYIN_FMIN, PYIN_FMAX
        )
        out = torch.zeros(B, T, N_PARAMS, device=latents.device, dtype=values.dtype)
        out[..., self.trainable_idx] = values
        out[..., self.frozen_idx] = self.frozen_vals.to(values.dtype)
        out[..., self.freq_idx] = f0
        return out

    @torch.no_grad()
    def decode_to_audio(self, latents: Tensor) -> Tensor:
        params = self.latents_to_params(latents.float())
        audio = pink_trombone_ola(
            params,
            seed=0,
            ir_length=self.ir_length,
            control_rate=SAMPLE_RATE / self.samples_per_frame,
        )
        return audio[:, None]

    def encode_hash(self) -> str:
        h = hashlib.sha256()
        h.update(self.controller.config.model_dump_json().encode())
        for k, v in sorted(self.controller.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        h.update(
            json.dumps(
                {
                    "stack": self.stack,
                    "f0_ref": self.f0_ref,
                    "target_rms": self.target_rms,
                    "pyin": [PYIN_FMIN, PYIN_FMAX, PYIN_FRAME_LENGTH],
                }
            ).encode()
        )
        return h.hexdigest()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pool"] = None
        return state
