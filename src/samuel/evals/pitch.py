"""Pitch MAE metric over voiced frames, reported in cents.

Uses ``librosa.pyin`` on CPU. Unvoiced frames in the *target* are ignored;
voiced-target frames where the *prediction* is unvoiced are reported
separately as ``unvoiced_miss_frac`` so the MAE stays well-defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class PitchMetrics:
    mae_cents: float  # mean |cents error| over voiced-target & voiced-pred frames
    unvoiced_miss_frac: float  # fraction of voiced-target frames where pred is unvoiced
    n_voiced_frames: int


def pitch_track(
    audio: np.ndarray,
    sample_rate: int,
    fmin: float = 70.0,
    fmax: float = 500.0,
    voiced_prob_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (f0_hz, voiced_mask) aligned per-frame. Both shape [n_frames]."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio, fmin=fmin, fmax=fmax, sr=sample_rate
    )
    voiced_mask = (voiced_prob >= voiced_prob_threshold) & np.isfinite(f0)
    return f0, voiced_mask


def pitch_mae_cents(
    target: np.ndarray,
    pred: np.ndarray,
    sample_rate: int,
    fmin: float = 70.0,
    fmax: float = 500.0,
    voiced_prob_threshold: float = 0.5,
) -> PitchMetrics:
    """Pitch MAE (cents) for voiced-target frames.

    ``target`` and ``pred`` are 1-D float arrays at ``sample_rate``. pyin is
    run with the same params on both so frame counts align.
    """
    f0_t, voiced_t = pitch_track(
        target,
        sample_rate,
        fmin=fmin,
        fmax=fmax,
        voiced_prob_threshold=voiced_prob_threshold,
    )
    f0_p, voiced_p = pitch_track(
        pred,
        sample_rate,
        fmin=fmin,
        fmax=fmax,
        voiced_prob_threshold=voiced_prob_threshold,
    )
    m = min(len(f0_t), len(f0_p))
    f0_t, voiced_t = f0_t[:m], voiced_t[:m]
    f0_p, voiced_p = f0_p[:m], voiced_p[:m]

    n_voiced_target = int(voiced_t.sum())
    if n_voiced_target == 0:
        return PitchMetrics(
            mae_cents=float("nan"), unvoiced_miss_frac=float("nan"), n_voiced_frames=0
        )

    miss = voiced_t & ~voiced_p
    unvoiced_miss_frac = float(miss.sum()) / float(n_voiced_target)

    both = voiced_t & voiced_p
    if both.sum() == 0:
        return PitchMetrics(
            mae_cents=float("nan"),
            unvoiced_miss_frac=unvoiced_miss_frac,
            n_voiced_frames=n_voiced_target,
        )
    cents = 1200.0 * np.log2(f0_p[both] / f0_t[both])
    mae = float(np.mean(np.abs(cents)))
    return PitchMetrics(
        mae_cents=mae,
        unvoiced_miss_frac=unvoiced_miss_frac,
        n_voiced_frames=n_voiced_target,
    )
