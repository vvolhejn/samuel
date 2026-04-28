"""Precompute pyin pitch at the model's control rate for all manifest entries.

Saves one cache file per manifest, containing per-file (f0, voiced) arrays
sampled at control_rate Hz. The dataloader reads this cache and slices
alongside each audio chunk it yields.

Usage:
    uv run python scripts/precompute_pitch.py \\
        --manifest manifests/librilight_10h.jsonl \\
        --out manifests/pitch_cache/librilight_10h.npz \\
        --samples-per-frame 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from samuel.data import _load_resampled, load_manifest


def compute_pitch(
    audio: np.ndarray,
    sample_rate: int,
    samples_per_frame: int,
    fmin: float,
    fmax: float,
    frame_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """pyin pitch at hop=samples_per_frame. Returns (f0_hz, voiced_mask).

    Voicing comes from librosa's Viterbi-decoded ``voiced_flag`` (equivalently,
    the frames where ``f0`` is finite). ``voiced_prob`` is a smoothed transition
    probability that mostly sits below 0.5 even on confidently voiced frames,
    so thresholding it is much stricter than what pyin actually decides.
    """
    f0, voiced_flag, _voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=samples_per_frame,
    )
    voiced = voiced_flag & np.isfinite(f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    return f0, voiced.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument(
        "--samples-per-frame",
        type=int,
        default=2048,
        help="pyin hop length; matches the model's control hop in samples.",
    )
    ap.add_argument("--fmin", type=float, default=70.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument(
        "--frame-length",
        type=int,
        default=4096,
        help="pyin frame_length; must be > samples-per-frame.",
    )
    args = ap.parse_args()

    if args.frame_length <= args.samples_per_frame:
        raise SystemExit(
            f"frame-length ({args.frame_length}) must be > samples-per-frame "
            f"({args.samples_per_frame})"
        )

    files = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    for i, df in enumerate(tqdm(files, desc="pyin")):
        try:
            audio = _load_resampled(df.path, args.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"skip {df.path}: {e}")
            out[f"f0_{i}"] = np.zeros(0, dtype=np.float32)
            out[f"voiced_{i}"] = np.zeros(0, dtype=bool)
            continue
        f0, voiced = compute_pitch(
            audio,
            sample_rate=args.sample_rate,
            samples_per_frame=args.samples_per_frame,
            fmin=args.fmin,
            fmax=args.fmax,
            frame_length=args.frame_length,
        )
        out[f"f0_{i}"] = f0
        out[f"voiced_{i}"] = voiced

    out["sample_rate"] = np.array(args.sample_rate)
    out["samples_per_frame"] = np.array(args.samples_per_frame)
    out["control_rate"] = np.array(
        args.sample_rate / args.samples_per_frame, dtype=np.float64
    )
    out["pyin_fmin"] = np.array(args.fmin, dtype=np.float64)
    out["pyin_fmax"] = np.array(args.fmax, dtype=np.float64)
    out["pyin_frame_length"] = np.array(args.frame_length)
    out["n_files"] = np.array(len(files))
    np.savez_compressed(args.out, **out)
    print(
        f"saved {args.out}  ({len(files)} files, "
        f"sr={args.sample_rate}, hop={args.samples_per_frame}, "
        f"control_rate={args.sample_rate / args.samples_per_frame:.3f} Hz)"
    )


if __name__ == "__main__":
    main()
