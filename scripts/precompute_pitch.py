"""Precompute pyin pitch at the model's control rate for all manifest entries.

Saves one cache file per manifest, containing per-file (f0, voiced) arrays
sampled at control_rate Hz. The dataloader reads this cache and slices
alongside each audio chunk it yields.

Usage:
    uv run python scripts/precompute_pitch.py \\
        --manifest manifests/librilight_10h.jsonl \\
        --out manifests/pitch_cache/librilight_10h.npz
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
    sample_rate: int = 44100,
    control_rate: float = 12.5,
    fmin: float = 70.0,
    fmax: float = 500.0,
    voiced_prob_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """pyin pitch at control_rate Hz. Returns (f0_hz, voiced_mask)."""
    hop = int(round(sample_rate / control_rate))  # 3528 for 44.1k @ 12.5 Hz
    # frame_length must be > hop; pick a power of two above hop
    frame_length = 4096
    f0, _voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop,
    )
    voiced = (voiced_prob >= voiced_prob_threshold) & np.isfinite(f0)
    # Replace NaN f0 with 0 (unvoiced); the mask tells us which frames are valid
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    return f0, voiced.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--control-rate", type=float, default=12.5)
    ap.add_argument("--fmin", type=float, default=70.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    args = ap.parse_args()

    files = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    f0_list: list[np.ndarray] = []
    voiced_list: list[np.ndarray] = []
    for df in tqdm(files, desc="pyin"):
        try:
            audio = _load_resampled(df.path, args.sample_rate)
        except Exception as e:  # noqa: BLE001
            print(f"skip {df.path}: {e}")
            f0_list.append(np.zeros(0, dtype=np.float32))
            voiced_list.append(np.zeros(0, dtype=bool))
            continue
        f0, voiced = compute_pitch(
            audio,
            sample_rate=args.sample_rate,
            control_rate=args.control_rate,
            fmin=args.fmin,
            fmax=args.fmax,
        )
        f0_list.append(f0)
        voiced_list.append(voiced)

    # Pack ragged arrays into a dict indexed by file index
    out = {}
    for i, (f0, voiced) in enumerate(zip(f0_list, voiced_list)):
        out[f"f0_{i}"] = f0
        out[f"voiced_{i}"] = voiced
    out["sample_rate"] = np.array(args.sample_rate)
    out["control_rate"] = np.array(args.control_rate)
    out["n_files"] = np.array(len(files))
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out}  ({len(files)} files)")


if __name__ == "__main__":
    main()
