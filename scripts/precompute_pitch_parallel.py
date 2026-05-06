"""Parallel pyin pitch precompute. Drop-in replacement for precompute_pitch.py
that fans the per-file pyin work out to a process pool."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from samuel.data import _load_resampled, load_manifest


def _worker(args):
    i, path_str, sample_rate, samples_per_frame, fmin, fmax, frame_length = args
    try:
        audio = _load_resampled(Path(path_str), sample_rate)
    except Exception as e:  # noqa: BLE001
        return i, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool), str(e)
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=samples_per_frame,
    )
    voiced = voiced_flag & np.isfinite(f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0).astype(np.float32)
    return i, f0, voiced.astype(bool), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--samples-per-frame", type=int, default=2048)
    ap.add_argument("--fmin", type=float, default=70.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument("--frame-length", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    args = ap.parse_args()

    if args.frame_length <= args.samples_per_frame:
        raise SystemExit(
            f"frame-length ({args.frame_length}) must be > samples-per-frame "
            f"({args.samples_per_frame})"
        )

    files = load_manifest(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (
            i,
            str(df.path),
            args.sample_rate,
            args.samples_per_frame,
            args.fmin,
            args.fmax,
            args.frame_length,
        )
        for i, df in enumerate(files)
    ]

    out: dict[str, np.ndarray] = {}
    n_err = 0
    print(f"running pyin with {args.workers} workers on {len(files)} files")
    with mp.get_context("spawn").Pool(processes=args.workers) as pool:
        for i, f0, voiced, err in tqdm(
            pool.imap_unordered(_worker, tasks, chunksize=8),
            total=len(tasks),
            desc="pyin",
        ):
            if err is not None:
                n_err += 1
                if n_err <= 10:
                    print(f"skip idx={i}: {err}")
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
        f"saved {args.out}  ({len(files)} files, {n_err} errors, "
        f"sr={args.sample_rate}, hop={args.samples_per_frame}, "
        f"control_rate={args.sample_rate / args.samples_per_frame:.3f} Hz)"
    )


if __name__ == "__main__":
    main()
