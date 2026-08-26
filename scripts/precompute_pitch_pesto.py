"""Precompute a causal PESTO pitch cache at the model's control rate.

Uses the vendored streaming PESTO (third_party/pesto, upstream `streaming`
branch): a cached-convolution CQT fed in samples_per_frame-sized chunks — one
frame per call — so frame ``t`` depends on nothing past its deadline
``(t + 1) * samples_per_frame`` (verified by truncation).

With the default ``--mirror 0`` each window contains only real past samples
(fully trailing). This matches offline PESTO's accuracy on speech but the
estimate lags the signal by ~5-6 control frames (the CQT kernels' group
delay). ``--mirror 1`` recenters the window on "now" by faking the future
half (the TISMIR 2025 "buffer refilling"), which removes the lag but is much
less accurate here: the shipped mir-1k_g7 checkpoint was not retrained on
refilled buffers (measured ~3x the octave-error rate of trailing windows).

Usage:
    uv run python scripts/precompute_pitch_pesto.py \\
        --manifest manifests/librilight_10h.jsonl \\
        --out manifests/pitch_cache/librilight_10h_pesto_spf512.npz
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pesto
import torch
from tqdm import tqdm

from samuel.data import _load_resampled, load_manifest

CHECKPOINT = "mir-1k_g7"
# The streaming CQT warms up from empty causal caches; frames computed while
# the longest CQT kernel still overlaps the initial zeros are meaningless.
WARMUP_FRAMES = 16


def _load_one(job: tuple[str, int, int]) -> tuple[int, np.ndarray | None]:
    """Load and resample one file in a worker; None marks a failed load."""
    path, sample_rate, index = job
    try:
        return index, _load_resampled(Path(path), sample_rate)
    except Exception as e:  # noqa: BLE001
        print(f"skip {path}: {e}")
        return index, None


def load_streaming_model(
    step_ms: float,
    sample_rate: int,
    batch_size: int,
    mirror: float,
    device: torch.device,
) -> torch.nn.Module:
    model = pesto.load_model(
        CHECKPOINT,
        step_size=step_ms,
        sampling_rate=sample_rate,
        streaming=True,
        mirror=mirror,
        mirror_fn="refill",
        max_batch_size=batch_size,
    )
    return model.to(device).eval()


@torch.inference_mode()
def process_batch(
    model: torch.nn.Module,
    audios: list[np.ndarray],
    spf: int,
    device: torch.device,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stream a batch of files chunk by chunk; returns (f0_hz, confidence) per file."""
    lengths = [-(-len(a) // spf) for a in audios]  # ceil -> frames per file
    T = max(lengths)
    x = torch.zeros(len(audios), T * spf, device=device)
    for b, a in enumerate(audios):
        x[b, : len(a)] = torch.from_numpy(a).to(device)

    preds, confs = [], []
    for chunk in x.split(spf, dim=-1):
        p, c, _vol = model(chunk, convert_to_freq=True, return_activations=False)
        preds.append(p.reshape(-1))
        confs.append(c.reshape(-1))
    f0 = torch.stack(preds, dim=1).float().cpu().numpy()  # [B, T]
    conf = torch.stack(confs, dim=1).float().cpu().numpy()
    return [(f0[b, : lengths[b]], conf[b, : lengths[b]]) for b in range(len(audios))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--samples-per-frame", type=int, default=512)
    ap.add_argument("--fmin", type=float, default=70.0)
    ap.add_argument("--fmax", type=float, default=500.0)
    ap.add_argument(
        "--voiced-threshold",
        type=float,
        default=0.5,
        help="frames with PESTO confidence above this are marked voiced",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument(
        "--load-workers",
        type=int,
        default=8,
        help="processes decoding audio; the next batch loads while the "
        "current one runs on the GPU",
    )
    ap.add_argument(
        "--mirror",
        type=float,
        default=0.0,
        help="fraction of the window's future half faked by buffer refilling; "
        "0 = fully trailing real samples (see module docstring)",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    step_ms = 1000.0 * args.samples_per_frame / args.sample_rate
    files = load_manifest(args.manifest)

    batches = [
        files[start : start + args.batch_size]
        for start in range(0, len(files), args.batch_size)
    ]
    pool = Pool(args.load_workers)

    def submit(batch):  # prefetch: decode audio while the GPU streams
        jobs = [(str(df.path), args.sample_rate, df.index_in_manifest) for df in batch]
        return pool.map_async(_load_one, jobs)

    out: dict[str, np.ndarray] = {}
    n_voiced = n_total = 0
    pbar = tqdm(total=len(files), desc="pesto")
    pending = submit(batches[0]) if batches else None
    for b, batch in enumerate(batches):
        loaded = pending.get()
        if b + 1 < len(batches):
            pending = submit(batches[b + 1])

        audios: list[np.ndarray] = []
        indices: list[int] = []
        for index, audio in loaded:
            if audio is None:
                out[f"f0_{index}"] = np.zeros(0, np.float32)
                out[f"voiced_{index}"] = np.zeros(0, bool)
            else:
                audios.append(audio)
                indices.append(index)
        if not audios:
            pbar.update(len(batch))
            continue

        # Fresh model per batch: the streaming caches are stateful and there
        # is no reset API ("do not use the same model for different streams").
        model = load_streaming_model(
            step_ms, args.sample_rate, len(audios), args.mirror, device
        )
        results = process_batch(model, audios, args.samples_per_frame, device)
        for i, (f0, conf) in zip(indices, results):
            voiced = conf > args.voiced_threshold
            voiced[:WARMUP_FRAMES] = False
            # Only trust estimates inside the range the model's buckets cover.
            voiced &= (f0 >= args.fmin) & (f0 <= args.fmax)
            f0 = np.where(voiced, f0, 0.0).astype(np.float32)
            out[f"f0_{i}"] = f0
            out[f"voiced_{i}"] = voiced
            n_voiced += int(voiced.sum())
            n_total += len(voiced)
        pbar.update(len(batch))
        pbar.set_postfix(voiced_frac=f"{n_voiced / max(n_total, 1):.3f}")
    pbar.close()
    pool.close()
    pool.join()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out["sample_rate"] = np.array(args.sample_rate)
    out["samples_per_frame"] = np.array(args.samples_per_frame)
    out["control_rate"] = np.array(
        args.sample_rate / args.samples_per_frame, dtype=np.float64
    )
    out["source"] = np.array("pesto")
    out["fmin"] = np.array(args.fmin, dtype=np.float64)
    out["fmax"] = np.array(args.fmax, dtype=np.float64)
    out["pesto_checkpoint"] = np.array(CHECKPOINT)
    out["pesto_voiced_threshold"] = np.array(args.voiced_threshold)
    out["pesto_warmup_frames"] = np.array(WARMUP_FRAMES)
    out["pesto_mirror"] = np.array(args.mirror)
    out["n_files"] = np.array(len(files))
    np.savez_compressed(args.out, **out)
    print(
        f"saved {args.out}  ({len(files)} files, voiced fraction "
        f"{n_voiced / max(n_total, 1):.3f})"
    )


if __name__ == "__main__":
    main()
