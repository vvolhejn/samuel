"""Precompute a causal YIN pitch cache at the model's control rate.

YIN (de Cheveigné & Kawahara 2002) on a trailing window, decoded by causal
forward filtering instead of the classic first-trough rule: each frame's CMNDF
troughs become observation likelihoods over a BIN_CENTS log-f0 grid, a
Gaussian transition prior (``--transition-sigma-cents`` per frame, plus a
small uniform jump term) carries the pitch posterior forward, and the frame's
f0 is the trough nearest the filtered argmax. The filter only ever sees past
and current frames, so every frame is final the moment its samples have
arrived and a streaming implementation reproduces this cache exactly.

Frame ``t`` reads samples ``[t * spf, t * spf + FRAME_LENGTH)``; the stored
track is shifted by ``SHIFT_FRAMES`` so cache frame ``t`` only depends on
samples up to its deadline ``(t + 1) * spf``. The first SHIFT_FRAMES frames
are stored unvoiced.

Implemented directly (sub-0.1-cent exact on steady tones) because librosa
exposes neither the CMNDF trough depths needed for the voicing decision and
observation likelihoods nor control over the integration window that bounds
the causal dependency horizon.

Usage:
    uv run python scripts/precompute_pitch_yin.py \\
        --manifest manifests/librilight_10h.jsonl \\
        --out manifests/pitch_cache/librilight_10h_yin_spf512.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import betainc
from tqdm import tqdm

from samuel.data import _load_resampled, load_manifest

# Trailing analysis window per frame. With integration window W = 1024 and
# max lag 630 (70 Hz at 44.1 kHz) every sample the frame reads lies within
# FRAME_LENGTH = 2048 of its start, i.e. within the deadline of control frame
# t + SHIFT_FRAMES.
FRAME_LENGTH = 2048
YIN_WINDOW = 1024
SHIFT_FRAMES = 3

# CMNDF troughs below this depth enter the filter as pitch candidates;
# voicing uses the (stricter) --voiced-threshold on the frame's deepest one.
# A trough's observation weight is the probability that it is YIN's pick (the
# first trough below threshold s) with s ~ Beta(2, 18), mean 0.1 — pYIN's
# soft first-trough rule. A plain depth-based weight locks onto subharmonics
# an octave down; a uniform threshold distribution overweights shallow early
# troughs an octave up.
CANDIDATE_THRESHOLD = 0.5
THRESHOLD_BETA_A = 2.0
THRESHOLD_BETA_B = 18.0
BIN_CENTS = 20.0
OBS_UNIFORM_MIX = 0.02
JUMP_PROB = 0.01  # per-frame probability of an arbitrary pitch jump
DECODE_WINDOW_BINS = 5  # snap the filtered argmax to a trough within ±100 cents


@dataclass
class Job:
    index: int
    path: str
    sample_rate: int
    samples_per_frame: int
    fmin: float
    fmax: float
    voiced_threshold: float
    transition_sigma_cents: float


def cmndf_frames(
    audio: np.ndarray,
    sample_rate: int,
    hop: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, int, int]:
    """CMNDF matrix ``[T, tau_max + 2]`` for trailing windows at ``hop``.

    Frame ``k`` reads samples ``[k * hop, k * hop + FRAME_LENGTH)``; the tail
    is zero-padded. One lag past tau_max is included so troughs at tau_max
    have a right neighbour.
    """
    T = -(-len(audio) // hop)
    tau_min = int(np.ceil(sample_rate / fmax))
    tau_max = int(np.floor(sample_rate / fmin))
    n_tau = tau_max + 2
    assert YIN_WINDOW + n_tau - 1 <= FRAME_LENGTH

    padded = np.zeros((T - 1) * hop + FRAME_LENGTH, dtype=np.float32)
    padded[: len(audio)] = audio
    frames = sliding_window_view(padded, FRAME_LENGTH)[::hop]  # [T, FRAME_LENGTH]

    # difference function d(tau) = e0 + e_tau - 2 * corr(tau), via FFT
    n_fft = 2 * FRAME_LENGTH
    head = frames[:, :YIN_WINDOW]
    spec = np.fft.rfft(frames, n_fft)
    corr = np.fft.irfft(spec * np.conj(np.fft.rfft(head, n_fft)), n_fft)
    corr = corr[:, :n_tau]  # c(tau) = sum_j head[j] * frame[j + tau]

    sq = np.cumsum(frames.astype(np.float64) ** 2, axis=1)
    e0 = sq[:, YIN_WINDOW - 1 : YIN_WINDOW]  # [T, 1]
    taus = np.arange(n_tau)
    e_tau = sq[:, taus + YIN_WINDOW - 1] - np.where(taus > 0, sq[:, taus - 1], 0.0)
    d = np.maximum(e0 + e_tau - 2.0 * corr, 0.0)

    # cumulative-mean-normalized difference
    cmndf = np.ones_like(d)
    running = np.cumsum(d[:, 1:], axis=1)
    cmndf[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(running, 1e-12)
    return cmndf, tau_min, tau_max


def filtered_track(
    cmndf: np.ndarray,
    tau_min: int,
    tau_max: int,
    sample_rate: int,
    fmin: float,
    fmax: float,
    sigma_cents: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame (f0_hz, deepest_trough_depth) via causal forward filtering."""
    T = cmndf.shape[0]
    n_bins = int(1200.0 * np.log2(fmax / fmin) / BIN_CENTS) + 1
    uniform = 1.0 / n_bins

    # candidate troughs: local minima of the CMNDF below CANDIDATE_THRESHOLD
    inner = cmndf[:, tau_min : tau_max + 1]
    is_trough = (
        (inner < cmndf[:, tau_min - 1 : tau_max])
        & (inner <= cmndf[:, tau_min + 1 : tau_max + 2])
        & (inner < CANDIDATE_THRESHOLD)
    )
    tt, rel = np.nonzero(is_trough)
    tau = rel + tau_min
    depth = cmndf[tt, tau]

    # parabolic interpolation around each trough
    y0, y2 = cmndf[tt, tau - 1], cmndf[tt, tau + 1]
    denom = y0 - 2.0 * depth + y2
    safe = np.abs(denom) > 1e-12
    offset = np.where(safe, 0.5 * (y0 - y2) / np.where(safe, denom, 1.0), 0.0)
    offset = np.clip(offset, -0.5, 0.5)
    f0_cand = sample_rate / (tau + offset)

    keep = (f0_cand >= fmin) & (f0_cand <= fmax)
    tt, depth, f0_cand = tt[keep], depth[keep], f0_cand[keep]
    bins = (1200.0 * np.log2(f0_cand / fmin) / BIN_CENTS).round().astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)

    # soft first-trough weights: trough j (in tau order, i.e. descending f0)
    # is YIN's pick for thresholds in (depth_j, min over earlier depths), so
    # its weight is the threshold distribution's mass on that interval
    starts = np.r_[True, tt[1:] != tt[:-1]] if len(tt) else np.zeros(0, bool)
    idx = np.arange(len(tt))
    group_start = np.maximum.accumulate(np.where(starts, idx, 0))
    k = idx - group_start  # rank of the trough within its frame
    k_max = int(k.max()) + 1 if len(k) else 1
    depth2d = np.full((T, k_max + 1), np.inf)
    depth2d[tt, k + 1] = depth
    prefix_min = np.minimum.accumulate(depth2d, axis=1)[:, :-1]  # exclusive
    ceiling = prefix_min[tt, k]
    cdf = lambda s: betainc(  # noqa: E731
        THRESHOLD_BETA_A, THRESHOLD_BETA_B, np.clip(s, 0.0, 1.0)
    )
    weight = np.maximum(cdf(ceiling) - cdf(depth), 0.0)

    # per (frame, bin): summed observation weight, f0 of the strongest trough
    depth_at = np.full((T, n_bins), np.inf)
    np.minimum.at(depth_at, (tt, bins), depth)
    obs = np.zeros((T, n_bins))
    np.add.at(obs, (tt, bins), weight)
    f0_at = np.full((T, n_bins), np.nan)
    order = np.lexsort((-weight, bins, tt))[::-1]  # strongest per bin last
    f0_at[tt[order], bins[order]] = f0_cand[order]

    # normalise observations, mixed with a uniform floor
    obs_sum = obs.sum(axis=1, keepdims=True)
    obs = np.where(
        obs_sum > 0.0,
        (1.0 - OBS_UNIFORM_MIX) * obs / np.maximum(obs_sum, 1e-300)
        + OBS_UNIFORM_MIX * uniform,
        uniform,
    )

    # transition prior: Gaussian in log-f0 plus a uniform jump term
    grid = np.arange(n_bins) * BIN_CENTS
    gauss = np.exp(-0.5 * ((grid[:, None] - grid[None, :]) / sigma_cents) ** 2)
    gauss /= gauss.sum(axis=0, keepdims=True)  # A[i, j] = P(bin i | bin j)
    trans = (1.0 - JUMP_PROB) * gauss + JUMP_PROB * uniform

    alpha = np.full(n_bins, uniform)
    choice = np.empty(T, dtype=np.int64)
    for t in range(T):
        alpha = obs[t] * (trans @ alpha)
        alpha /= alpha.sum()
        choice[t] = alpha.argmax()

    # decode: nearest trough to the filtered argmax, else the deepest trough
    off = np.arange(-DECODE_WINDOW_BINS, DECODE_WINDOW_BINS + 1)
    win = choice[:, None] + off[None, :]
    win_clipped = np.clip(win, 0, n_bins - 1)
    rows = np.arange(T)
    cand = f0_at[rows[:, None], win_clipped]
    ok = np.isfinite(cand) & (win == win_clipped)
    dist = np.where(ok, np.abs(off)[None, :], n_bins)
    pick = dist.argmin(axis=1)
    picked = cand[rows, pick]
    has_pick = ok[rows, pick]

    deepest_bin = depth_at.argmin(axis=1)
    deepest_depth = depth_at[rows, deepest_bin]
    deepest_f0 = f0_at[rows, deepest_bin]  # nan when the frame has no trough
    f0 = np.where(has_pick, picked, deepest_f0)
    f0 = np.where(np.isfinite(f0), f0, 0.0)
    return f0.astype(np.float32), deepest_depth.astype(np.float32)


def compute_file(job: Job) -> tuple[int, np.ndarray, np.ndarray]:
    try:
        audio = _load_resampled(Path(job.path), job.sample_rate)
    except Exception as e:  # noqa: BLE001
        print(f"skip {job.path}: {e}")
        return job.index, np.zeros(0, np.float32), np.zeros(0, bool)

    hop = job.samples_per_frame
    T = -(-len(audio) // hop)
    cmndf, tau_min, tau_max = cmndf_frames(
        audio, job.sample_rate, hop, job.fmin, job.fmax
    )
    f0_raw, depth = filtered_track(
        cmndf,
        tau_min,
        tau_max,
        job.sample_rate,
        job.fmin,
        job.fmax,
        job.transition_sigma_cents,
    )

    f0 = np.zeros(T, dtype=np.float32)
    voiced = np.zeros(T, dtype=bool)
    t = np.arange(SHIFT_FRAMES, T)
    k = t - SHIFT_FRAMES
    f0[t] = f0_raw[k]
    voiced[t] = depth[k] < job.voiced_threshold
    voiced &= (f0 >= job.fmin) & (f0 <= job.fmax)
    f0[~voiced] = 0.0
    return job.index, f0, voiced


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
        default=0.2,
        help="frames whose deepest CMNDF trough is below this are marked voiced",
    )
    ap.add_argument(
        "--transition-sigma-cents",
        type=float,
        default=25.0,
        help="stddev of the per-frame Gaussian pitch-transition prior",
    )
    ap.add_argument("--num-workers", type=int, default=16)
    args = ap.parse_args()

    files = load_manifest(args.manifest)
    jobs = [
        Job(
            index=df.index_in_manifest,
            path=str(df.path),
            sample_rate=args.sample_rate,
            samples_per_frame=args.samples_per_frame,
            fmin=args.fmin,
            fmax=args.fmax,
            voiced_threshold=args.voiced_threshold,
            transition_sigma_cents=args.transition_sigma_cents,
        )
        for df in files
    ]

    out: dict[str, np.ndarray] = {}
    n_voiced = n_total = 0
    with Pool(args.num_workers) as pool:
        for i, f0, voiced in tqdm(
            pool.imap_unordered(compute_file, jobs, chunksize=4),
            total=len(jobs),
            desc="yin",
        ):
            out[f"f0_{i}"] = f0
            out[f"voiced_{i}"] = voiced
            n_voiced += int(voiced.sum())
            n_total += len(voiced)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out["sample_rate"] = np.array(args.sample_rate)
    out["samples_per_frame"] = np.array(args.samples_per_frame)
    out["control_rate"] = np.array(
        args.sample_rate / args.samples_per_frame, dtype=np.float64
    )
    out["source"] = np.array("yin")
    out["fmin"] = np.array(args.fmin, dtype=np.float64)
    out["fmax"] = np.array(args.fmax, dtype=np.float64)
    out["yin_frame_length"] = np.array(FRAME_LENGTH)
    out["yin_window"] = np.array(YIN_WINDOW)
    out["yin_shift_frames"] = np.array(SHIFT_FRAMES)
    out["yin_voiced_threshold"] = np.array(args.voiced_threshold)
    out["yin_candidate_threshold"] = np.array(CANDIDATE_THRESHOLD)
    out["yin_bin_cents"] = np.array(BIN_CENTS)
    out["yin_transition_sigma_cents"] = np.array(args.transition_sigma_cents)
    out["yin_jump_prob"] = np.array(JUMP_PROB)
    out["n_files"] = np.array(len(files))
    np.savez_compressed(args.out, **out)
    print(
        f"saved {args.out}  ({len(files)} files, voiced fraction "
        f"{n_voiced / max(n_total, 1):.3f})"
    )


if __name__ == "__main__":
    main()
