"""LibriLight iterable dataset.

Expects the jsonl manifest produced by
``nanoGPTaudio/data/librilight/subsample.py`` — one ``DatasetFile`` per line
with absolute paths.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class DatasetFile(BaseModel):
    path: Path
    duration: float | None = None
    sample_rate: int | None = None
    size_bytes: int | None = None


def load_manifest(path: Path) -> list[DatasetFile]:
    files: list[DatasetFile] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            files.append(DatasetFile(**json.loads(line)))
    return files


def _shard(files: list[DatasetFile], rank: int, world_size: int) -> list[DatasetFile]:
    return files[rank::world_size]


def split_train_val(
    files: list[DatasetFile], val_fraction: float
) -> tuple[list[DatasetFile], list[DatasetFile]]:
    """Last ``val_fraction`` of the manifest is the held-out validation set."""
    n = len(files)
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0 else 0
    return files[: n - n_val], files[n - n_val :] if n_val > 0 else []


def _load_resampled(path: Path, target_sr: int) -> np.ndarray:
    """Load a file and resample to ``target_sr`` mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


def fill_unvoiced(
    f0: np.ndarray,
    voiced: np.ndarray,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Linearly interpolate f0 across unvoiced runs.

    Interior runs are bridged between bracketing voiced f0 values; leading /
    trailing runs extend the nearest voiced value (numpy's ``interp`` clamps
    out-of-range x to the endpoint y); fully-unvoiced inputs fall back to
    ``(fmin + fmax) / 2``. Output is clamped to ``[fmin, fmax]``.
    """
    n = f0.shape[0]
    if n == 0:
        return f0.astype(np.float32, copy=True)
    if not voiced.any():
        return np.full(n, 0.5 * (fmin + fmax), dtype=np.float32)
    idx = np.arange(n)
    out = np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)
    return np.clip(out, fmin, fmax)


class PitchCache(BaseModel):
    """Per-file pyin f0 cache with the metadata it was computed under."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # {file_index: (f0_hz [n_frames], voiced_mask [n_frames])}
    by_file: dict[int, tuple[np.ndarray, np.ndarray]]
    sample_rate: int
    samples_per_frame: int
    fmin: float
    fmax: float


def _load_pitch_cache(
    path: Path,
    sample_rate: int,
    samples_per_frame: int,
) -> PitchCache:
    """Load a pitch cache, validating it matches (sample_rate, samples_per_frame)."""
    z = np.load(path)
    cache_sr = int(z["sample_rate"])
    cache_spf = int(z["samples_per_frame"])
    if cache_sr != sample_rate or cache_spf != samples_per_frame:
        raise ValueError(
            f"pitch cache {path} was computed at "
            f"sample_rate={cache_sr}, samples_per_frame={cache_spf} "
            f"but training expects sample_rate={sample_rate}, "
            f"samples_per_frame={samples_per_frame}. Regenerate with:\n"
            f"  uv run python scripts/precompute_pitch.py "
            f"--manifest <m> --out {path} "
            f"--sample-rate {sample_rate} --samples-per-frame {samples_per_frame}"
        )
    n = int(z["n_files"])
    by_file: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n):
        by_file[i] = (z[f"f0_{i}"], z[f"voiced_{i}"])
    return PitchCache(
        by_file=by_file,
        sample_rate=cache_sr,
        samples_per_frame=cache_spf,
        fmin=float(z["pyin_fmin"]),
        fmax=float(z["pyin_fmax"]),
    )


class LibriLightChunks(IterableDataset):
    """Non-overlapping fixed-length chunks over a jsonl manifest.

    DDP sharding: the manifest is split round-robin across ``world_size``
    ranks; inside a rank the per-worker split is also round-robin. Each epoch
    the rank's slice is shuffled with ``epoch_seed ^ rank``.

    Always yields ``{"audio": ...}`` dicts. When ``pitch_cache_path`` is
    given, the dict also has a ``"pitch"`` key with the per-chunk pyin f0
    (linearly interpolated through unvoiced regions) at ``samples_per_frame``
    hop.
    """

    def __init__(
        self,
        manifest_path: Path,
        sample_rate: int = 44100,
        chunk_seconds: float = 4.0,
        rank: int = 0,
        world_size: int = 1,
        epoch: int = 0,
        seed: int = 0,
        drop_last: bool = True,
        pitch_cache_path: Path | None = None,
        samples_per_frame: int | None = None,
        val_fraction: float = 0.0,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        chunk_samples = int(round(sample_rate * chunk_seconds))
        if samples_per_frame is not None:
            # Round down so chunks tile evenly into control frames.
            chunk_samples = (chunk_samples // samples_per_frame) * samples_per_frame
            if chunk_samples == 0:
                raise ValueError(
                    f"chunk_seconds={chunk_seconds} too small for "
                    f"samples_per_frame={samples_per_frame}"
                )
        self.chunk_samples = chunk_samples
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.seed = seed
        self.drop_last = drop_last
        full_manifest = load_manifest(manifest_path)
        # Pitch-cache lookup uses the full-manifest index, so build the id→idx
        # map *before* slicing off the val tail.
        self._idx_for_file: dict[int, int] = {
            id(df): i for i, df in enumerate(full_manifest)
        }
        train_files, _ = split_train_val(full_manifest, val_fraction)
        self._all_files = train_files
        self._rank_files = _shard(self._all_files, rank, world_size)

        self.pitch_cache_path = pitch_cache_path
        self.samples_per_frame = samples_per_frame
        self._pitch: PitchCache | None = None
        if pitch_cache_path is not None:
            if samples_per_frame is None:
                raise ValueError(
                    "samples_per_frame is required when pitch_cache_path is set"
                )
            self._pitch = _load_pitch_cache(
                pitch_cache_path, sample_rate, samples_per_frame
            )
            assert self.chunk_samples % samples_per_frame == 0
            self.pitch_frames_per_chunk = self.chunk_samples // samples_per_frame

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_files(self) -> list[DatasetFile]:
        info = get_worker_info()
        if info is None:
            return list(self._rank_files)
        return self._rank_files[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        files = self._worker_files()
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = random.Random(
            self.seed ^ (self.epoch * 1315423911) ^ (self.rank * 2654435761) ^ wid
        )
        rng.shuffle(files)

        spf = self.samples_per_frame
        T_pitch = self.pitch_frames_per_chunk if self._pitch is not None else 0
        fmin = self._pitch.fmin if self._pitch is not None else 70.0
        fmax = self._pitch.fmax if self._pitch is not None else 500.0

        for df in files:
            try:
                audio = _load_resampled(df.path, self.sample_rate)
            except Exception:  # noqa: BLE001 - skip unreadable files
                continue
            file_idx = self._idx_for_file.get(id(df))
            pitch_f0 = pitch_voiced = None
            if self._pitch is not None and file_idx is not None:
                pitch_f0, pitch_voiced = self._pitch.by_file[file_idx]
            n = len(audio)
            for i in range(0, n, self.chunk_samples):
                chunk = audio[i : i + self.chunk_samples]
                if len(chunk) < self.chunk_samples:
                    if self.drop_last:
                        continue
                    pad = np.zeros(self.chunk_samples - len(chunk), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad])

                if self._pitch is None:
                    yield {"audio": torch.from_numpy(chunk)}
                    continue

                p_start = i // spf  # type: ignore[operator]
                p_end = p_start + T_pitch
                if pitch_f0 is None or p_end > len(pitch_f0):
                    f0_chunk = np.zeros(T_pitch, dtype=np.float32)
                    voiced_chunk = np.zeros(T_pitch, dtype=bool)
                    have = 0 if pitch_f0 is None else max(0, len(pitch_f0) - p_start)
                    if have > 0:
                        f0_chunk[:have] = pitch_f0[p_start : p_start + have]
                        voiced_chunk[:have] = pitch_voiced[p_start : p_start + have]
                else:
                    f0_chunk = pitch_f0[p_start:p_end].astype(np.float32, copy=False)
                    voiced_chunk = pitch_voiced[p_start:p_end]
                f0_filled = fill_unvoiced(f0_chunk, voiced_chunk, fmin, fmax)
                yield {
                    "audio": torch.from_numpy(chunk),
                    "pitch": torch.from_numpy(f0_filled),
                }


def build_dataloader(
    manifest_path: Path,
    batch_size: int,
    num_workers: int = 4,
    sample_rate: int = 44100,
    chunk_seconds: float = 4.0,
    rank: int = 0,
    world_size: int = 1,
    epoch: int = 0,
    seed: int = 0,
    drop_last: bool = True,
    pin_memory: bool = True,
    pitch_cache_path: Path | None = None,
    samples_per_frame: int | None = None,
    val_fraction: float = 0.0,
) -> DataLoader:
    dataset = LibriLightChunks(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        chunk_seconds=chunk_seconds,
        rank=rank,
        world_size=world_size,
        epoch=epoch,
        seed=seed,
        drop_last=drop_last,
        pitch_cache_path=pitch_cache_path,
        samples_per_frame=samples_per_frame,
        val_fraction=val_fraction,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
