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
from pydantic import BaseModel
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


def _load_resampled(path: Path, target_sr: int) -> np.ndarray:
    """Load a file and resample to ``target_sr`` mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype(np.float32, copy=False)


class LibriLightChunks(IterableDataset):
    """Non-overlapping fixed-length chunks over a jsonl manifest.

    DDP sharding: the manifest is split round-robin across ``world_size``
    ranks; inside a rank the per-worker split is also round-robin. Each epoch
    the rank's slice is shuffled with ``epoch_seed ^ rank``.
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
        control_rate: float = 12.5,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        self.chunk_samples = int(round(sample_rate * chunk_seconds))
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.seed = seed
        self.drop_last = drop_last
        self.control_rate = control_rate
        self.samples_per_pitch_frame = int(round(sample_rate / control_rate))
        self.pitch_frames_per_chunk = self.chunk_samples // self.samples_per_pitch_frame
        self._all_files = load_manifest(manifest_path)
        # Index map: original file index in manifest → file object (used to look
        # up pitch from the cache since pitch is keyed on manifest index).
        self._file_to_idx = {id(df): i for i, df in enumerate(self._all_files)}
        self._rank_files = _shard(self._all_files, rank, world_size)
        self.pitch_cache_path = pitch_cache_path
        self._pitch: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        if pitch_cache_path is not None:
            z = np.load(pitch_cache_path)
            self._pitch = {}
            n = int(z["n_files"])
            for i in range(n):
                self._pitch[i] = (z[f"f0_{i}"], z[f"voiced_{i}"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _worker_files(self) -> list[DatasetFile]:
        info = get_worker_info()
        if info is None:
            return list(self._rank_files)
        return self._rank_files[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        files = self._worker_files()
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = random.Random(
            self.seed ^ (self.epoch * 1315423911) ^ (self.rank * 2654435761) ^ wid
        )
        rng.shuffle(files)

        T_pitch = self.pitch_frames_per_chunk

        for df in files:
            try:
                audio = _load_resampled(df.path, self.sample_rate)
            except Exception:  # noqa: BLE001 - skip unreadable files
                continue
            file_idx = self._file_to_idx.get(id(df))
            pitch_f0 = pitch_voiced = None
            if self._pitch is not None and file_idx is not None:
                pitch_f0, pitch_voiced = self._pitch[file_idx]
            n = len(audio)
            for i in range(0, n, self.chunk_samples):
                chunk = audio[i : i + self.chunk_samples]
                if len(chunk) < self.chunk_samples:
                    if self.drop_last:
                        continue
                    pad = np.zeros(self.chunk_samples - len(chunk), dtype=np.float32)
                    chunk = np.concatenate([chunk, pad])

                if self._pitch is None:
                    yield torch.from_numpy(chunk)
                    continue

                p_start = i // self.samples_per_pitch_frame
                p_end = p_start + T_pitch
                if pitch_f0 is None or p_end > len(pitch_f0):
                    # Cache shorter than expected — pad with unvoiced zeros.
                    f0 = np.zeros(T_pitch, dtype=np.float32)
                    voiced = np.zeros(T_pitch, dtype=bool)
                    have = 0 if pitch_f0 is None else max(0, len(pitch_f0) - p_start)
                    if have > 0:
                        f0[:have] = pitch_f0[p_start : p_start + have]
                        voiced[:have] = pitch_voiced[p_start : p_start + have]
                else:
                    f0 = pitch_f0[p_start:p_end].astype(np.float32, copy=False)
                    voiced = pitch_voiced[p_start:p_end]
                yield {
                    "audio": torch.from_numpy(chunk),
                    "pitch": torch.from_numpy(f0),
                    "voiced": torch.from_numpy(voiced.astype(np.bool_)),
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
    control_rate: float = 12.5,
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
        control_rate=control_rate,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
