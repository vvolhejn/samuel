"""Tests for LibriLightChunks."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from samuel.data import LibriLightChunks, _shard, load_manifest


def _write_wav(path: Path, duration_s: float, sr: int, seed: int = 0):
    n = int(duration_s * sr)
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n).astype(np.float32) * 0.1
    sf.write(str(path), y, sr)


def _build_manifest(tmp_path: Path, durations_s: list[float], sr: int) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    with open(manifest, "w") as f:
        for i, dur in enumerate(durations_s):
            wav = tmp_path / f"clip_{i}.wav"
            _write_wav(wav, dur, sr, seed=i)
            f.write(
                json.dumps(
                    {"path": str(wav.resolve()), "duration": dur, "sample_rate": sr}
                )
                + "\n"
            )
    return manifest


class TestLibriLightChunks:
    def test_manifest_roundtrip(self, tmp_path: Path):
        manifest = _build_manifest(tmp_path, [1.0, 2.0], sr=16000)
        files = load_manifest(manifest)
        assert len(files) == 2
        assert files[0].duration == 1.0
        assert files[0].sample_rate == 16000

    def test_chunk_shape_and_resample(self, tmp_path: Path):
        """Resamples 16 kHz wavs to 44.1 kHz and chunks at exactly chunk_seconds."""
        sr_in = 16000
        manifest = _build_manifest(tmp_path, [1.5, 1.5], sr=sr_in)
        ds = LibriLightChunks(
            manifest_path=manifest,
            sample_rate=44100,
            chunk_seconds=0.5,
            rank=0,
            world_size=1,
        )
        chunks = list(ds)
        expected_samples = int(round(44100 * 0.5))
        assert all(c.shape == (expected_samples,) for c in chunks)
        assert all(c.dtype == torch.float32 for c in chunks)
        # 1.5 s / 0.5 s = 3 chunks per file, 2 files -> 6 chunks
        assert len(chunks) == 6

    def test_rank_sharding(self, tmp_path: Path):
        manifest = _build_manifest(tmp_path, [0.5] * 8, sr=16000)
        files = load_manifest(manifest)
        r0 = _shard(files, 0, 2)
        r1 = _shard(files, 1, 2)
        assert len(r0) == 4 and len(r1) == 4
        assert set(f.path for f in r0) | set(f.path for f in r1) == set(
            f.path for f in files
        )
        assert not (set(f.path for f in r0) & set(f.path for f in r1))
