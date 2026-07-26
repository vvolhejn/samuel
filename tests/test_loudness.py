"""Tests for dataset loudness normalization (``data.target_rms``)."""

import numpy as np
import torch

from samuel.config import TrainConfig
from samuel.server import _rms_normalize as _rms_normalize_np
from samuel.train import _rms_normalize


class TestRmsNormalize:
    def test_hits_target_rms(self):
        wav = torch.randn(4, 8192) * torch.tensor([[0.01], [0.2], [1.0], [5.0]])
        out = _rms_normalize(wav, 0.05)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.full((4,), 0.05), atol=1e-6)

    def test_preserves_within_clip_contour(self):
        """Only a single per-clip gain: the energy envelope must survive.

        The whole point of normalising the data rather than gain-matching the
        synth output is that stop gaps and syllable dips stay visible to the
        loss, so the model has a reason to shape them via intensity.
        """
        wav = torch.randn(2, 8192)
        wav[:, 4096:] *= 0.01  # loud half, quiet half
        out = _rms_normalize(wav, 0.05)
        ratio = out / wav
        assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio), atol=1e-5)

    def test_handles_digital_silence(self):
        out = _rms_normalize(torch.zeros(1, 1024), 0.05)
        assert torch.isfinite(out).all()

    def test_numpy_and_torch_agree(self):
        """server._rms_normalize must reproduce training exactly."""
        wav = (np.random.default_rng(0).standard_normal(8192) * 0.7).astype(np.float32)
        np_out = _rms_normalize_np(wav, 0.05)
        torch_out = _rms_normalize(torch.from_numpy(wav)[None], 0.05)[0].numpy()
        assert np.allclose(np_out, torch_out, atol=1e-7)


class TestLoudnessConfig:
    def test_target_rms_default(self):
        cfg = TrainConfig.model_validate(
            {"run": {"name": "t"}, "data": {"manifest_path": "manifests/t.jsonl"}}
        )
        assert cfg.data.target_rms == 0.05
        # Nothing corrects the output level, so the model keeps a level knob.
        assert "intensity" in cfg.model.param_spec
        assert "intensity" not in cfg.model.frozen_values

    def test_target_rms_is_tunable(self):
        cfg = TrainConfig.model_validate(
            {
                "run": {"name": "t"},
                "data": {"manifest_path": "manifests/t.jsonl", "target_rms": 0.1},
            }
        )
        assert cfg.data.target_rms == 0.1
