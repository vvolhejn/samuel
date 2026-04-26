"""Tests for PinkTromboneController."""

import math

import pytest
import torch

from samuel.encoder import SEANetEncoderConfig
from samuel.losses import MultiScaleLogMagSTFTLoss
from samuel.model import (
    _DEFAULT_FROZEN_VALUES,
    _DEFAULT_PARAM_SPEC,
    PinkTromboneController,
    PinkTromboneControllerConfig,
)
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola


def _small_config() -> PinkTromboneControllerConfig:
    return PinkTromboneControllerConfig(
        encoder=SEANetEncoderConfig(n_filters=8, dimension=32, n_residual_layers=1),
        frame_rate=12.5,
    )


class TestController:
    def test_forward_shape(self):
        cfg = _small_config()
        model = PinkTromboneController(cfg)
        S = SAMPLE_RATE  # 1 second
        wav = torch.zeros(2, 1, S)
        params = model(wav)
        T_ctrl = math.ceil(S / model.samples_per_frame)
        assert params.shape == (2, T_ctrl, N_PARAMS)

    def test_initial_output_matches_init_table(self):
        """Zero-weight head + logit(init_norm) bias -> params equal the configured init."""
        cfg = _small_config()
        model = PinkTromboneController(cfg)
        with torch.no_grad():
            model.head.weight.zero_()
        S = model.samples_per_frame * 4
        wav = torch.randn(1, 1, S)  # encoder output does not matter, weight=0
        params = model(wav)[0, 0]  # [N_PARAMS]
        for name, (lo, hi, init) in _DEFAULT_PARAM_SPEC.items():
            i = PARAM_NAMES.index(name)
            assert params[i].item() == pytest.approx(init, abs=1e-3)
        for name, val in _DEFAULT_FROZEN_VALUES.items():
            i = PARAM_NAMES.index(name)
            assert params[i].item() == pytest.approx(val, abs=1e-6)

    def test_grad_flows_through_synth(self):
        """Gradients reach encoder weights through the Pink Trombone synth.

        The head is zero-initialized so day-0 training has zero encoder grads;
        we perturb the head weights first to verify the end-to-end wiring.
        """
        cfg = _small_config()
        model = PinkTromboneController(cfg)
        with torch.no_grad():
            model.head.weight.normal_(std=0.01)
        loss_fn = MultiScaleLogMagSTFTLoss(n_ffts=(256,), hop_div=4)
        S = model.samples_per_frame * 4
        wav = torch.randn(1, 1, S) * 0.1
        params = model(wav)
        pred = pink_trombone_ola(
            params, ir_length=64, control_rate=cfg.frame_rate, seed=0
        )
        target = torch.zeros_like(pred)
        loss = loss_fn(pred, target)
        loss.backward()

        assert model.head.bias.grad is not None
        assert model.head.bias.grad.abs().sum().item() > 0

        grads = [
            p.grad
            for p in model.encoder.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads, "no encoder grads populated"
        any_nonzero = any(g.abs().sum().item() > 0 for g in grads)
        assert any_nonzero, "all encoder grads are zero"

    def test_config_coverage_validation(self):
        """Missing a Pink Trombone param in both spec and frozen should error."""
        bad_frozen = dict(_DEFAULT_FROZEN_VALUES)
        bad_frozen.pop("vibratoWobble")
        cfg = PinkTromboneControllerConfig(
            encoder=SEANetEncoderConfig(n_filters=8, dimension=16),
            frozen_values=bad_frozen,
        )
        with pytest.raises(ValueError, match="covered by neither"):
            PinkTromboneController(cfg)
