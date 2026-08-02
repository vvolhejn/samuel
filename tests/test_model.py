"""Tests for PinkTromboneController."""

import math

import pytest
import torch

from samuel.encoder import SEANetEncoderConfig
from samuel.losses import MFCCLoss
from samuel.model import (
    _DEFAULT_FROZEN_VALUES,
    _DEFAULT_PARAM_SPEC,
    PinkTromboneController,
    PinkTromboneControllerConfig,
    slew_rate_limit,
)
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, pink_trombone_ola


def _small_config() -> PinkTromboneControllerConfig:
    return PinkTromboneControllerConfig(
        encoder=SEANetEncoderConfig(n_filters=8, dimension=32, n_residual_layers=1),
        samples_per_frame=2048,
        n_buckets=8,
    )


def _zero_f0(model: PinkTromboneController, B: int, T_ctrl: int) -> torch.Tensor:
    return torch.full((B, T_ctrl), 200.0)


class TestController:
    def test_forward_shape(self):
        cfg = _small_config()
        model = PinkTromboneController(cfg)
        S = cfg.samples_per_frame * 4
        wav = torch.zeros(2, 1, S)
        T_ctrl = math.ceil(S / model.samples_per_frame)
        f0 = _zero_f0(model, 2, T_ctrl)
        params = model(wav, f0)
        assert params.shape == (2, T_ctrl, N_PARAMS)

    def test_external_f0(self):
        cfg = _small_config()
        model = PinkTromboneController(cfg).eval()
        S = cfg.samples_per_frame * 4
        T_ctrl = S // cfg.samples_per_frame
        wav = torch.randn(2, 1, S)
        f0 = torch.linspace(80, 400, T_ctrl).expand(2, T_ctrl).contiguous()
        params = model(wav, f0)
        assert torch.allclose(params[..., PARAM_NAMES.index("frequency")], f0)

    def test_intensity_trainable_by_default(self):
        """Default config leaves the model its own level knob.

        Nothing corrects the synth output level downstream (the dataset is
        normalised to ``data.target_rms`` instead), so ``intensity`` has to be
        predicted.
        """
        cfg = _small_config()
        assert "intensity" in cfg.param_spec
        assert "intensity" not in cfg.frozen_values
        model = PinkTromboneController(cfg).eval()
        S = cfg.samples_per_frame * 4
        T_ctrl = S // cfg.samples_per_frame
        params = model(torch.randn(2, 1, S), _zero_f0(model, 2, T_ctrl))
        intensity = params[..., PARAM_NAMES.index("intensity")]
        assert intensity.min() >= 0.0 and intensity.max() <= 1.0

    def test_params_can_be_frozen(self):
        """frozen_values pins a param the head would otherwise predict."""
        cfg = _small_config()
        cfg.param_spec = {k: v for k, v in cfg.param_spec.items() if k != "intensity"}
        cfg.frozen_values = {**cfg.frozen_values, "intensity": 1.0}
        model = PinkTromboneController(cfg).eval()
        S = cfg.samples_per_frame * 4
        T_ctrl = S // cfg.samples_per_frame
        params = model(torch.randn(2, 1, S), _zero_f0(model, 2, T_ctrl))
        assert (params[..., PARAM_NAMES.index("intensity")] == 1.0).all()

    def test_eval_outputs_are_bucket_centers(self):
        cfg = _small_config()
        model = PinkTromboneController(cfg).eval()
        S = cfg.samples_per_frame * 4
        T_ctrl = S // cfg.samples_per_frame
        wav = torch.randn(1, 1, S)
        f0 = _zero_f0(model, 1, T_ctrl)
        params = model(wav, f0)
        for j, name in enumerate(model.trainable_names_):
            i = PARAM_NAMES.index(name)
            vals = params[..., i].flatten()
            centers = model.bucket_centers[j]
            min_diff = (vals.unsqueeze(-1) - centers).abs().min(-1).values
            assert min_diff.max().item() < 1e-5

    def test_grad_flows_through_synth(self):
        cfg = _small_config()
        model = PinkTromboneController(cfg)
        with torch.no_grad():
            model.head.weight.normal_(std=0.01)
        loss_fn = MFCCLoss(samples_per_frame=cfg.samples_per_frame)
        S = cfg.samples_per_frame * 4
        T_ctrl = S // cfg.samples_per_frame
        wav = torch.randn(1, 1, S) * 0.1
        f0 = _zero_f0(model, 1, T_ctrl)
        params = model(wav, f0, tau=2.0)
        pred = pink_trombone_ola(
            params, ir_length=64, control_rate=cfg.frame_rate, seed=0
        )
        S_out = pred.shape[-1]
        target = torch.zeros_like(pred)
        loss = loss_fn(pred, target[..., :S_out])
        loss.backward()

        assert model.head.weight.grad is not None
        assert model.head.weight.grad.abs().sum().item() > 0

        grads = [
            p.grad
            for p in model.encoder.parameters()
            if p.requires_grad and p.grad is not None
        ]
        assert grads, "no encoder grads populated"
        any_nonzero = any(g.abs().sum().item() > 0 for g in grads)
        assert any_nonzero, "all encoder grads are zero"

    def test_config_coverage_validation(self):
        """Missing a Pink Trombone param in spec, frozen, and external f0 should error."""
        bad_frozen = dict(_DEFAULT_FROZEN_VALUES)
        bad_frozen.pop("vibratoWobble")
        cfg = PinkTromboneControllerConfig(
            encoder=SEANetEncoderConfig(n_filters=8, dimension=16),
            frozen_values=bad_frozen,
        )
        with pytest.raises(ValueError, match="covered by neither"):
            PinkTromboneController(cfg)

    def test_frequency_in_spec_rejected(self):
        bad_spec = dict(_DEFAULT_PARAM_SPEC)
        bad_spec["frequency"] = (80.0, 400.0, 200.0)
        cfg = PinkTromboneControllerConfig(
            encoder=SEANetEncoderConfig(n_filters=8, dimension=16),
            param_spec=bad_spec,
        )
        with pytest.raises(ValueError, match="frequency.*externally"):
            PinkTromboneController(cfg)


class TestSlewRateLimit:
    @pytest.mark.parametrize("mode", ["clamp", "tanh"])
    def test_bound_is_respected(self, mode):
        torch.manual_seed(0)
        values = torch.randn(3, 64, 2) * 10.0
        max_delta = torch.tensor([0.1, 1.0])
        out = slew_rate_limit(values, max_delta, mode)
        assert out.shape == values.shape
        assert torch.equal(out[:, 0], values[:, 0])
        diffs = (out[:, 1:] - out[:, :-1]).abs()
        assert (diffs <= max_delta + 1e-5).all()

    def test_slow_trajectory_passes_through(self):
        """A trajectory already inside the limit must be left untouched."""
        t = torch.linspace(0, 1, 32).view(1, 32, 1)
        max_delta = torch.tensor([1.0])
        assert torch.allclose(slew_rate_limit(t, max_delta, "clamp"), t)

    def test_clamp_tracks_a_step_as_a_ramp(self):
        step = torch.cat([torch.zeros(1, 4, 1), torch.ones(1, 4, 1)], dim=1)
        out = slew_rate_limit(step, torch.tensor([0.25]), "clamp")
        expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0])
        assert torch.allclose(out.flatten(), expected)

    def test_gradient_at_the_knee_differs_by_mode(self):
        """tanh's reason to exist: a soft knee instead of a hard corner.

        A frame asking for 1.5x the limit gets exactly zero gradient under
        clamp, but still a usable one under tanh. Far past the limit tanh
        saturates numerically too, so this only helps near the boundary.
        """
        d = torch.tensor([0.1])

        def grad_on_second_frame(mode: str) -> float:
            values = torch.tensor([[[0.0], [0.15]]], requires_grad=True)
            slew_rate_limit(values, d, mode).sum().backward()
            return values.grad[0, 1, 0].item()

        assert grad_on_second_frame("clamp") == 0.0
        assert grad_on_second_frame("tanh") > 0.1

    def test_clamp_passes_gradient_back_through_saturated_frames(self):
        """A saturated frame is a dead end for its own logits only.

        d(out_t)/d(in_{t-1}) is 1 across a clipped link, so the loss still
        reaches earlier frames at full strength — the model learns to start
        moving sooner rather than losing the signal.
        """
        values = torch.tensor([[[0.0], [9.0], [9.0]]], requires_grad=True)
        slew_rate_limit(values, torch.tensor([0.1]), "clamp")[:, -1].sum().backward()
        assert values.grad.flatten().tolist() == [1.0, 0.0, 0.0]

    def test_model_output_is_rate_limited(self):
        cfg = _small_config()
        cfg.rate_limits = {"tongueIndex": 0.5}
        cfg.rate_limit_scale = 2.0  # effective limit 1.0 range/s
        model = PinkTromboneController(cfg).eval()
        S = cfg.samples_per_frame * 16
        T_ctrl = S // cfg.samples_per_frame
        wav = torch.randn(2, 1, S)
        params = model(wav, _zero_f0(model, 2, T_ctrl))

        lo, hi, _ = cfg.param_spec["tongueIndex"]
        per_frame = 1.0 * (hi - lo) / cfg.frame_rate
        traj = params[..., PARAM_NAMES.index("tongueIndex")]
        assert ((traj[:, 1:] - traj[:, :-1]).abs() <= per_frame + 1e-4).all()
        # Unlisted params stay free.
        other = params[..., PARAM_NAMES.index("constrictionIndex")]
        assert (other[:, 1:] - other[:, :-1]).abs().max() > per_frame

    def test_rate_limit_on_non_trainable_param_rejected(self):
        cfg = _small_config()
        cfg.rate_limits = {"vibratoGain": 1.0}
        with pytest.raises(ValueError, match="rate_limits names non-trainable"):
            PinkTromboneController(cfg)

    def test_null_scale_disables_the_limiter(self):
        """Limits can stay in the YAML; the scalar is the on/off switch."""
        cfg = _small_config()
        cfg.rate_limits = {"tongueIndex": 0.001}
        model = PinkTromboneController(cfg).eval()  # rate_limit_scale is None
        assert model.rate_limited_names_ == []
        S = cfg.samples_per_frame * 16
        wav = torch.randn(2, 1, S)
        traj = model(wav, _zero_f0(model, 2, S // cfg.samples_per_frame))[
            ..., PARAM_NAMES.index("tongueIndex")
        ]
        assert (traj[:, 1:] - traj[:, :-1]).abs().max() > 0.0
