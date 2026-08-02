"""Tests for the trajectory-regularisation terms in train.py.

The point of the acceleration penalty is that it separates *speed* from
*jitter*, which the first-difference penalty cannot. These tests pin that
distinction down, since the sweep it supports is read off those numbers.
"""

import torch

from samuel.encoder import SEANetEncoderConfig
from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES
from samuel.train import (
    _acceleration_loss,
    _reversal_rate,
    _smoothness_loss,
)

_PARAM = "tongueIndex"
_WEIGHTS = {_PARAM: 1.0}


def _module() -> PinkTromboneController:
    return PinkTromboneController(
        PinkTromboneControllerConfig(
            encoder=SEANetEncoderConfig(n_filters=8, dimension=32, n_residual_layers=1),
            samples_per_frame=2048,
            n_buckets=8,
        )
    )


def _params_from(traj: list[float], module: PinkTromboneController) -> torch.Tensor:
    """Embed a 1-D trajectory as the tongueIndex column of a params tensor."""
    lo, hi = module.config.param_spec[_PARAM][:2]
    out = torch.zeros(1, len(traj), N_PARAMS)
    # Values given in [0, 1] of the param's range; the losses renormalise back.
    out[0, :, PARAM_NAMES.index(_PARAM)] = lo + torch.tensor(traj) * (hi - lo)
    return out


class TestAccelerationPenalty:
    def test_steady_ramp_is_free_however_fast(self):
        """The whole reason for the second difference: speed is not jitter."""
        module = _module()
        slow = _params_from([0.0, 0.1, 0.2, 0.3, 0.4], module)
        fast = _params_from([0.0, 0.5, 1.0, 1.5, 2.0], module)
        for traj in (slow, fast):
            assert _acceleration_loss(traj, module, _WEIGHTS).item() < 1e-6
        # ...whereas the first-difference penalty charges 5x more for the same
        # gesture just because it happens sooner.
        assert (
            _smoothness_loss(fast, module, _WEIGHTS).item()
            > 4.9 * _smoothness_loss(slow, module, _WEIGHTS).item()
        )

    def test_zigzag_is_charged(self):
        module = _module()
        zigzag = _params_from([0.0, 0.1, 0.0, 0.1, 0.0], module)
        ramp = _params_from([0.0, 0.1, 0.2, 0.3, 0.4], module)
        assert _acceleration_loss(zigzag, module, _WEIGHTS).item() > 0.1
        # Same mean |delta| for both, so the smoothness penalty cannot tell
        # them apart at all -- the acceleration penalty is what distinguishes.
        s_zig = _smoothness_loss(zigzag, module, _WEIGHTS).item()
        s_ramp = _smoothness_loss(ramp, module, _WEIGHTS).item()
        assert abs(s_zig - s_ramp) < 1e-6

    def test_unweighted_params_contribute_nothing(self):
        module = _module()
        zigzag = _params_from([0.0, 0.1, 0.0, 0.1], module)
        assert _acceleration_loss(zigzag, module, {}).item() == 0.0

    def test_is_differentiable(self):
        module = _module()
        traj = _params_from([0.0, 0.1, 0.0, 0.1], module).requires_grad_(True)
        _acceleration_loss(traj, module, _WEIGHTS).backward()
        assert traj.grad.abs().sum() > 0


class TestReversalRate:
    def test_monotone_glide_scores_zero(self):
        module = _module()
        assert _reversal_rate(_params_from([0.0, 0.2, 0.5, 0.9], module), module) == 0.0

    def test_frame_by_frame_zigzag_scores_one(self):
        module = _module()
        traj = _params_from([0.0, 0.1, 0.0, 0.1, 0.0, 0.1], module)
        assert _reversal_rate(traj, module) == 1.0

    def test_a_rate_limited_ramp_still_shows_its_oscillation(self):
        """mean |delta| is blind to this; that is why the metric exists.

        Two trajectories with identical mean |delta| -- one a glide, one a
        zigzag of the same step size -- must be told apart.
        """
        module = _module()
        glide = _params_from([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], module)
        zigzag = _params_from([0.0, 0.1, 0.0, 0.1, 0.0, 0.1], module)
        assert _smoothness_loss(glide, module, _WEIGHTS).item() == (
            _smoothness_loss(zigzag, module, _WEIGHTS).item()
        )
        assert _reversal_rate(glide, module) < _reversal_rate(zigzag, module)

    def test_constant_trajectory_is_not_a_reversal(self):
        module = _module()
        assert _reversal_rate(_params_from([0.3] * 6, module), module) == 0.0
