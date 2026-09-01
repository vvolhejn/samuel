"""Tests for the trajectory-regularisation terms in train.py.

The point of the acceleration penalty is that it separates *speed* from
*jitter*, which the first-difference penalty cannot. These tests pin that
distinction down, since the sweep it supports is read off those numbers.
"""

import torch

from samuel.config import DataConfig, LossConfig, RunConfig, TrainConfig
from samuel.encoder import SEANetEncoderConfig
from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES
from samuel.train import (
    _acceleration_loss,
    _rest_pose_distance,
    _rest_pose_loss,
    _reversal_rate,
    _silent_frame_rest_metrics,
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


class TestRestPosePrior:
    def _targets(self, module: PinkTromboneController, at: float) -> dict[str, float]:
        """``{_PARAM: raw value}`` for a target given in [0, 1] of the range."""
        lo, hi = module.config.param_spec[_PARAM][:2]
        return {_PARAM: lo + at * (hi - lo)}

    def test_zero_at_the_target(self):
        module = _module()
        traj = _params_from([0.25] * 4, module)
        assert (
            _rest_pose_loss(traj, module, self._targets(module, 0.25), {}).item() < 1e-6
        )

    def test_measures_normalised_distance(self):
        module = _module()
        traj = _params_from([0.2, 0.4], module)  # mean distance 0.2 from 0.5
        loss = _rest_pose_loss(traj, module, self._targets(module, 0.5), {})
        assert abs(loss.item() - 0.2) < 1e-5

    def test_constant_force_regardless_of_distance(self):
        """L1, not L2: that is what makes it a fixed offset to the recon gradient."""
        module = _module()
        targets = self._targets(module, 1.0)
        grads = []
        for at in (0.2, 0.8):
            traj = _params_from([at] * 4, module).requires_grad_(True)
            _rest_pose_loss(traj, module, targets, {}).backward()
            grads.append(traj.grad.abs().sum().item())
        assert abs(grads[0] - grads[1]) < 1e-6

    def test_weights_scale_the_pull(self):
        module = _module()
        targets = self._targets(module, 1.0)
        traj = _params_from([0.2] * 4, module)
        base = _rest_pose_loss(traj, module, targets, {}).item()
        assert (
            abs(
                _rest_pose_loss(traj, module, targets, {_PARAM: 0.5}).item()
                - 0.5 * base
            )
            < 1e-6
        )

    def test_unlisted_params_contribute_nothing(self):
        module = _module()
        traj = _params_from([0.2] * 4, module)
        assert _rest_pose_loss(traj, module, {}, {}).item() == 0.0
        dist = _rest_pose_distance(traj, module, self._targets(module, 1.0))
        for name, v in zip(module.trainable_names_, dist.tolist()):
            if name != _PARAM:
                assert v == 0.0

    def test_is_differentiable_toward_the_target(self):
        module = _module()
        traj = _params_from([0.2] * 4, module).requires_grad_(True)
        _rest_pose_loss(traj, module, self._targets(module, 1.0), {}).backward()
        # Below the target, so increasing the param must decrease the loss.
        assert traj.grad[0, :, PARAM_NAMES.index(_PARAM)].max() < 0


class TestSilentFrameRestMetrics:
    """The eval-side readout: rest distance on the frames the recon loss ignores."""

    def _cfg(self, targets: dict[str, float]) -> TrainConfig:
        return TrainConfig(
            run=RunConfig(name="test"),
            data=DataConfig(manifest_path="manifests/unused.jsonl", target_rms=0.05),
            loss=LossConfig(rest=0.1, rest_targets=targets),
        )

    def test_splits_silent_from_loud_frames(self):
        module = _module()
        spf = module.samples_per_frame
        lo, hi = module.config.param_spec[_PARAM][:2]
        # Frame 0 loud, frame 1 silent; the param sits at the target only
        # during the silent frame, so the silent-frame distance must be 0
        # while the all-frame distance is not.
        params = _params_from([0.0, 1.0], module)
        target = torch.cat([torch.full((1, spf), 0.05), torch.zeros(1, spf)], dim=1)
        cfg = self._cfg({_PARAM: hi})
        out = _silent_frame_rest_metrics(params, module, target, cfg)
        assert out["eval/silent_frac"] == 0.5
        assert out["eval/rest_loss_silent"] == 0.0
        assert out[f"eval/rest_dist_silent/{_PARAM}"] == 0.0
        assert _rest_pose_distance(params, module, {_PARAM: hi}).sum().item() > 0.4
        assert lo < hi  # sanity: range orientation assumed above

    def test_reports_only_the_fraction_when_nothing_is_silent(self):
        module = _module()
        params = _params_from([0.0, 1.0], module)
        target = torch.full((1, 2 * module.samples_per_frame), 0.05)
        out = _silent_frame_rest_metrics(
            params, module, target, self._cfg({_PARAM: 1.0})
        )
        assert out == {"eval/silent_frac": 0.0}


class TestReversalRate:
    def test_monotone_glide_scores_zero(self):
        module = _module()
        assert _reversal_rate(_params_from([0.0, 0.2, 0.5, 0.9], module), module) == 0.0

    def test_frame_by_frame_zigzag_scores_one(self):
        module = _module()
        traj = _params_from([0.0, 0.1, 0.0, 0.1, 0.0, 0.1], module)
        assert _reversal_rate(traj, module) == 1.0

    def test_glide_and_zigzag_are_told_apart(self):
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
