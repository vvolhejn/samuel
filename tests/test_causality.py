"""Measured causality of the controller and of the synth's control-rate path.

Both are checked by gradient: an output is influenced by an input exactly where
the gradient is nonzero, which catches a leak wherever it hides -- padding,
resampling, an off-by-one -- in a way reading the convolutions cannot.
"""

import pytest
import torch

from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, _upsample_params

_T_CTRL = 12


def _module(lookahead: int = 0) -> PinkTromboneController:
    cfg = PinkTromboneControllerConfig(
        samples_per_frame=512, lookahead_frames=lookahead
    )
    return PinkTromboneController(cfg).eval()


def _last_influencing_sample(module: PinkTromboneController, t: int) -> int:
    """Index of the last input sample control frame ``t`` depends on."""
    spf = module.samples_per_frame
    wav = (torch.randn(1, 1, _T_CTRL * spf) * 0.1).requires_grad_(True)
    f0 = torch.full((1, _T_CTRL), 120.0)
    _, aux = module(wav, f0, return_aux=True)
    (grad,) = torch.autograd.grad(aux["logits"][0, t].sum(), wav)
    nonzero = (grad[0, 0].abs() > 0).nonzero().flatten()
    return int(nonzero[-1]) if len(nonzero) else -1


class TestControllerLookahead:
    @pytest.mark.parametrize("lookahead", [0, 1, 3])
    def test_sees_exactly_the_allotted_future(self, lookahead):
        module = _module(lookahead)
        spf = module.samples_per_frame
        # Frames near the end are clipped by the clip boundary, so check the
        # interior where the bound is the model's and not the input's.
        for t in (2, 5, 7):
            limit = (t + 1 + lookahead) * spf - 1
            assert _last_influencing_sample(module, t) == limit

    def test_default_is_zero_lookahead(self):
        assert PinkTromboneControllerConfig().lookahead_frames == 0

    def test_encoder_hop_matches_the_control_hop(self):
        """Otherwise forward() resamples the latents and leaks a few ms."""
        module = _module()
        assert module.encoder.hop_length == module.samples_per_frame


class TestParamUpsampling:
    def test_frame_starts_on_its_own_value(self):
        params = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        up = _upsample_params(params, 8)[0, :, 0]
        assert up[::8].tolist() == [0.0, 1.0, 2.0, 3.0]

    def test_ramps_toward_the_next_frame_only(self):
        params = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        up = _upsample_params(params, 4)[0, :, 0]
        # Frame 0 spans [0, 1), frame 1 spans [1, 2), ... in frame-index units.
        assert up[:4].tolist() == [0.0, 0.25, 0.5, 0.75]
        assert up[4:8].tolist() == [1.0, 1.25, 1.5, 1.75]

    def test_last_frame_holds(self):
        """No successor exists, so it must not reach for one."""
        params = torch.tensor([[[0.0], [1.0]]])
        assert _upsample_params(params, 4)[0, 4:, 0].tolist() == [1.0] * 4

    def test_mapping_is_independent_of_clip_length(self):
        """align_corners interpolation fails this; a streaming synth needs it."""
        short = torch.arange(4, dtype=torch.float32).view(1, 4, 1)
        long = torch.arange(40, dtype=torch.float32).view(1, 40, 1)
        assert torch.equal(
            _upsample_params(short, 16)[0, :48], _upsample_params(long, 16)[0, :48]
        )

    def test_a_sample_depends_on_two_control_frames(self):
        params = torch.zeros(1, 5, 1, requires_grad=True)
        up = _upsample_params(params, 8)
        # A sample inside frame 1 may touch frames 1 and 2, nothing later.
        (grad,) = torch.autograd.grad(up[0, 12, 0], params)
        assert (grad[0, :, 0].abs() > 0).nonzero().flatten().tolist() == [1, 2]


class TestTractIsFrameLocal:
    def test_ir_path_does_not_reach_into_the_next_frame(self):
        """Audio in frame t must not depend on tract params of frame t+1."""
        from samuel.pink_trombone import pink_trombone_ola

        spf, T = 512, 4
        params = torch.rand(1, T, N_PARAMS, requires_grad=True)
        with torch.no_grad():
            params[..., PARAM_NAMES.index("frequency")] = 150.0
        audio = pink_trombone_ola(
            params, seed=0, ir_length=64, control_rate=44100 / spf
        )
        # Sample at the very start of frame 1: the FIR is frame 1's, and the
        # glottis ramp at offset 0 sits exactly on frame 1's value.
        (grad,) = torch.autograd.grad(audio[0, spf], params)
        tract = [PARAM_NAMES.index(n) for n in ("tongueIndex", "tongueDiameter")]
        assert grad[0, 2:, tract].abs().sum() == 0
