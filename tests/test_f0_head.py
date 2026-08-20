"""Causal pitch head: bucketing, what it feeds the synth, and its supervision."""

import math

import pytest
import torch

from samuel.model import (
    A4_HZ,
    F0HeadConfig,
    PinkTromboneController,
    PinkTromboneControllerConfig,
)
from samuel.pink_trombone import PARAM_NAMES
from samuel.train import _f0_losses

_FREQ = PARAM_NAMES.index("frequency")


def _module(enabled: bool = True, **kw) -> PinkTromboneController:
    cfg = PinkTromboneControllerConfig(
        samples_per_frame=512, f0=F0HeadConfig(enabled=enabled, **kw)
    )
    return PinkTromboneController(cfg)


class TestDisabledByDefault:
    def test_default_is_off(self):
        assert PinkTromboneControllerConfig().f0.enabled is False

    def test_f0_is_still_required_when_off(self):
        module = _module(enabled=False).eval()
        with pytest.raises(ValueError, match="f0 is required"):
            module(torch.randn(1, 1, 512 * 4), None)

    def test_no_pitch_weights_when_off(self):
        assert not [k for k in _module(enabled=False).state_dict() if "f0" in k]


class TestBucketing:
    def test_centers_are_equal_tempered_from_a440(self):
        module = _module()
        centers = module.f0_log_centers.exp()
        # Every center is an exact number of grid steps away from A4.
        steps = 12 * module.config.f0.buckets_per_semitone * (centers / A4_HZ).log2()
        assert torch.allclose(steps, steps.round(), atol=1e-4)

    def test_a440_is_a_center(self):
        centers = _module().f0_log_centers.exp()
        assert torch.isclose(centers, torch.tensor(A4_HZ), atol=1e-3).any()

    def test_default_grid_is_quarter_tones(self):
        cfg = F0HeadConfig()
        assert cfg.buckets_per_semitone == 2
        assert cfg.cents_per_bucket == 50.0

    def test_named_semitones_land_on_centers(self):
        """Quarter-tone grid => every other center is a semitone of A440."""
        centers = _module().f0_log_centers.exp()
        for hz in (73.416, 110.0, 220.0, 493.883):  # D2, A2, A3, B4
            assert torch.isclose(centers, torch.tensor(hz), rtol=1e-4).any()

    def test_grid_is_snapped_outward_to_cover_the_range(self):
        """Otherwise a label at the edge sits past the end bucket."""
        cfg = F0HeadConfig(enabled=True, fmin=70.0, fmax=500.0)
        centers = cfg.log_centers().exp()
        assert centers[0].item() <= 70.0
        assert centers[-1].item() >= 500.0

    def test_finer_grid_gives_more_buckets(self):
        coarse = F0HeadConfig(buckets_per_semitone=1)
        fine = F0HeadConfig(buckets_per_semitone=4)
        assert fine.cents_per_bucket == 25.0
        assert fine.n_buckets == pytest.approx(4 * coarse.n_buckets, abs=6)

    def test_target_is_the_nearest_center_in_cents(self):
        module = _module()
        centers = module.f0_log_centers.exp()
        # Exactly on a center, and a few cents either side of it.
        probe = torch.tensor([[centers[9], centers[9] * 1.005, centers[9] * 0.995]])
        assert module.f0_bucket_targets(probe)[0].tolist() == [9, 9, 9]

    def test_labels_outside_the_range_clamp_to_the_end_buckets(self):
        module = _module()
        n = module.config.f0.n_buckets
        probe = torch.tensor([[10.0, 5000.0]])
        assert module.f0_bucket_targets(probe)[0].tolist() == [0, n - 1]

    def test_a_range_too_narrow_to_bucket_is_rejected(self):
        with pytest.raises(ValueError, match="fewer than two buckets"):
            F0HeadConfig(fmin=440.0, fmax=440.0).log_centers()


class TestPredictedPitchReachesTheSynth:
    def test_frequency_column_comes_from_the_head(self):
        module = _module().eval()
        out, aux = module(torch.randn(1, 1, 512 * 4) * 0.1, None, return_aux=True)
        assert torch.allclose(out[..., _FREQ], aux["f0_hz"])

    def test_prediction_stays_inside_the_bucket_range(self):
        module = _module(fmin=70.0, fmax=500.0).eval()
        out, _ = module(torch.randn(2, 1, 512 * 8), None, return_aux=True)
        assert (out[..., _FREQ] >= 70.0 - 1e-3).all()
        assert (out[..., _FREQ] <= 500.0 + 1e-3).all()

    def test_argmax_at_eval_lands_exactly_on_a_center(self):
        module = _module().eval()
        _, aux = module(torch.randn(1, 1, 512 * 4), None, return_aux=True)
        centers = module.f0_log_centers.exp()
        nearest = (aux["f0_hz"].unsqueeze(-1) - centers).abs().min(dim=-1).values
        assert nearest.max().item() < 1e-3

    def test_soft_mix_is_geometric_not_arithmetic(self):
        """A linear mix of log-spaced centers would sit above the perceptual mid."""
        module = _module(buckets_per_semitone=1, fmin=110.0, fmax=116.6).train()
        with torch.no_grad():  # force a dead-even mixture
            module.f0_head.weight.zero_()
            module.f0_head.bias.zero_()
        _, aux = module(torch.zeros(1, 1, 512 * 2), None, tau=1e-6, return_aux=True)
        # Gumbel at tau -> 0 is a hard sample, so it must land on a center,
        # never on the arithmetic mean between two of them.
        centers = module.f0_log_centers.exp()
        for hz in aux["f0_hz"].flatten().tolist():
            assert (centers - hz).abs().min().item() < 1e-2

    def test_pitch_is_differentiable_through_to_the_synth_input(self):
        module = _module().train()
        out, _ = module(
            torch.randn(1, 1, 512 * 4) * 0.1, None, tau=2.0, return_aux=True
        )
        out[..., _FREQ].sum().backward()
        assert module.f0_head.weight.grad.abs().sum() > 0


class TestSupervision:
    def _aux(self, module, wav):
        return module(wav, None, tau=2.0, return_aux=True)[1]

    def test_f0_term_ignores_unvoiced_frames(self):
        """The pyin label there is fill_unvoiced's interpolation, not a reading."""
        module = _module().train()
        aux = self._aux(module, torch.randn(1, 1, 512 * 4) * 0.1)
        label = torch.tensor([[120.0, 120.0, 300.0, 300.0]])
        voiced = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        first_two, _, _ = _f0_losses(aux, module, label, voiced)
        # Changing only the unvoiced labels must not move the loss.
        moved = label.clone()
        moved[0, 2:] = 90.0
        again, _, _ = _f0_losses(aux, module, moved, voiced)
        assert torch.allclose(first_two, again)

    def test_all_unvoiced_batch_does_not_divide_by_zero(self):
        module = _module().train()
        aux = self._aux(module, torch.randn(1, 1, 512 * 4) * 0.1)
        f0_loss, voiced_loss, readouts = _f0_losses(
            aux, module, torch.full((1, 4), 120.0), torch.zeros(1, 4)
        )
        assert f0_loss.item() == 0.0
        assert math.isfinite(voiced_loss.item())
        assert math.isnan(readouts["f0_mae_cents"])

    def test_voiced_term_is_bce_over_every_frame(self):
        module = _module().train()
        with torch.no_grad():
            module.voiced_head.weight.zero_()
            module.voiced_head.bias.zero_()
        aux = self._aux(module, torch.randn(1, 1, 512 * 4) * 0.1)
        _, voiced_loss, readouts = _f0_losses(
            aux, module, torch.full((1, 4), 120.0), torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        )
        # Logit 0 everywhere -> p=0.5 -> BCE = log 2 on every frame.
        assert voiced_loss.item() == pytest.approx(math.log(2), abs=1e-5)
        assert readouts["voiced_frac"] == 0.5

    def test_both_terms_train_their_heads(self):
        module = _module().train()
        aux = self._aux(module, torch.randn(1, 1, 512 * 4) * 0.1)
        f0_loss, voiced_loss, _ = _f0_losses(
            aux, module, torch.full((1, 4), 120.0), torch.ones(1, 4)
        )
        (f0_loss + voiced_loss).backward()
        assert module.f0_head.weight.grad.abs().sum() > 0
        assert module.voiced_head.weight.grad.abs().sum() > 0


class TestPitchHeadIsCausal:
    def test_no_dependence_on_future_samples(self):
        module = _module().eval()
        spf, T = 512, 10
        wav = (torch.randn(1, 1, T * spf) * 0.1).requires_grad_(True)
        _, aux = module(wav, None, return_aux=True)
        for t in (2, 5):
            (grad,) = torch.autograd.grad(
                aux["f0_logits"][0, t].sum() + aux["voiced_logits"][0, t],
                wav,
                retain_graph=True,
            )
            last = int((grad[0, 0].abs() > 0).nonzero().flatten()[-1])
            assert last == (t + 1) * spf - 1
