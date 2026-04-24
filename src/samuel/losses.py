"""Multi-scale log-magnitude STFT loss, ported from notebooks/tract_fit.ipynb."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiScaleLogMagSTFTLoss(nn.Module):
    """L1 distance between log-magnitude STFTs at several window sizes.

    Matches the loss in ``notebooks/tract_fit.ipynb``:

        L = mean_n | log1p(|STFT_n(pred)|) - log1p(|STFT_n(target)|) |

    averaged over the set of ``n_ffts``. Phase-invariant, so pred and target
    need not be waveform-aligned.
    """

    def __init__(
        self,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        hop_div: int = 4,
    ):
        super().__init__()
        self.n_ffts = tuple(n_ffts)
        self.hop_div = hop_div
        for n in self.n_ffts:
            self.register_buffer(f"window_{n}", torch.hann_window(n), persistent=False)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        assert pred.shape == target.shape, (pred.shape, target.shape)
        loss = pred.new_zeros(())
        for n_fft in self.n_ffts:
            window: Tensor = getattr(self, f"window_{n_fft}")
            hop = n_fft // self.hop_div
            s_p = torch.stft(
                pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
            ).abs()
            s_t = torch.stft(
                target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
            ).abs()
            loss = loss + (torch.log1p(s_p) - torch.log1p(s_t)).abs().mean()
        return loss / len(self.n_ffts)


class PitchCentsLoss(nn.Module):
    """Masked L1 pitch error in cents.

    The prediction is the model's commanded ``frequency`` parameter (in Hz)
    per control frame, no pitch estimation needed on the synth output.
    Target is pyin-extracted f0 at the same rate, with a voiced mask
    indicating which frames to include.

    When no voiced frames are present in the batch, returns zero.
    """

    def forward(
        self,
        pred_hz: Tensor,  # [B, T]
        target_hz: Tensor,  # [B, T]
        voiced_mask: Tensor,  # [B, T] bool
    ) -> Tensor:
        pred = pred_hz.clamp(min=1.0)
        target = target_hz.clamp(min=1.0)
        cents = 1200.0 * torch.log2(pred / target)
        mask = (voiced_mask & (target_hz > 0)).to(cents.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (cents.abs() * mask).sum() / denom


class LoudnessEnvelopeLoss(nn.Module):
    """L1 between log-RMS envelopes of pred and target.

    Forces the model to match per-clip energy over time, which is a strong
    per-clip signal (speech has pauses; silence has zero energy).
    """

    def __init__(self, win_size: int = 2048, hop: int = 512):
        super().__init__()
        self.win_size = win_size
        self.hop = hop

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # [B, S] -> [B, 1, S] for avg_pool
        p = pred.unsqueeze(1)
        t = target.unsqueeze(1)
        p_rms = F.avg_pool1d(p.pow(2), self.win_size, self.hop, padding=0).sqrt()
        t_rms = F.avg_pool1d(t.pow(2), self.win_size, self.hop, padding=0).sqrt()
        # log scale so scale-invariant, clip to avoid log(0)
        eps = 1e-5
        return (torch.log(p_rms + eps) - torch.log(t_rms + eps)).abs().mean()


class MultiScaleLogMagSTFTLossWithEnvelope(nn.Module):
    """Multi-scale STFT loss + loudness-envelope loss."""

    def __init__(
        self,
        n_ffts: Sequence[int] = (512, 1024, 2048),
        hop_div: int = 4,
        envelope_weight: float = 0.1,
    ):
        super().__init__()
        self.stft = MultiScaleLogMagSTFTLoss(n_ffts=n_ffts, hop_div=hop_div)
        self.envelope = LoudnessEnvelopeLoss()
        self.envelope_weight = envelope_weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.stft(pred, target) + self.envelope_weight * self.envelope(
            pred, target
        )
