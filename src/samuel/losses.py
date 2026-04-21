"""Multi-scale log-magnitude STFT loss, ported from notebooks/tract_fit.ipynb."""

from collections.abc import Sequence

import torch
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
