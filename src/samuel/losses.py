"""Multi-scale log-magnitude STFT loss, ported from notebooks/tract_fit.ipynb."""

import math
from collections.abc import Sequence

import librosa
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from samuel.pink_trombone import SAMPLE_RATE


class MFCCLoss(nn.Module):
    """L1 distance between MFCCs, computed once per ``samples_per_frame`` frame.

    Mirrors the ``mfcc`` branch of ``LossKit`` in
    ``scripts/per_frame_fit_search.py``: ``n_fft = hop = samples_per_frame``,
    no overlap, single STFT slice per frame; an 80-bin mel filterbank with
    ``log1p`` of squared magnitudes; orthonormal DCT-II truncated to
    ``n_mfcc`` coefficients.
    """

    def __init__(
        self,
        samples_per_frame: int = 2048,
        n_mels: int = 80,
        n_mfcc: int = 20,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.samples_per_frame = samples_per_frame
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

        window = torch.hann_window(samples_per_frame)
        self.register_buffer("window", window, persistent=False)

        mel_fb = torch.from_numpy(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=samples_per_frame,
                n_mels=n_mels,
                fmin=0.0,
                fmax=sample_rate / 2,
            )
        ).float()  # [n_mels, n_freqs]
        self.register_buffer("mel_fb", mel_fb, persistent=False)

        k = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(n_mels, dtype=torch.float32)
        dct = torch.cos(math.pi * k * (n + 0.5) / n_mels) * math.sqrt(2.0 / n_mels)
        dct[0] *= 1.0 / math.sqrt(2.0)
        self.register_buffer("dct", dct, persistent=False)  # [n_mfcc, n_mels]

    def features(self, x: Tensor) -> Tensor:
        """``x [B, S]`` (S a multiple of samples_per_frame) -> ``[B, n_mfcc, T]``."""
        spec = torch.stft(
            x,
            n_fft=self.samples_per_frame,
            hop_length=self.samples_per_frame,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()  # [B, n_freqs, T]
        mel = torch.einsum("mf,bft->bmt", self.mel_fb, spec.pow(2))
        log_mel = torch.log1p(mel)
        return torch.einsum("km,bmt->bkt", self.dct, log_mel)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        assert pred.shape == target.shape, (pred.shape, target.shape)
        return (self.features(pred) - self.features(target)).abs().mean()


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
