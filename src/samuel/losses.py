"""Multi-scale log-magnitude STFT loss, ported from notebooks/tract_fit.ipynb."""

import math
from collections.abc import Sequence

import librosa
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from samuel.pink_trombone import SAMPLE_RATE


class MFCCLoss(nn.Module):
    """L1 distance between MFCCs, computed at hop ``samples_per_frame``.

    Mirrors the ``mfcc`` branch of ``LossKit`` in
    ``scripts/per_frame_fit_search.py`` when ``n_fft is None`` (defaults to
    ``samples_per_frame``, no overlap). Setting ``n_fft > samples_per_frame``
    introduces window overlap (e.g. 4× at ``n_fft = 2048`` with hop=512)
    so each frame's spectrum sees more temporal context. 80-bin mel
    filterbank, ``log1p`` of squared magnitudes, orthonormal DCT-II
    truncated to ``n_mfcc`` coefficients.
    """

    def __init__(
        self,
        samples_per_frame: int = 2048,
        n_mels: int = 80,
        n_mfcc: int = 20,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int | None = None,
    ):
        super().__init__()
        if n_fft is None:
            n_fft = samples_per_frame
        if n_fft < samples_per_frame:
            raise ValueError(
                f"n_fft ({n_fft}) must be >= samples_per_frame ({samples_per_frame})"
            )
        self.samples_per_frame = samples_per_frame
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

        window = torch.hann_window(n_fft)
        self.register_buffer("window", window, persistent=False)

        mel_fb = torch.from_numpy(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
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
        """``x [B, S]`` -> ``[B, n_mfcc, T]`` with ``T`` set by hop and n_fft."""
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
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


class MelSpecLoss(nn.Module):
    """L1 distance between log-mel spectrograms (no DCT vs. ``MFCCLoss``).

    Same windowing as ``MFCCLoss`` (``n_fft = hop = samples_per_frame``,
    no overlap, ``log1p`` of squared magnitudes) but compares the full mel
    energy spectrum instead of the DCT-truncated cepstrum. Cheaper than
    MFCC, retains spectral detail above the first 20 cepstral coefficients.
    """

    def __init__(
        self,
        samples_per_frame: int = 2048,
        n_mels: int = 80,
        sample_rate: int = SAMPLE_RATE,
    ):
        super().__init__()
        self.samples_per_frame = samples_per_frame
        self.n_mels = n_mels
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

    def features(self, x: Tensor) -> Tensor:
        """``x [B, S]`` (S a multiple of samples_per_frame) -> ``[B, n_mels, T]``."""
        spec = torch.stft(
            x,
            n_fft=self.samples_per_frame,
            hop_length=self.samples_per_frame,
            window=self.window,
            center=False,
            return_complex=True,
        ).abs()  # [B, n_freqs, T]
        mel = torch.einsum("mf,bft->bmt", self.mel_fb, spec.pow(2))
        return torch.log1p(mel)

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


class OnsetAlignLoss(nn.Module):
    """Encourage the synth's closure-release events to coincide with onsets
    in the target audio.

    Detects onsets in target via spectral flux on a mel spectrogram (no grad,
    target is fixed) at the same hop as the controller's frame rate. The loss
    asks: for each onset frame, is there a release somewhere in a small
    neighborhood? Spurious releases (release with no onset) are not penalized
    here.

    Important: this loss keys directly off the controller's
    ``constrictionDiameter`` parameter rather than the full diameter
    profile's min. Two reasons:

      1. ``min(diameter)`` blocks gradient to all tract positions that
         aren't currently the minimum, so when the constriction isn't
         already tighter than the tongue, the loss can't tell the
         controller to push ``cd`` down.

      2. The controller directly emits ``constrictionDiameter``; this is
         the lever the loss should push on. Keying off ``cd`` is a clean,
         monotonic signal with gradient everywhere.

    The synth's transient trigger still uses the full ``min(diameter)``
    (steep sigmoid), so transients fire only on true closures.

    Formally:
        env[t]   = max-normalized spectral flux at frame t (target)
        c[t]     = sigmoid(k * (threshold - cd[t-1])) (pred, "was closed before t?")
        c_d[t]   = max-pool of c over a ±radius window
        loss     = mean ReLU(env - c_d)

    Asymmetric: each strong target onset wants a "was closed before"
    nearby, but a closure at a non-onset frame is fine here (the
    reconstruction loss handles "don't close when you shouldn't").
    """

    def __init__(
        self,
        samples_per_frame: int = 512,
        n_fft: int = 2048,
        n_mels: int = 80,
        sample_rate: int = SAMPLE_RATE,
        align_radius: int = 2,
        closure_k: float = 4.0,
        closure_threshold: float = 0.3,
    ):
        super().__init__()
        self.samples_per_frame = samples_per_frame
        self.n_fft = n_fft
        self.align_radius = align_radius
        self.closure_k = closure_k
        self.closure_threshold = closure_threshold
        window = torch.hann_window(n_fft)
        self.register_buffer("window", window, persistent=False)
        mel_fb = torch.from_numpy(
            librosa.filters.mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=0.0,
                fmax=sample_rate / 2,
            )
        ).float()
        self.register_buffer("mel_fb", mel_fb, persistent=False)

    @torch.no_grad()
    def _onset_envelope(self, target: Tensor) -> Tensor:
        """Spectral-flux onset envelope at frame rate, normalized per clip."""
        spec = torch.stft(
            target,
            n_fft=self.n_fft,
            hop_length=self.samples_per_frame,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()  # [B, F, T+1]
        mel = torch.einsum("mf,bft->bmt", self.mel_fb, spec.pow(2))
        log_mel = torch.log1p(mel)
        # Spectral flux: positive change frame-to-frame, summed over mel bins.
        flux = F.relu(log_mel[:, :, 1:] - log_mel[:, :, :-1]).sum(dim=1)  # [B, T]
        # Normalize per clip so the loss scale doesn't depend on volume.
        flux = flux / (flux.amax(dim=-1, keepdim=True) + 1e-8)
        return flux

    def _soft_closure_prev(self, cd: Tensor) -> Tensor:
        """Soft closure indicator at the *previous* frame, gentler than the
        synth's transient trigger so gradient flows over the whole range of
        ``cd`` and not just where the constriction is already tighter than
        the tongue."""
        closure = torch.sigmoid(self.closure_k * (self.closure_threshold - cd))
        return F.pad(closure[:, :-1], (1, 0))

    def forward(self, cd: Tensor, target: Tensor) -> Tensor:
        """``cd`` is [B, T_ctrl] (constrictionDiameter); ``target`` is [B, S]."""
        c_prev = self._soft_closure_prev(cd)  # [B, T_ctrl]
        env = self._onset_envelope(target)  # [B, T_env]
        k = 2 * self.align_radius + 1
        c_d = F.max_pool1d(
            c_prev.unsqueeze(1),
            kernel_size=k,
            stride=1,
            padding=self.align_radius,
        ).squeeze(1)
        T = min(env.shape[-1], c_d.shape[-1])
        return F.relu(env[..., :T] - c_d[..., :T]).mean()


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
