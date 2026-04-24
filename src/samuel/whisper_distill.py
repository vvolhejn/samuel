"""Whisper encoder wrapper for distillation + perceptual loss.

Two losses ride on the frozen ``whisper-large-v3`` encoder:

- **Latent distillation.** Project our SEANet latent ``z`` into Whisper's
  residual-stream dim (1280) and L1-match it to Whisper's final encoder
  output for the same audio. Gives the encoder a direct supervision signal
  that is independent of the differentiable synth.

- **Perceptual loss.** Run Whisper on the reconstructed waveform and the
  target waveform, then L1-match activations at a subset of layers. Lets
  the synth gradients be shaped by a feature space that cares about speech
  content (phonemes, pitch, voicing) rather than raw spectrogram bins.

Both share one forward through Whisper per batch for the target (no grad)
and one for the prediction (grad). The encoder is frozen and lives in the
same module so ``.to(device)`` moves everything together.
"""

from __future__ import annotations

from collections.abc import Sequence

import julius
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import WhisperFeatureExtractor, WhisperModel

WHISPER_SR = 16_000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
MAX_SOURCE_POSITIONS = 1500  # whisper encoder's positional embedding length


class WhisperDistiller(nn.Module):
    """Frozen Whisper encoder exposing per-layer hidden states for audio at 16 kHz.

    The encoder is sliced at its positional embedding so we can feed it
    arbitrary-length mel spectrograms (up to 30 s). This avoids padding
    every 4 s clip to 30 s, which is a ~15x compute saving.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        perceptual_layers: Sequence[int] | None = None,
        source_sample_rate: int = 44_100,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        model = WhisperModel.from_pretrained(model_name, dtype=dtype)
        self.encoder = model.encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.dtype = dtype

        fe = WhisperFeatureExtractor.from_pretrained(model_name)
        mel_filters = torch.from_numpy(np.asarray(fe.mel_filters, dtype=np.float32))
        # transformers stores filters as [n_freq, n_mels]; we want [n_mels, n_freq]
        if mel_filters.shape[0] != N_MELS:
            mel_filters = mel_filters.T.contiguous()
        assert mel_filters.shape == (N_MELS, N_FFT // 2 + 1), mel_filters.shape
        self.register_buffer("mel_filters", mel_filters, persistent=False)
        self.register_buffer(
            "window", torch.hann_window(N_FFT, dtype=torch.float32), persistent=False
        )

        self.d_model = int(model.config.d_model)
        self.num_layers = int(model.config.encoder_layers)
        if perceptual_layers is None:
            # 4 evenly-spaced layers spanning low → high-level features.
            perceptual_layers = [7, 15, 23, 31]
        self.perceptual_layers = tuple(sorted(set(perceptual_layers)))
        assert all(0 <= i < self.num_layers for i in self.perceptual_layers)

        self.source_sample_rate = source_sample_rate

    def train(self, mode: bool = True) -> "WhisperDistiller":  # type: ignore[override]
        super().train(mode)
        self.encoder.eval()  # always eval; we never train whisper
        return self

    # --- audio → log-mel ----------------------------------------------------

    def _resample_to_16k(self, audio: Tensor) -> Tensor:
        if self.source_sample_rate == WHISPER_SR:
            return audio
        return julius.resample_frac(audio, self.source_sample_rate, WHISPER_SR)

    def log_mel(self, audio_16k: Tensor) -> Tensor:
        """Whisper-style log-mel spectrogram. audio_16k [B, S] -> [B, n_mels, T]."""
        window: Tensor = self.window  # type: ignore[assignment]
        filters: Tensor = self.mel_filters  # type: ignore[assignment]
        stft = torch.stft(
            audio_16k,
            N_FFT,
            HOP_LENGTH,
            window=window,
            return_complex=True,
        )
        # drop the last frame to match openai/whisper's log_mel_spectrogram.
        magnitudes = stft[..., :-1].abs().pow(2)
        mel_spec = filters @ magnitudes  # [B, n_mels, T]
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        # per-clip dynamic range clip: max − 8.0
        log_spec_max = log_spec.amax(dim=(-2, -1), keepdim=True)
        log_spec = torch.maximum(log_spec, log_spec_max - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    # --- whisper encoder forward (variable length) --------------------------

    def _encode_mel(self, mel: Tensor) -> list[Tensor]:  # noqa: D401
        mel = mel.to(self.dtype)
        """Run whisper encoder on [B, n_mels, T_mel]; returns hidden states per layer.

        Output list has ``num_layers + 1`` tensors. Index 0 = post-conv
        embedding + positional, indices 1..num_layers = after each encoder
        layer. The final layer output passes through the encoder's final
        LayerNorm (matching the pretrained head's input).
        """
        enc = self.encoder
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x))  # [B, d, T]
        x = x.permute(0, 2, 1).contiguous()  # [B, T, d]
        T = x.shape[1]
        if T > MAX_SOURCE_POSITIONS:
            raise ValueError(
                f"whisper encoder positional embedding only covers {MAX_SOURCE_POSITIONS} "
                f"tokens (30 s audio); got {T}. Reduce chunk_seconds."
            )
        pos_idx = torch.arange(T, device=x.device)
        x = x + enc.embed_positions(pos_idx)

        hiddens: list[Tensor] = [x]
        for layer in enc.layers:
            x = layer(x, attention_mask=None)
            hiddens.append(x)
        # Match the public encoder output: final LayerNorm on the last hidden.
        hiddens[-1] = enc.layer_norm(hiddens[-1])
        return hiddens

    def encode(self, audio: Tensor) -> list[Tensor]:
        """Source-sample-rate audio [B, S] -> list of hidden states."""
        audio_16k = self._resample_to_16k(audio)
        mel = self.log_mel(audio_16k)
        return self._encode_mel(mel)

    # --- losses -------------------------------------------------------------

    def perceptual_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """L1 distance between pred/target activations at selected layers."""
        pred_h = self.encode(pred)
        with torch.no_grad():
            tgt_h = self.encode(target)
        total = pred.new_zeros(())
        for i in self.perceptual_layers:
            # +1 because hiddens[0] is the pre-layer embedding.
            a = pred_h[i + 1]
            b = tgt_h[i + 1].detach()
            # Normalize by per-layer activation scale so deep and shallow
            # layers contribute comparably.
            scale = b.abs().mean().clamp(min=1e-6)
            total = total + F.l1_loss(a, b) / scale
        return total / len(self.perceptual_layers)

    def distill_target(self, target: Tensor) -> Tensor:
        """Final encoder hidden state for the target audio, no grad."""
        with torch.no_grad():
            h = self.encode(target)
        return h[-1].detach()

    def whisper_frame_rate(self) -> float:
        """Encoder output frame rate at 16 kHz (= 50 Hz)."""
        return WHISPER_SR / (HOP_LENGTH * 2)  # conv2 stride = 2
