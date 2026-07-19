"""Perceptual loss on a frozen self-supervised speech encoder.

Instead of comparing raw spectra, we compare the hidden activations of a frozen
SSL speech model (WavLM / HuBERT / wav2vec2). These representations are far more
*phonetic* than a mel/MFCC spectrum: a smeared /t/ is cheap in log-mag STFT but
expensive in feature space. This is the loss shown to beat spectrogram distance
for speech enhancement in Close et al., "Perceive and predict" (2023), where
WavLM/HuBERT/XLSR feature distances outperform spectral losses.

The encoder is frozen and kept in eval mode: gradients flow *through* it into the
audio, never into its weights. The whole path stays differentiable —

    controller -> pink_trombone (44.1 kHz) -> resample 16 kHz -> frozen SSL -> L1

so this can be used as a training loss, not just an offline metric (that's what
the faster-whisper WER eval is for).

The model is chosen by HF name so we can swap encoders without touching callers:
``microsoft/wavlm-base-plus`` (default), ``facebook/hubert-base-ls960``,
``facebook/wav2vec2-base-960h``, etc.
"""

from __future__ import annotations

import julius
import torch
from torch import Tensor, nn

_SSL_SR = 16000


class SSLFeatureLoss(nn.Module):
    """Distance between frozen-SSL-encoder features of ``pred`` and ``target``.

    Args:
        model_name: HuggingFace model id of the SSL encoder.
        layer: Which hidden state to compare. ``0`` is the feature-projection
            output; ``1..N`` are transformer layer outputs. Mid layers (~6-9 for
            base models) are the most phonetic; the last layer drifts toward the
            pretraining objective. Pass ``-1`` for the last layer.
        distance: ``"L1"`` (default), ``"L2"``, or ``"cosine"``.
        source_sr: Sample rate of the waveforms handed to ``forward`` (44.1 kHz
            Pink-Trombone output).
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        layer: int = 6,
        distance: str = "L1",
        source_sr: int = 44100,
    ) -> None:
        super().__init__()
        if distance not in ("L1", "L2", "cosine"):
            raise ValueError(f"unknown distance {distance!r}")
        self.model_name = model_name
        self.layer = layer
        self.distance = distance
        self.is_whisper = "whisper" in model_name.lower()

        if self.is_whisper:
            # Whisper is encoder-decoder and takes a log-mel spectrogram, not a
            # raw waveform. Keep only the encoder; build a differentiable log-mel
            # that matches HF's WhisperFeatureExtractor so the frozen encoder sees
            # the input distribution it was trained on, with the graph intact.
            from transformers import WhisperFeatureExtractor, WhisperModel

            fe = WhisperFeatureExtractor.from_pretrained(model_name)
            self._n_fft = fe.n_fft
            self._hop = fe.hop_length
            self._n_samples = fe.n_samples  # 30 s * 16 kHz = 480000
            self.register_buffer("_window", torch.hann_window(fe.n_fft))
            self.register_buffer(
                "_mel_filters", torch.tensor(fe.mel_filters, dtype=torch.float32)
            )  # [n_fft//2+1, n_mels]
            # large-v3's checkpoint is stored fp16; training is fp32, so upcast
            # the frozen encoder to keep dtypes consistent through the graph.
            self.model = WhisperModel.from_pretrained(model_name).encoder.float()
        else:
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)

        # julius sinc resampler is differentiable and pure-torch (no torchaudio).
        self.resample = julius.ResampleFrac(source_sr, _SSL_SR)

    def _log_mel(self, x16: Tensor) -> tuple[Tensor, int]:
        """[B, S16] 16 kHz waveform -> ([B, n_mels, 3000] log-mel, valid_frames).

        Replicates WhisperFeatureExtractor: pad/truncate to 30 s, power STFT
        (drop last frame), mel projection, log10, per-utterance dynamic-range
        clamp and (log+4)/4 scaling. ``valid_frames`` is how many mel frames hold
        real audio (the rest is 30 s zero-padding) so callers can crop it away.
        """
        S = x16.shape[-1]
        valid_frames = min(S // self._hop, self._n_samples // self._hop)
        if S < self._n_samples:
            x16 = torch.nn.functional.pad(x16, (0, self._n_samples - S))
        else:
            x16 = x16[..., : self._n_samples]
        stft = torch.stft(
            x16, self._n_fft, self._hop, window=self._window, return_complex=True
        )
        mags = stft[..., :-1].abs() ** 2  # [B, freq, 3000]
        mel = self._mel_filters.T @ mags  # [B, n_mels, 3000]
        log = torch.clamp(mel, min=1e-10).log10()
        # per-utterance dynamic-range floor, then Whisper's fixed scaling
        peak = log.amax(dim=(-2, -1), keepdim=True)
        log = torch.maximum(log, peak - 8.0)
        log = (log + 4.0) / 4.0
        return log, valid_frames

    def _features(self, wav: Tensor) -> Tensor:
        """[B, S] waveform at source_sr -> [B, T, D] SSL hidden states."""
        x = self.resample(wav)
        if self.is_whisper:
            mel, valid = self._log_mel(x)
            out = self.model(mel, output_hidden_states=True)
            feats = out.hidden_states[self.layer]  # [B, 1500, D]
            # Encoder conv stack downsamples time by 2; crop off the 30 s padding.
            valid_enc = max(1, valid // 2)
            return feats[:, :valid_enc]
        # WavLM/HuBERT/wav2vec2 were trained on zero-mean unit-variance input;
        # the HF feature extractor does this per-utterance. Replicate in torch
        # so the graph stays intact.
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)
        out = self.model(x, output_hidden_states=True)
        return out.hidden_states[self.layer]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """``pred``/``target`` are ``[B, S]`` waveforms at ``source_sr``."""
        f_pred = self._features(pred)
        with torch.no_grad():
            f_tgt = self._features(target)

        # Feature lengths should match (equal input length), but guard against
        # off-by-one from the conv stack.
        T = min(f_pred.shape[1], f_tgt.shape[1])
        f_pred, f_tgt = f_pred[:, :T], f_tgt[:, :T]

        if self.distance == "L1":
            return (f_pred - f_tgt).abs().mean()
        if self.distance == "L2":
            return (f_pred - f_tgt).pow(2).mean()
        # cosine distance, averaged over frames
        cos = torch.nn.functional.cosine_similarity(f_pred, f_tgt, dim=-1)
        return (1.0 - cos).mean()
