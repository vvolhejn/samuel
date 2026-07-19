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


class ASRDistillLoss(nn.Module):
    """Frame-wise KL between CTC character posteriors of ``pred`` and ``target``.

    Both waveforms go through a frozen CTC-fine-tuned ASR model
    (wav2vec2-base-960h: 32-char vocab, ~50 posterior frames/sec) and we take
    KL(target ‖ pred) per frame. The CTC head is a content bottleneck: unlike
    ``SSLFeatureLoss`` this only constrains *which character is being said
    when*, not timbre/prosody/speaker. Note no CTC marginalization is involved
    — with reference audio available, the frame alignment is given, so this is
    plain posterior distillation.

    Args:
        model_name: HF id of a CTC-fine-tuned ASR model.
        source_sr: Sample rate of the waveforms handed to ``forward``.
        temperature: Softmax temperature applied to both teacher and student
            logits. The CTC head is a confident classifier (~one-hot posteriors),
            and KL against a near-one-hot target has unbounded gradients — the
            resulting spikes saturated the controller's Gumbel head at T=1
            (run ctc-kl_20260719-094658). T>1 caps the teacher's peakedness.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        source_sr: int = 44100,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCTC

        self.model_name = model_name
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)
        # Models with layer-norm feature extractors (e.g. the -lv60 variants)
        # were fine-tuned on per-utterance z-scored input; group-norm ones
        # (base-960h) on raw waveforms. Unlike the symmetric feature loss, the
        # CTC head is scale-sensitive, so match the convention.
        self.do_normalize = self.model.config.feat_extract_norm == "layer"
        self.temperature = temperature
        self.resample = julius.ResampleFrac(source_sr, _SSL_SR)

    def _log_probs(self, wav: Tensor) -> Tensor:
        """[B, S] waveform at source_sr -> [B, T, V] CTC log-posteriors."""
        x = self.resample(wav)
        if self.do_normalize:
            x = (x - x.mean(dim=-1, keepdim=True)) / (
                x.std(dim=-1, keepdim=True) + 1e-7
            )
        return (self.model(x).logits / self.temperature).log_softmax(dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """``pred``/``target`` are ``[B, S]`` waveforms at ``source_sr``."""
        lp_pred = self._log_probs(pred)
        with torch.no_grad():
            lp_tgt = self._log_probs(target)

        T = min(lp_pred.shape[1], lp_tgt.shape[1])
        lp_pred, lp_tgt = lp_pred[:, :T], lp_tgt[:, :T]

        # KL(target ‖ pred) summed over the vocab, averaged over batch+frames.
        kl = (lp_tgt.exp() * (lp_tgt - lp_pred)).sum(dim=-1)
        return kl.mean()


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
        from transformers import AutoModel

        if distance not in ("L1", "L2", "cosine"):
            raise ValueError(f"unknown distance {distance!r}")
        self.model_name = model_name
        self.layer = layer
        self.distance = distance

        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)

        # julius sinc resampler is differentiable and pure-torch (no torchaudio).
        self.resample = julius.ResampleFrac(source_sr, _SSL_SR)

    def _features(self, wav: Tensor) -> Tensor:
        """[B, S] waveform at source_sr -> [B, T, D] SSL hidden states."""
        x = self.resample(wav)
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
