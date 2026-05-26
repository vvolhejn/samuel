"""Whisper-based intelligibility metric (WER / CER vs target transcription).

The reference is Whisper's transcription of the *target* clip; the hypothesis
is Whisper's transcription of the *synthesised* clip. This is a direct proxy
for "can an ASR still parse what the synth says" — far closer to perceptual
intelligibility than MFCC / STFT losses on their own.

The model is lazy-loaded once on the trainer's device. CUDA uses float16
(fast); CPU falls back to int8 quantisation. A CUDA load failure (typically
missing cuDNN / cuBLAS at runtime) is caught and logged, and we fall back to
CPU — eval is slower but training never crashes.
"""

from __future__ import annotations

from dataclasses import dataclass

import jiwer
import librosa
import numpy as np
import torch

_ASR_SR = 16000

_WORD_TF = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)
_CHAR_TF = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


@dataclass
class AsrScores:
    wer: float
    cer: float
    ref: str
    hyp: str


class WhisperEvaluator:
    """faster-whisper wrapper with target-transcript cache.

    The same ``target`` waveform may be scored many times across eval calls
    (the val clip set is fixed for the whole training run). ``score`` keys
    transcripts on the target's ``id(...)`` so we only run Whisper on each
    target once.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._target_cache: dict[int, str] = {}
        self.model_size = model_size
        self.model = _load_whisper(model_size, device)

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        if sr != _ASR_SR:
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=sr, target_sr=_ASR_SR
            )
        # faster-whisper occasionally trips on non-finite inputs (synth at
        # pathological tract shapes); guard before handing it the buffer.
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    def transcribe_target(
        self, target: np.ndarray, sr: int, key: int | None = None
    ) -> str:
        """Cached transcription of a target clip.

        ``key`` is the cache key — a stable per-clip identifier (e.g. the
        full-manifest file index). When ``None``, the array's ``id`` is used
        which only deduplicates within a single eval call.
        """
        cache_key = key if key is not None else id(target)
        if cache_key not in self._target_cache:
            self._target_cache[cache_key] = self.transcribe(target, sr)
        return self._target_cache[cache_key]

    def score(
        self,
        target: np.ndarray,
        pred: np.ndarray,
        sr: int,
        target_key: int | None = None,
    ) -> AsrScores:
        ref = self.transcribe_target(target, sr, key=target_key)
        hyp = self.transcribe(pred, sr)
        return _score_text(ref, hyp)


def _score_text(ref: str, hyp: str) -> AsrScores:
    if not ref.strip():
        return AsrScores(wer=float("nan"), cer=float("nan"), ref=ref, hyp=hyp)
    try:
        wer = float(
            jiwer.wer(
                ref, hyp, reference_transform=_WORD_TF, hypothesis_transform=_WORD_TF
            )
        )
    except ValueError:
        wer = float("nan")
    try:
        cer = float(
            jiwer.cer(
                ref, hyp, reference_transform=_CHAR_TF, hypothesis_transform=_CHAR_TF
            )
        )
    except ValueError:
        cer = float("nan")
    return AsrScores(wer=wer, cer=cer, ref=ref, hyp=hyp)


def _load_whisper(model_size: str, device: str):
    """Load faster-whisper, falling back to CPU if CUDA init fails.

    A CUDA load can fail at runtime when ctranslate2 can't find cuDNN /
    cuBLAS, even though torch finds CUDA. We want training to keep running
    if that happens, not crash; the WER eval just gets slower.
    """
    from faster_whisper import WhisperModel

    if device == "cuda":
        try:
            return WhisperModel(model_size, device="cuda", compute_type="float16")
        except Exception as e:  # noqa: BLE001  (we want every CUDA failure mode)
            print(
                f"[asr] WhisperModel(cuda) failed ({type(e).__name__}: {e}); "
                f"falling back to CPU."
            )
    return WhisperModel(model_size, device="cpu", compute_type="int8")
