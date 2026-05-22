"""Unit tests for the ASR scoring helper.

Does NOT touch the Whisper model (those require a ~150 MB download and add
network/disk variance to the unit suite). The WhisperEvaluator import is
smoke-checked, but transcription is exercised via integration in the smoke
training run.
"""

from __future__ import annotations

import math

from samuel.evals.asr import _score_text


def test_identical_text_zero_wer_cer():
    s = _score_text("hello world this is fine", "hello world this is fine")
    assert s.wer == 0.0
    assert s.cer == 0.0


def test_empty_ref_returns_nan():
    s = _score_text("", "hello")
    assert math.isnan(s.wer)
    assert math.isnan(s.cer)


def test_case_and_punct_ignored():
    s = _score_text("Hello, world.", "hello world")
    assert s.wer == 0.0


def test_completely_different_text_wer_one():
    # Same length, every word substituted → WER == 1.
    s = _score_text("alpha beta gamma delta", "one two three four")
    assert s.wer == 1.0


def test_import_evaluator_class():
    # Don't instantiate (would download Whisper); just confirm the symbol
    # exists so refactors don't silently break the public surface.
    from samuel.evals.asr import WhisperEvaluator  # noqa: F401
