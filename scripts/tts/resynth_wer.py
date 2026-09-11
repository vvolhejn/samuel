"""Codec resynthesis WER: the intelligibility ceiling of a TTS trained on it.

Encodes LibriSpeech test-clean utterances with the run's codec, decodes them
back to audio, transcribes with the same ASR as training/eval/librispeech.py
and reports corpus WER against the reference transcripts. Any TTS on this
codec inherits this error floor, so read TTS WERs relative to it.

Usage:
    uv run python scripts/tts/resynth_wer.py configs/tts/a_unstacked.yaml \\
        --librispeech-root /lustre/scwpod02/client/kyutai/datasets/LibriSpeech/test-clean
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import jiwer
import sphn
import torch
from tqdm import tqdm
from training.args import load_args
from training.eval.librispeech import DEFAULT_ASR, build_transcriber, load_mono
from training.modules.builders import build_codec


def read_transcripts(root: Path) -> list[tuple[Path, str]]:
    items = []
    for trans in sorted(root.glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            items.append((trans.parent / f"{utt}.flac", text))
    return items


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="training config naming the codec")
    ap.add_argument("--librispeech-root", required=True)
    ap.add_argument("--num-items", type=int, default=200)
    ap.add_argument("--asr", default=DEFAULT_ASR)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-audio", default=None)
    ap.add_argument(
        "--out", default=None, help="results json (default: next to the config)"
    )
    args = ap.parse_args()
    from whisper_normalizer.english import EnglishTextNormalizer

    normalize = EnglishTextNormalizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = build_codec(load_args(args.config)).to(device)
    transcribe = build_transcriber(args.asr, device)
    items = read_transcripts(Path(args.librispeech_root))
    items = items[:: max(1, len(items) // args.num_items)][: args.num_items]

    refs, hyps_codec, hyps_orig = [], [], []
    for i in tqdm(range(0, len(items), args.batch_size)):
        chunk = items[i : i + args.batch_size]
        wavs = [load_mono(str(p), codec.sample_rate) for p, _ in chunk]
        audio = torch.zeros(len(wavs), 1, max(len(w) for w in wavs))
        for b, w in enumerate(wavs):
            audio[b, 0, : len(w)] = w
        with torch.no_grad():
            out = codec.decode_to_audio(codec.encode_to_latent(audio.to(device)))
        gens, origs = [], []
        for b, w in enumerate(wavs):
            gen = out[b, 0, : len(w)].cpu().numpy()
            gens.append(
                sphn.resample(
                    gen, src_sample_rate=codec.sample_rate, dst_sample_rate=16000
                )
            )
            origs.append(
                sphn.resample(
                    w.numpy(), src_sample_rate=codec.sample_rate, dst_sample_rate=16000
                )
            )
            if args.save_audio:
                Path(args.save_audio).mkdir(parents=True, exist_ok=True)
                sphn.write_wav(
                    str(Path(args.save_audio) / chunk[b][0].with_suffix(".wav").name),
                    gen,
                    codec.sample_rate,
                )
        hyps_codec += [normalize(h) for h in transcribe(gens)]
        hyps_orig += [normalize(h) for h in transcribe(origs)]
        refs += [normalize(t) for _, t in chunk]
    results = {
        "config": args.config,
        "asr": args.asr,
        "num_items": len(refs),
        "wer_resynth": jiwer.wer(refs, hyps_codec),
        "wer_original": jiwer.wer(refs, hyps_orig),
    }
    print(json.dumps(results, indent=2))
    out = (
        Path(args.out)
        if args.out
        else Path(args.config).with_suffix(".resynth_wer.json")
    )
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
