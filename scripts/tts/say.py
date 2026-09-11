"""Synthesize text with a Samuel-codec TTS run: text -> trajectories -> wav.

Usage:
    uv run python scripts/tts/say.py runs/tts/b_stack8 "Hello there." out.wav \\
        [--voice prompt.wav] [--use-ema] [--temp 0.3] [--cfg 2.0] [--params out.json]
"""

from __future__ import annotations

import argparse
import json

import sphn
import torch
from training.eval.librispeech import load_mono, load_run

from samuel.pink_trombone import PARAM_NAMES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("text")
    ap.add_argument("out_wav")
    ap.add_argument(
        "--voice", default=None, help="voice prompt wav (default: no prompt)"
    )
    ap.add_argument("--voice-sec", type=float, default=5.0)
    ap.add_argument("--use-ema", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--temp", type=float, default=0.3)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--eos-threshold", type=float, default=-1.0)
    ap.add_argument("--max-sec", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--params", default=None, help="also dump the Pink Trombone trajectory as json"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, codec, step = load_run(
        args.run_dir, device, use_ema=args.use_ema, checkpoint=args.checkpoint
    )
    torch.manual_seed(args.seed)
    tokens = torch.tensor(
        model.flow_lm.conditioner.tokenizer.sp.encode(args.text), dtype=torch.long
    )
    if args.voice:
        wav = load_mono(args.voice, codec.sample_rate)[
            : int(args.voice_sec * codec.sample_rate)
        ]
        with torch.no_grad():
            voice = codec.encode_to_latent(wav[None, None].to(device))[0]
    else:
        voice = torch.zeros(0, codec.latent_dim, device=device)
    with torch.no_grad():
        (latents,) = model.generate(
            [tokens],
            [voice],
            max_frames=int(args.max_sec * codec.frame_rate),
            temp=args.temp,
            cfg_coef=args.cfg,
            eos_threshold=args.eos_threshold,
        )
        if latents.shape[0] < codec.min_decode_frames:
            raise SystemExit("empty generation (EOS on the first frame)")
        audio = codec.decode_to_audio(latents[None].to(device))[0, 0]
    sphn.write_wav(args.out_wav, audio.cpu().numpy(), codec.sample_rate)
    print(
        f"step {step}: {latents.shape[0]} frames, {audio.shape[0] / codec.sample_rate:.2f}s -> {args.out_wav}"
    )
    if args.params:
        params = codec.latents_to_params(latents[None])[0].cpu()  # ty: ignore[unresolved-attribute]
        json.dump(
            {
                "frame_rate": codec.sample_rate / codec.samples_per_frame,
                "names": PARAM_NAMES,
                "params": params.tolist(),
            },  # ty: ignore[unresolved-attribute]
            open(args.params, "w"),
        )


if __name__ == "__main__":
    main()
