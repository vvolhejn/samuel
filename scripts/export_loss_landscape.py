"""Precompute the loss landscapes the webapp's /loss-landscape page shows.

Over every pair of tongueIndex, tongueDiameter and lipDiameter, with the third
held at the reference pose -- the page holds it there too -- store the loss
against a target sound, and (unless --no-grad) its gradient with respect to
the two mapped params (autograd through the synth, as in training), which the
page's gradient descent follows. Without stored gradients the page takes
finite differences of the loss grid instead.

The target (--target) is either:

- ``synth``: the synth's own output at one reference config (as in
  notebooks/1104_loss_landscape_2d.ipynb), sharing the candidates' noise seed.
- a recorded vowel (``ah``, ``e-1``, ``e-2``; promo/voice-memos/<name>.wav): a
  steady 2 s from the middle of it, scaled to the training data's level. The
  reference pose is then the published model's fit to that stretch (each
  param's median over its frames), the recording having no true pose.

Losses (--loss), one file per target and loss
(webapp/public/loss-landscape-<target>-<loss>.json), set up as training sets
them up:

- ``mfcc``: the MFCC loss on the middle frame of a short (93 ms) clip.
- ``mel``: the log-mel spectrogram loss, over the same clip.
- ``stft``: the multi-scale log-magnitude STFT loss, over the same clip.
- ``ssl``: the final model's self-supervised loss -- L1 between layer-6
  wav2vec2-base-960h features -- on a 2 s clip. It needs the longer clip: the
  encoder's features depend on context, and a constant pose's loss keeps
  changing with clip length up to ~2 s (along a tongueIndex sweep, the loss on
  1 s clips correlates 0.86 with 4 s, the training chunk length; 2 s, 0.97).

Every pose is constant over time, so the synth computes one impulse response
for the whole clip (a single control frame spanning it) rather than one per
frame; overlap-adding identical per-frame filters is the same convolution.

Run with: uv run --extra server python scripts/export_loss_landscape.py
--target synth --loss mfcc (~6 min on MPS; ssl ~1 h; with --no-grad about a
tenth of that).
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from tqdm import tqdm

from samuel.losses import MelSpecLoss, MFCCLoss, MultiScaleLogMagSTFTLoss
from samuel.model import _DEFAULT_FROZEN_VALUES, _DEFAULT_PARAM_SPEC
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola
from samuel.ssl_loss import SSLFeatureLoss

SAMPLES_PER_FRAME = 512
# 4096 samples -> 5 MFCC frames (n_fft 2048, hop 512); the middle one covers
# samples 1024-3072.
SHORT_SAMPLES = 8 * SAMPLES_PER_FRAME
SSL_SECONDS = 2.0
SEED = 0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SWEPT = ["tongueIndex", "tongueDiameter", "lipDiameter"]
# The model's ranges, except tongueDiameter from 1.75 rather than 1.5: below
# that the tongue all but closes the tract, and the maps' bottom rows turn
# erratic (the sound is mostly gone, but for a narrow opening near index 18).
RANGES = {n: _DEFAULT_PARAM_SPEC[n][:2] for n in SWEPT} | {
    "tongueDiameter": (1.75, 3.5)
}
LOSSES = ["mfcc", "mel", "stft", "ssl"]

CONFIG = {
    **_DEFAULT_FROZEN_VALUES,
    "voiceness": 0.64,
    "intensity": 0.74,
    "tongueIndex": 30.49,
    "tongueDiameter": 3.0,
    "constrictionIndex": 33.87,
    "constrictionDiameter": 2.01,
    "lipDiameter": 2.62,
    "frequency": 108.78,
}

RECORDINGS_DIR = Path("promo/voice-memos")
# Start (s) of a steady 2 s in the middle of each recording's vowel.
RECORDINGS = {"ah": 3.4, "e-1": 2.9, "e-2": 2.9}


def synth(overrides: dict[str, torch.Tensor], n_samples: int) -> torch.Tensor:
    """``overrides`` maps param name -> [B] values; returns [B, n_samples] audio
    of that constant pose, from a single impulse response."""
    B = len(next(iter(overrides.values())))
    base = torch.tensor([CONFIG[n] for n in PARAM_NAMES], device=DEVICE)
    params = base.expand(B, 1, -1).clone()
    for name, values in overrides.items():
        params[..., PARAM_NAMES.index(name)] = values.to(DEVICE)[:, None]
    # A [B] tensor of equal seeds gives every row the same noise; a scalar seed
    # would draw a different noise row per batch slot.
    seed = torch.full((B,), SEED)
    return pink_trombone_ola(params, seed=seed, control_rate=SAMPLE_RATE / n_samples)


def loss_features(loss: str):
    """``(n_samples, features)``: the clip length the loss compares, and audio
    ``[B, n_samples]`` -> a list of feature tensors, the loss being the mean
    over the list of each one's L1 distance."""
    if loss == "ssl":
        # As configured for the published model (configs/config.yaml, loss.ssl_*).
        ssl = SSLFeatureLoss("facebook/wav2vec2-base-960h", layer=6, distance="L1")
        ssl = ssl.to(DEVICE)
        return round(SSL_SECONDS * SAMPLE_RATE), lambda x: [ssl._features(x)]
    # The spectral losses stay on CPU: torch.stft on MPS crashed the kernel.
    if loss == "mfcc":
        mfcc = MFCCLoss(samples_per_frame=SAMPLES_PER_FRAME, n_fft=2048)

        def features(x):
            feats = mfcc.features(x.cpu())  # [B, n_mfcc, T]
            return [feats[..., feats.shape[-1] // 2]]

    elif loss == "mel":
        mel = MelSpecLoss(samples_per_frame=SAMPLES_PER_FRAME)

        def features(x):
            return [mel.features(x.cpu())]

    else:
        stft = MultiScaleLogMagSTFTLoss()

        def features(x):
            # MultiScaleLogMagSTFTLoss.forward's terms, kept per batch item.
            x = x.cpu()
            out = []
            for n_fft in stft.n_ffts:
                window = getattr(stft, f"window_{n_fft}")
                s = torch.stft(
                    x,
                    n_fft=n_fft,
                    hop_length=n_fft // stft.hop_div,
                    window=window,
                    return_complex=True,
                ).abs()
                out.append(torch.log1p(s))
            return out

    return SHORT_SAMPLES, features


def recording_target(name: str) -> tuple[torch.Tensor, dict[str, float]]:
    """A recording's steady 2 s (``[1, S]``, at the training level) and the
    published model's fit to it."""
    import librosa
    import numpy as np

    # Brings in the published checkpoint and the inference path the webapp uses.
    from samuel import server

    server._model = server._load_model()
    y, _ = librosa.load(RECORDINGS_DIR / f"{name}.wav", sr=SAMPLE_RATE)
    n = round(SSL_SECONDS * SAMPLE_RATE)
    seg = y[round(RECORDINGS[name] * SAMPLE_RATE) :][:n]
    seg = server._rms_normalize(seg, server._target_rms())
    fit = server._mimic(seg)["params"]
    return torch.from_numpy(seg)[None], {
        n: float(np.median(fit[n])) for n in PARAM_NAMES
    }


def write_target_audio(target: str, audio: torch.Tensor) -> Path:
    """The target's 2 s, for the page's listen button, at a common loudness."""
    import soundfile

    path = Path(f"webapp/public/loss-landscape-{target}.wav")
    a = audio[0].detach().cpu()
    a = a * (0.05 / a.pow(2).mean().sqrt())  # RMS -26 dBFS
    soundfile.write(path, a.clamp(-1, 1).numpy(), SAMPLE_RATE, subtype="PCM_16")
    return path


def rounded(chunks: list[torch.Tensor]) -> list[float]:
    return [round(v, 5) for v in torch.cat(chunks).tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["synth", *RECORDINGS], required=True)
    parser.add_argument("--loss", choices=LOSSES, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="default: webapp/public/loss-landscape-<target>-<loss>.json",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="default: 128, or 32 for ssl"
    )
    parser.add_argument("--n", type=int, default=64, help="grid points per axis")
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="only write the target's audio (webapp/public/loss-landscape-<target>.wav)",
    )
    parser.add_argument(
        "--no-grad",
        action="store_true",
        help="store the loss only; the page then takes finite differences",
    )
    args = parser.parse_args()
    out_path = args.out or Path(
        f"webapp/public/loss-landscape-{args.target}-{args.loss}.json"
    )
    batch_size = args.batch_size or (32 if args.loss == "ssl" else 128)

    reference = {n: torch.tensor([CONFIG[n]]) for n in SWEPT}
    if args.target == "synth":
        with torch.no_grad():
            audio = synth(reference, round(SSL_SECONDS * SAMPLE_RATE))
    else:
        audio, fit = recording_target(args.target)
        CONFIG.update(fit)
    print(f"wrote {write_target_audio(args.target, audio)}")
    if args.audio_only:
        return

    n_samples, features = loss_features(args.loss)
    with torch.no_grad():
        if args.target == "synth":
            ref = features(synth(reference, n_samples))
        else:
            # The short clips are the middle of the 2 s.
            start = (audio.shape[-1] - n_samples) // 2
            ref = features(audio[:, start : start + n_samples].to(DEVICE))

    axes = [torch.linspace(*RANGES[n], args.n) for n in SWEPT]
    pairs = list(itertools.combinations(range(len(SWEPT)), 2))
    progress = tqdm(total=len(pairs) * args.n**2)
    slices = []
    for x, y in pairs:
        yy, xx = torch.meshgrid(axes[y], axes[x], indexing="ij")
        losses, grads_x, grads_y = [], [], []
        for xc, yc in zip(
            xx.flatten().split(batch_size), yy.flatten().split(batch_size)
        ):
            xc = xc.to(DEVICE).requires_grad_(not args.no_grad)
            yc = yc.to(DEVICE).requires_grad_(not args.no_grad)
            with torch.set_grad_enabled(not args.no_grad):
                feats = features(synth({SWEPT[x]: xc, SWEPT[y]: yc}, n_samples))
                loss = sum(
                    (f - r).abs().flatten(1).mean(dim=-1) for f, r in zip(feats, ref)
                ) / len(feats)
            if not args.no_grad:
                # Batch items are independent, so the gradient of the sum is
                # each item's own gradient.
                gx, gy = torch.autograd.grad(loss.sum(), [xc, yc])
                grads_x.append(gx.cpu())
                grads_y.append(gy.cpu())
            losses.append(loss.detach().cpu())
            progress.update(len(xc))
        # Rows of y (from lo up), each over x. Gradients are per unit of each param.
        s = {"x": x, "y": y, "loss": rounded(losses)}
        if not args.no_grad:
            s |= {"grad_x": rounded(grads_x), "grad_y": rounded(grads_y)}
        slices.append(s)
    progress.close()
    all_losses = [v for s in slices for v in s["loss"]]

    out = {
        "reference": {n: CONFIG[n] for n in PARAM_NAMES if n in CONFIG},
        # A recording's reference is a fit, not the target itself.
        "target": args.target,
        "params": [
            {"key": n, "lo": float(a[0]), "hi": float(a[-1]), "n": len(a)}
            for n, a in zip(SWEPT, axes)
        ],
        "slices": slices,
    }
    out_path.write_text(json.dumps(out, separators=(",", ":")))
    print(
        f"wrote {out_path} ({out_path.stat().st_size / 1e3:.0f} kB), "
        f"loss range {min(all_losses):.3f}..{max(all_losses):.3f}"
    )


if __name__ == "__main__":
    main()
