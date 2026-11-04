"""Precompute the loss landscape the webapp's /loss-landscape page shows.

Same setup as notebooks/1104_loss_landscape_2d.ipynb (reference config, middle-frame
MFCC loss, shared noise seed), over every pair of tongueIndex, tongueDiameter and
lipDiameter with the third held at the reference -- the page holds it there too.
Alongside the loss it stores the loss's gradient with respect to the two mapped
params (autograd through the synth, as in training), which the page's gradient
descent follows.

Run with: uv run python scripts/export_loss_landscape.py
"""

import argparse
import itertools
import json
from pathlib import Path

import torch
from tqdm import tqdm

from samuel.losses import MFCCLoss
from samuel.model import _DEFAULT_FROZEN_VALUES, _DEFAULT_PARAM_SPEC
from samuel.pink_trombone import PARAM_NAMES, SAMPLE_RATE, pink_trombone_ola

SAMPLES_PER_FRAME = 512
CONTROL_RATE = SAMPLE_RATE / SAMPLES_PER_FRAME
N_FRAMES = 8
SEED = 0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SWEPT = ["tongueIndex", "tongueDiameter", "lipDiameter"]

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

loss_fn = MFCCLoss(samples_per_frame=SAMPLES_PER_FRAME, n_fft=2048)


def make_params(overrides: dict[str, torch.Tensor]) -> torch.Tensor:
    """``overrides`` maps param name -> [B] values; returns [B, N_FRAMES, 12]."""
    B = len(next(iter(overrides.values())))
    base = torch.tensor([CONFIG[n] for n in PARAM_NAMES], device=DEVICE)
    params = base.expand(B, N_FRAMES, -1).clone()
    for name, values in overrides.items():
        params[..., PARAM_NAMES.index(name)] = values.to(DEVICE)[:, None]
    return params


def middle_frame_features(params: torch.Tensor) -> torch.Tensor:
    seed = torch.full((len(params),), SEED)
    audio = pink_trombone_ola(params, seed=seed, control_rate=CONTROL_RATE).cpu()
    feats = loss_fn.features(audio)  # [B, n_mfcc, T]
    return feats[..., feats.shape[-1] // 2]


def rounded(chunks: list[torch.Tensor]) -> list[float]:
    return [round(v, 5) for v in torch.cat(chunks).tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("webapp/public/loss-landscape.json")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n", type=int, default=64, help="grid points per axis")
    args = parser.parse_args()

    axes = [torch.linspace(*_DEFAULT_PARAM_SPEC[n][:2], args.n) for n in SWEPT]
    with torch.no_grad():
        ref = middle_frame_features(
            make_params({n: torch.tensor([CONFIG[n]]) for n in SWEPT})
        )
    slices = []
    for x, y in tqdm(list(itertools.combinations(range(len(SWEPT)), 2))):
        yy, xx = torch.meshgrid(axes[y], axes[x], indexing="ij")
        losses, grads_x, grads_y = [], [], []
        for xc, yc in zip(
            xx.flatten().split(args.batch_size), yy.flatten().split(args.batch_size)
        ):
            xc = xc.to(DEVICE).requires_grad_()
            yc = yc.to(DEVICE).requires_grad_()
            feats = middle_frame_features(make_params({SWEPT[x]: xc, SWEPT[y]: yc}))
            loss = (feats - ref).abs().mean(dim=-1)
            # Batch items are independent, so the gradient of the sum is each
            # item's own gradient.
            gx, gy = torch.autograd.grad(loss.sum(), [xc, yc])
            losses.append(loss.detach())
            grads_x.append(gx.cpu())
            grads_y.append(gy.cpu())
        # Rows of y (from lo up), each over x. Gradients are per unit of each param.
        slices.append(
            {
                "x": x,
                "y": y,
                "loss": rounded(losses),
                "grad_x": rounded(grads_x),
                "grad_y": rounded(grads_y),
            }
        )
    all_losses = [v for s in slices for v in s["loss"]]

    out = {
        "reference": {n: CONFIG[n] for n in PARAM_NAMES if n in CONFIG},
        "params": [
            {"key": n, "lo": float(a[0]), "hi": float(a[-1]), "n": len(a)}
            for n, a in zip(SWEPT, axes)
        ],
        "slices": slices,
    }
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print(
        f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB), loss range {min(all_losses):.3f}..{max(all_losses):.3f}"
    )


if __name__ == "__main__":
    main()
