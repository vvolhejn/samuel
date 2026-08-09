"""Pick a weight for ``loss.rest`` by measuring the gradient it competes with.

Loads a checkpoint, runs the eval clips through it, and compares — per control
frame and per parameter — the reconstruction gradient reaching the trajectory
against the constant gradient a unit-weight rest-pose prior would apply:

    d(rest)/d(p_norm[b, t, j]) = rest / (B * T)     (L1, mean-reduced)

so writing ``G = |d(recon)/d(p_norm)| * B * T``, the prior dominates exactly
where ``rest > G``. Reporting the distribution of ``G`` separately on speech
and silent frames gives the usable window: above the silent-frame mass (so the
rest pose actually wins where the recon loss has no opinion) and well below
the speech-frame mass (so phonation is untouched).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from samuel.config import TrainConfig
from samuel.losses import MFCCLoss
from samuel.model import PinkTromboneController
from samuel.pink_trombone import pink_trombone_ola
from samuel.ssl_loss import SSLFeatureLoss
from samuel.train import (
    CombinedReconLoss,
    _acceleration_loss,
    _eval_setup,
    _normalized_trainable_params,
    _rms_normalize,
    _smoothness_loss,
)

REST_TARGETS = {
    "lipDiameter": 0.3,
    "constrictionDiameter": 3.0,
    "tongueDiameter": 3.5,
}
# The weights chosen from this script's output, echoed back so the last block
# reports the force each param actually feels.
REST_WEIGHTS = {"lipDiameter": 0.3, "constrictionDiameter": 1.5, "tongueDiameter": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=str, default="last.pt")
    ap.add_argument("--n-clips", type=int, default=16)
    ap.add_argument("--silence-db", type=float, default=-20.0)
    ap.add_argument("--rest", type=float, default=0.15, help="candidate loss.rest")
    args = ap.parse_args()

    cfg_raw = json.loads((args.run_dir / "config.json").read_text())
    cfg = TrainConfig.from_hydra(OmegaConf.create(cfg_raw))
    cfg.log.asr_whisper_size = ""  # no ASR needed here
    cfg.log.n_eval_clips = args.n_clips
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PinkTromboneController(cfg.model).to(device)
    ckpt = torch.load(args.run_dir / "checkpoints" / args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    # train() mode on purpose: eval snaps to argmax, which is not
    # differentiable. We want the gradients as they actually reach the
    # trajectories during training, i.e. through the soft Gumbel path.
    model.train()
    print(f"loaded step {ckpt.get('step')}")

    setup = _eval_setup(cfg, cfg.model.samples_per_frame, device)
    # Same components the run trains with (mfcc + ssl at their configured weights).
    components = [
        (
            "mfcc",
            cfg.loss.mfcc,
            MFCCLoss(
                samples_per_frame=cfg.model.samples_per_frame,
                n_fft=cfg.loss.mfcc_n_fft,
            ),
        ),
        (
            "ssl",
            cfg.loss.ssl,
            SSLFeatureLoss(
                model_name=cfg.loss.ssl_model,
                layer=cfg.loss.ssl_layer,
                distance=cfg.loss.ssl_distance,
                source_sr=cfg.data.sample_rate,
            ),
        ),
    ]
    loss_fn = CombinedReconLoss(components).to(device)

    target = _rms_normalize(setup.val_wavs.to(device), cfg.data.target_rms)
    f0 = setup.val_f0.to(device)

    # tau: same constant the run trains at, and the soft path so gradients
    # match the training regime rather than eval's argmax.
    params, _ = model(target.unsqueeze(1), f0, cfg.optim.tau_end, return_aux=True)
    params = params.float()
    p_norm = _normalized_trainable_params(params, model)

    pred = pink_trombone_ola(
        params, ir_length=cfg.synth.ir_length, control_rate=cfg.model.frame_rate
    )
    S = min(pred.shape[-1], target.shape[-1])
    recon = loss_fn(pred[..., :S], target[..., :S])
    grad = torch.autograd.grad(recon, params)[0]
    # d/d p_norm = d/d p * (hi - lo) — undo the [0, 1] rescale.
    lo = model.bucket_centers[:, 0]
    hi = model.bucket_centers[:, -1]
    g_raw = grad.index_select(-1, model._trainable_idx) * (hi - lo).view(1, 1, -1)

    B, T, _ = p_norm.shape
    G = g_raw.abs() * (B * T)  # [B, T, n_t]; rest weight is comparable to this

    spf = model.samples_per_frame
    tgt = target[:, : T * spf]
    frame_rms = tgt.reshape(B, T, spf).pow(2).mean(-1).sqrt()
    silent = frame_rms < cfg.data.target_rms * 10 ** (args.silence_db / 20.0)
    print(f"\nclips={B} frames={T} silent_frac={silent.float().mean():.3f}")

    names = model.trainable_names_
    print(
        "\nrecon-gradient scale G = |d recon/d p_norm| * B*T  (rest wins if weight > G)"
    )
    print(f"{'param':22s} {'':>6s} {'p25':>9s} {'median':>9s} {'p75':>9s} {'mean':>9s}")
    for j, n in enumerate(names):
        if n not in REST_TARGETS:
            continue
        for label, mask in (("speech", ~silent), ("silent", silent)):
            v = G[..., j][mask].detach().cpu().numpy()
            print(
                f"{n:22s} {label:>6s} "
                f"{np.percentile(v, 25):9.4f} {np.median(v):9.4f} "
                f"{np.percentile(v, 75):9.4f} {v.mean():9.4f}"
            )
    print("\nrest force vs. that gradient, at the configured weights:")
    for j, n in enumerate(names):
        if n not in REST_TARGETS:
            continue
        force = args.rest * REST_WEIGHTS.get(n, 1.0)
        frac = {
            label: force / float(np.median(G[..., j][mask].detach().cpu().numpy()))
            for label, mask in (("speech", ~silent), ("silent", silent))
        }
        print(
            f"  {n:22s} force={force:.3f}  "
            f"{frac['speech'] * 100:5.1f}% of speech median, "
            f"{frac['silent'] * 100:5.1f}% of silent median"
        )

    # Loss-contribution view: what a given weight adds to the total loss,
    # against the smooth/accel terms already in the recipe.
    tgt_vec = torch.tensor(
        [REST_TARGETS.get(n, 0.0) for n in names], dtype=lo.dtype, device=lo.device
    )
    mask_vec = torch.tensor(
        [1.0 if n in REST_TARGETS else 0.0 for n in names],
        dtype=lo.dtype,
        device=lo.device,
    )
    tgt_norm = (tgt_vec - lo) / (hi - lo)
    dist = (
        (p_norm - tgt_norm.view(1, 1, -1)).abs().mean(dim=(0, 1)) * mask_vec
    ).detach()
    smooth = _smoothness_loss(params, model, cfg.loss.smooth_weights).item()
    accel = _acceleration_loss(params, model, cfg.loss.accel_weights).item()
    print("\nrest-pose distance at this checkpoint (normalised units):")
    for n, v in zip(names, dist.tolist()):
        if n in REST_TARGETS:
            print(f"  {n:22s} target={REST_TARGETS[n]:5.2f} mean|Δ|={v:.3f}")
    print(f"  {'sum (raw rest_loss)':22s} {float(dist.sum()):.3f}")
    print(
        f"\nreference terms: recon={recon.item():.4f} "
        f"smooth={cfg.loss.smooth}*{smooth:.4f}={cfg.loss.smooth * smooth:.4f} "
        f"accel={cfg.loss.accel}*{accel:.4f}={cfg.loss.accel * accel:.4f}"
    )
    for w in (0.003, 0.01, 0.03, 0.1):
        print(f"  rest={w:<6g} -> contribution {w * float(dist.sum()):.4f}")


if __name__ == "__main__":
    main()
