"""Decompose pink_trombone_ola forward time into sub-stages.

Run with: uv run python scripts/benchmark_synth.py
"""

from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig

from samuel.config import TrainConfig
from samuel.data import build_dataloader
from samuel.model import PinkTromboneController
from samuel.pink_trombone import (
    _NOSE_R_CPU,
    _NOSE_START,
    _TRACT_N,
    _compute_batch_irs,
    _compute_batch_irs_eig,
    _compute_diameter_profile,
    _ola_convolve,
    _samples_per_frame,
    _upsample_params,
    PARAM_NAMES,
    SimplexNoise,
    glottis,
)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None
    torch.manual_seed(0)

    model = PinkTromboneController(cfg.model).to(device)

    loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        sample_rate=cfg.data.sample_rate,
        chunk_seconds=cfg.data.chunk_seconds,
        rank=0,
        world_size=1,
        epoch=0,
        seed=0,
    )

    # One warm batch
    batch = next(iter(loader)).to(device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        params = model(batch.unsqueeze(1))
    params = params.float().detach().requires_grad_(True)
    B, T, P = params.shape
    print(f"params shape: {tuple(params.shape)}  (B={B}, T={T}, P={P})")

    ir_length = cfg.synth.ir_length
    control_rate = cfg.model.frame_rate
    ir_impl = cfg.synth.ir_impl
    print(f"ir_length={ir_length}, control_rate={control_rate}, ir_impl={ir_impl}")

    n_trials = 5
    warmup = 2

    times: dict[str, list[float]] = {}

    def timed(label: str, fn):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        times.setdefault(label, []).append(time.perf_counter() - t0)
        return out

    for trial in range(n_trials):
        seed = torch.randint(0, 2**31, (B,))
        simplex = timed("simplex_init", lambda: SimplexNoise(device=device, seed=seed))
        spf = _samples_per_frame(control_rate)
        params_up = timed("upsample_params", lambda: _upsample_params(params, spf))

        p = {name: params_up[..., i] for i, name in enumerate(PARAM_NAMES)}

        glottis_out, noise_mod = timed(
            "glottis",
            lambda: glottis(
                noise_param=p["noise"],
                frequency=p["frequency"],
                tenseness=p["tenseness"],
                intensity=p["intensity"],
                loudness=p["loudness"],
                vibrato_wobble=p["vibratoWobble"],
                vibrato_freq=p["vibratoFrequency"],
                vibrato_gain=p["vibratoGain"],
                simplex=simplex,
            ),
        )

        # --- inline _tract_ola pieces ---
        N = _TRACT_N
        ns = _NOSE_START
        hop = spf
        S = glottis_out.shape[-1]
        T_f = S // hop

        mid = torch.arange(T_f, device=device) * hop + hop // 2
        ti_f = p["tongueIndex"][:, mid]
        td_f = p["tongueDiameter"][:, mid]
        ci_f = p["constrictionIndex"][:, mid]
        cd_f = p["constrictionDiameter"][:, mid]

        def compute_rf():
            diameter_f = _compute_diameter_profile(ti_f, td_f, ci_f, cd_f, N)
            amplitude_f = diameter_f**2
            r_inner = (amplitude_f[:, :, :-1] - amplitude_f[:, :, 1:]) / (
                amplitude_f[:, :, :-1] + amplitude_f[:, :, 1:] + 1e-10
            )
            r_f = torch.cat(
                [
                    torch.zeros(B, T_f, 1, device=device, dtype=glottis_out.dtype),
                    r_inner,
                ],
                dim=2,
            )
            velum_f = torch.where(
                (ci_f > ns) & (cd_f < -0.2),
                torch.full_like(ci_f, 0.4),
                torch.full_like(ci_f, 0.01),
            )
            A_L_f = amplitude_f[:, :, ns]
            A_R_f = amplitude_f[:, :, ns + 1]
            A_N_f = velum_f**2
            sum_A_f = A_L_f + A_R_f + A_N_f + 1e-10
            r_L_f = (2 * A_L_f - sum_A_f) / sum_A_f
            r_R_f = (2 * A_R_f - sum_A_f) / sum_A_f
            r_N_f = (2 * A_N_f - sum_A_f) / sum_A_f
            return r_f, r_L_f, r_R_f, r_N_f

        r_f, r_L_f, r_R_f, r_N_f = timed("compute_r_coefficients", compute_rf)

        BT = B * T_f
        r_flat = r_f.reshape(BT, N)
        r_L_flat = r_L_f.reshape(BT)
        r_R_flat = r_R_f.reshape(BT)
        r_N_flat = r_N_f.reshape(BT)
        ci_flat = (ci_f.reshape(BT).detach().floor().long() + 1).clamp(0, N - 1)
        nose_r = _NOSE_R_CPU.to(device=device, dtype=glottis_out.dtype)

        _ir_fn = _compute_batch_irs_eig if ir_impl == "eig" else _compute_batch_irs

        h_glottis_flat = timed(
            "compute_ir_glottis",
            lambda: _ir_fn(
                r_flat,
                r_L_flat,
                r_R_flat,
                r_N_flat,
                nose_r,
                inject_glottis=True,
                inject_pos=ci_flat,
                L=ir_length,
            ),
        )
        h_turb_flat = timed(
            "compute_ir_turb",
            lambda: _ir_fn(
                r_flat,
                r_L_flat,
                r_R_flat,
                r_N_flat,
                nose_r,
                inject_glottis=False,
                inject_pos=ci_flat,
                L=ir_length,
            ),
        )

        h_glottis = h_glottis_flat.reshape(B, T_f, ir_length)
        h_turb = h_turb_flat.reshape(B, T_f, ir_length)

        thinness = torch.clamp(8 * (0.7 - p["constrictionDiameter"]), 0.0, 1.0)
        openness = torch.clamp(30 * (p["constrictionDiameter"] - 0.3), 0.0, 1.0)
        valid_mask = (
            (p["constrictionIndex"] >= 2)
            & (p["constrictionIndex"] <= N)
            & (p["constrictionDiameter"] > 0)
        ).to(glottis_out.dtype)
        turb_source = (
            noise_mod * glottis_out * 0.66 * (thinness * openness) / 2 * valid_mask
        )

        oral = timed(
            "ola_convolve_oral", lambda: _ola_convolve(glottis_out, h_glottis, hop)
        )
        turb_out = timed(
            "ola_convolve_turb", lambda: _ola_convolve(turb_source, h_turb, hop)
        )

        out = (oral + turb_out) * 0.125
        loss = out.pow(2).mean()

        def do_backward():
            loss.backward(retain_graph=(trial < n_trials - 1))
            if params.grad is not None:
                params.grad = None

        timed("backward", do_backward)
        print(f"trial {trial}: out.shape={tuple(out.shape)}")

    print(
        f"\n=== pink_trombone_ola stage breakdown (averaged over trials {warmup + 1}..n) ==="
    )
    total = 0.0
    for label, xs in times.items():
        xs = xs[warmup:]
        if not xs:
            continue
        mean = sum(xs) / len(xs)
        total += mean
        print(
            f"  {label:<24s} {mean * 1000:9.1f} ms  (min {min(xs) * 1000:7.1f}, max {max(xs) * 1000:7.1f})"
        )
    print(f"  {'TOTAL (sum of stages)':<24s} {total * 1000:9.1f} ms")


if __name__ == "__main__":
    main()
