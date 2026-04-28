"""Hydra entry point for training the Pink Trombone controller.

Single-node, DDP-capable. Launch examples:

    # 1 GPU
    uv run python -m samuel.train run.name=smoke optim.max_steps=20

    # N GPUs on this node
    uv run torchrun --standalone --nproc_per_node=2 -m samuel.train run.name=multi
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import plotly.graph_objects as go
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from samuel.config import TrainConfig
from samuel.data import (
    _load_pitch_cache,
    _load_resampled,
    build_dataloader,
    fill_unvoiced,
    load_manifest,
)
from samuel.evals.pitch import pitch_mae_cents
from samuel.losses import MFCCLoss
from samuel.model import PinkTromboneController
from samuel.pink_trombone import PARAM_NAMES, pink_trombone, pink_trombone_ola


def _ddp_info() -> tuple[int, int, int, bool]:
    """Return (rank, local_rank, world_size, is_ddp)."""
    if "LOCAL_RANK" in os.environ:
        return (
            int(os.environ["RANK"]),
            int(os.environ["LOCAL_RANK"]),
            int(os.environ["WORLD_SIZE"]),
            True,
        )
    return 0, 0, 1, False


def _broadcast_str(s: str, src: int = 0, is_ddp: bool = False) -> str:
    if not is_ddp:
        return s
    obj = [s]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]


def _make_run_dir(cfg: TrainConfig, rank: int, is_ddp: bool) -> Path:
    if rank == 0:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = cfg.run.runs_root / f"{ts}_{cfg.run.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            f.write(cfg.model_dump_json(indent=2))
        path_str = str(run_dir)
    else:
        path_str = ""
    path_str = _broadcast_str(path_str, src=0, is_ddp=is_ddp)
    return Path(path_str)


def _warmup_lr(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return 1.0


def _tau_for_step(step: int, cfg: TrainConfig) -> float:
    anneal = cfg.optim.tau_anneal_steps or cfg.optim.max_steps
    if anneal <= 0 or step >= anneal:
        return cfg.optim.tau_end
    frac = step / anneal
    return cfg.optim.tau_start + frac * (cfg.optim.tau_end - cfg.optim.tau_start)


def _volume_match(pred: torch.Tensor, target: torch.Tensor, hop: int) -> torch.Tensor:
    """Per-frame RMS-match ``pred`` to ``target``. Both ``[B, S]``, S a multiple of hop."""
    B = pred.shape[0]
    T = pred.shape[-1] // hop
    pred_f = pred[..., : T * hop].view(B, T, hop)
    tgt_f = target[..., : T * hop].view(B, T, hop)
    pred_rms = pred_f.pow(2).mean(-1).clamp_min(1e-12).sqrt()
    tgt_rms = tgt_f.pow(2).mean(-1).clamp_min(1e-12).sqrt()
    gain = (tgt_rms / pred_rms).unsqueeze(-1)
    return (pred_f * gain).reshape(B, T * hop)


def _load_eval_clips(
    cfg: TrainConfig, device: torch.device, samples_per_frame: int
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Load ``n_audio_samples`` fixed clips from the tail of the manifest, with f0."""
    files = load_manifest(cfg.data.manifest_path)
    n = min(cfg.log.n_audio_samples, len(files))
    picked = files[-n:]
    # Round chunk_samples down to a multiple of samples_per_frame so eval and
    # training use the same alignment.
    chunk_samples = int(round(cfg.data.sample_rate * cfg.data.chunk_seconds))
    chunk_samples = (chunk_samples // samples_per_frame) * samples_per_frame
    T_ctrl = chunk_samples // samples_per_frame

    if cfg.data.pitch_cache_path is None:
        raise ValueError(
            "data.pitch_cache_path is required (eval needs precomputed f0)"
        )
    pitch, meta = _load_pitch_cache(
        cfg.data.pitch_cache_path, cfg.data.sample_rate, samples_per_frame
    )
    fmin = meta["fmin"]
    fmax = meta["fmax"]
    # Manifest order matches the cache index.
    n_total = len(files)
    eval_indices = list(range(n_total - n, n_total))

    clips: list[torch.Tensor] = []
    f0s: list[torch.Tensor] = []
    names: list[str] = []
    for file_idx, df in zip(eval_indices, picked):
        audio = _load_resampled(df.path, cfg.data.sample_rate)
        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))
        audio = audio[:chunk_samples]

        f0_full, voiced_full = pitch[file_idx]
        f0_chunk = np.zeros(T_ctrl, dtype=np.float32)
        voiced_chunk = np.zeros(T_ctrl, dtype=bool)
        have = min(T_ctrl, len(f0_full))
        if have > 0:
            f0_chunk[:have] = f0_full[:have]
            voiced_chunk[:have] = voiced_full[:have]
        f0_filled = fill_unvoiced(f0_chunk, voiced_chunk, fmin, fmax)

        clips.append(torch.from_numpy(audio))
        f0s.append(torch.from_numpy(f0_filled))
        names.append(df.path.name)

    return torch.stack(clips).to(device), torch.stack(f0s).to(device), names


def _param_traj_figure(
    params: torch.Tensor,
    trainable_names: list[str],
    frame_rate: float,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> go.Figure:
    """plotly line plot of trainable param trajectories for one clip.

    If ``bounds`` is given, each trainable param is rescaled to [0, 1] using
    its (lo, hi) so all trainable params share the same axis.
    """
    params = params.detach().cpu().numpy()  # [T, N_PARAMS]
    T = params.shape[0]
    t = np.arange(T) / frame_rate
    fig = go.Figure()
    for name in trainable_names:
        i = PARAM_NAMES.index(name)
        y = params[:, i]
        if bounds is not None:
            lo, hi = bounds[name]
            y = (y - lo) / (hi - lo)
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=name))
    yaxis_title = "normalized value" if bounds is not None else "value"
    fig.update_layout(xaxis_title="time (s)", yaxis_title=yaxis_title, height=400)
    if bounds is not None:
        fig.update_yaxes(range=[-0.05, 1.05])
    return fig


def _mel_fig_stacked(audios: list[tuple[str, np.ndarray]], sr: int) -> go.Figure:
    import librosa
    from plotly.subplots import make_subplots

    titles = [name for name, _ in audios]
    fig = make_subplots(
        rows=len(audios),
        cols=1,
        subplot_titles=titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.0,
    )
    for row, (_, audio) in enumerate(audios, start=1):
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32), sr=sr, n_mels=80
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        fig.add_trace(
            go.Heatmap(z=log_mel, colorscale="Viridis", showscale=False), row=row, col=1
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=160 * len(audios),
        margin=dict(l=10, r=10, t=24, b=10),
    )
    return fig


@torch.no_grad()
def _evaluate(
    model: PinkTromboneController,
    ddp_module: nn.Module,
    eval_clips: torch.Tensor,
    eval_f0: torch.Tensor,
    eval_names: list[str],
    loss_fn: MFCCLoss,
    cfg: TrainConfig,
    step: int,
    run_expensive: bool,
) -> dict[str, float]:
    """Evaluate on cached eval clips. Returns scalar metrics; logs audio/plots on expensive runs."""
    model_was_training = ddp_module.training
    ddp_module.eval()
    target = eval_clips  # [N, S]
    wav = target.unsqueeze(1)  # [N, 1, S]
    params = model(wav, eval_f0)  # [N, T_ctrl, 11]
    frame_rate = model.config.frame_rate
    samples_per_frame = model.samples_per_frame

    fixed_seed = torch.arange(target.shape[0], dtype=torch.long)
    pred_ola = pink_trombone_ola(
        params,
        seed=fixed_seed,
        ir_length=cfg.synth.ir_length,
        ir_impl=cfg.synth.ir_impl,
        control_rate=frame_rate,
    )
    S = min(pred_ola.shape[-1], target.shape[-1])
    pred_norm = _volume_match(
        pred_ola[..., :S].float(), target[..., :S].float(), samples_per_frame
    )
    S_norm = pred_norm.shape[-1]
    loss = loss_fn(pred_norm, target[..., :S_norm].float()).item()
    metrics: dict[str, float] = {"eval/loss_ola": loss}

    if run_expensive:
        pred_exact = pink_trombone(params, seed=fixed_seed, control_rate=frame_rate)
        S_ex = min(pred_exact.shape[-1], target.shape[-1])
        pred_exact_norm = _volume_match(
            pred_exact[..., :S_ex].float(),
            target[..., :S_ex].float(),
            samples_per_frame,
        )
        S_ex_norm = pred_exact_norm.shape[-1]
        metrics["eval/loss_exact"] = loss_fn(
            pred_exact_norm, target[..., :S_ex_norm].float()
        ).item()

        # Pitch MAE (librosa.pyin on CPU)
        pitch_ola = []
        pitch_exact = []
        miss_ola = []
        miss_exact = []
        for i in range(target.shape[0]):
            tgt_np = target[i].detach().cpu().numpy()
            ola_np = pred_norm[i].detach().cpu().numpy()
            ex_np = pred_exact_norm[i].detach().cpu().numpy()
            m_ola = pitch_mae_cents(
                tgt_np,
                ola_np,
                cfg.data.sample_rate,
                fmin=cfg.log.pitch_fmin,
                fmax=cfg.log.pitch_fmax,
                voiced_prob_threshold=cfg.log.pitch_voiced_prob_threshold,
            )
            m_ex = pitch_mae_cents(
                tgt_np,
                ex_np,
                cfg.data.sample_rate,
                fmin=cfg.log.pitch_fmin,
                fmax=cfg.log.pitch_fmax,
                voiced_prob_threshold=cfg.log.pitch_voiced_prob_threshold,
            )
            if not np.isnan(m_ola.mae_cents):
                pitch_ola.append(m_ola.mae_cents)
                miss_ola.append(m_ola.unvoiced_miss_frac)
            if not np.isnan(m_ex.mae_cents):
                pitch_exact.append(m_ex.mae_cents)
                miss_exact.append(m_ex.unvoiced_miss_frac)

        if pitch_ola:
            metrics["eval/pitch_mae_cents_ola"] = float(np.mean(pitch_ola))
            metrics["eval/unvoiced_miss_frac_ola"] = float(np.mean(miss_ola))
        if pitch_exact:
            metrics["eval/pitch_mae_cents_exact"] = float(np.mean(pitch_exact))
            metrics["eval/unvoiced_miss_frac_exact"] = float(np.mean(miss_exact))

        # Audio + figures to wandb
        sr = cfg.data.sample_rate
        gap = np.zeros(int(sr * 0.1), dtype=np.float32)
        wandb_logs: dict[str, object] = {}
        trainable_names = model.trainable_names_
        bounds = {n: model.config.param_spec[n][:2] for n in trainable_names}
        for i, name in enumerate(eval_names):
            tgt_np = target[i].detach().cpu().numpy()
            ola_np = pred_norm[i].detach().cpu().numpy()
            ex_np = pred_exact_norm[i].detach().cpu().numpy()
            short_name = name[:10]
            tag = f"eval/{i:02d}_{short_name}"
            combined = np.concatenate(
                [tgt_np.astype(np.float32), gap, _norm(ola_np), gap, _norm(ex_np)]
            )
            wandb_logs[f"{tag}/audio"] = wandb.Audio(combined, sample_rate=sr)
            wandb_logs[f"{tag}/params"] = wandb.Plotly(
                _param_traj_figure(
                    params[i], trainable_names, frame_rate, bounds=bounds
                )
            )
            wandb_logs[f"{tag}/mel"] = wandb.Plotly(
                _mel_fig_stacked([("target", tgt_np), ("ola", ola_np)], sr)
            )
        wandb.log(wandb_logs, step=step)

    if model_was_training:
        ddp_module.train()
    return metrics


def _norm(x: np.ndarray) -> np.ndarray:
    p = float(np.abs(x).max())
    return x / p * 0.9 if p > 1e-6 else x


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)

    rank, local_rank, world_size, is_ddp = _ddp_info()
    if is_ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    torch.manual_seed(cfg.run.seed + rank)

    run_dir = _make_run_dir(cfg, rank, is_ddp)

    # Model
    model = PinkTromboneController(cfg.model).to(device)
    frame_rate = cfg.model.frame_rate
    samples_per_frame = cfg.model.samples_per_frame
    loss_fn = MFCCLoss(samples_per_frame=samples_per_frame).to(device)

    ddp_module: nn.Module
    if is_ddp:
        ddp_module = DDP(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None
        )
        module: PinkTromboneController = ddp_module.module  # type: ignore[assignment]
    else:
        ddp_module = model
        module = model

    optimizer = torch.optim.AdamW(
        ddp_module.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )

    # Data
    loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        sample_rate=cfg.data.sample_rate,
        chunk_seconds=cfg.data.chunk_seconds,
        rank=rank,
        world_size=world_size,
        epoch=0,
        seed=cfg.run.seed,
        pitch_cache_path=cfg.data.pitch_cache_path,
        samples_per_frame=samples_per_frame,
    )

    # Eval clips (rank 0 only)
    eval_clips: torch.Tensor | None = None
    eval_f0: torch.Tensor | None = None
    eval_names: list[str] = []
    if rank == 0:
        eval_clips, eval_f0, eval_names = _load_eval_clips(
            cfg, device, samples_per_frame
        )

    # wandb (rank 0 only)
    if rank == 0:
        wandb.init(
            project=cfg.log.wandb_project,
            entity=cfg.log.wandb_entity,
            name=run_dir.name,
            dir=str(run_dir),
            config=json.loads(cfg.model_dump_json()),
            mode=cfg.log.wandb_mode,
        )

    step = 0
    pbar = tqdm(total=cfg.optim.max_steps, disable=(rank != 0), desc="training")

    data_iter = iter(loader)
    epoch = 0

    throughput_t0 = time.perf_counter()
    throughput_audio_s = 0.0

    if rank == 0 and eval_clips is not None and eval_f0 is not None:
        metrics = _evaluate(
            module,
            ddp_module,
            eval_clips,
            eval_f0,
            eval_names,
            loss_fn,
            cfg,
            step,
            run_expensive=True,
        )
        wandb.log(metrics, step=step)
        tqdm.write(f"[eval] step={step} loss_ola={metrics['eval/loss_ola']:.4f}")

    while step < cfg.optim.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
            data_iter = iter(loader)
            batch = next(data_iter)

        wav = batch["audio"].to(device, non_blocking=True)  # [B, S]
        f0 = batch["pitch"].to(device, non_blocking=True)  # [B, T_ctrl]
        target = wav
        wav_in = wav.unsqueeze(1)  # [B, 1, S]

        # LR warmup
        for g in optimizer.param_groups:
            g["lr"] = cfg.optim.lr * _warmup_lr(step, cfg.optim.warmup_steps)

        tau = _tau_for_step(step, cfg)
        optimizer.zero_grad(set_to_none=True)

        # Encoder + head in bf16; synth & loss in fp32
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            params = ddp_module(wav_in, f0, tau)
        params = params.float()

        pred = pink_trombone_ola(
            params,
            ir_length=cfg.synth.ir_length,
            ir_impl=cfg.synth.ir_impl,
            control_rate=frame_rate,
        )
        S = min(pred.shape[-1], target.shape[-1])
        pred_norm = _volume_match(pred[..., :S], target[..., :S], samples_per_frame)
        S_norm = pred_norm.shape[-1]
        loss = loss_fn(pred_norm, target[..., :S_norm])

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp_module.parameters(), cfg.optim.grad_clip
        )
        optimizer.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        throughput_audio_s += wav.shape[0] * wav.shape[-1] / cfg.data.sample_rate

        if rank == 0 and step % cfg.log.log_every == 0:
            elapsed = time.perf_counter() - throughput_t0
            throughput_per_gpu = throughput_audio_s / elapsed if elapsed > 0 else 0.0
            throughput_total = throughput_per_gpu * world_size
            throughput_t0 = time.perf_counter()
            throughput_audio_s = 0.0
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/tau": tau,
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": epoch,
                    "System/throughput_per_gpu": throughput_per_gpu,
                    "System/throughput_total": throughput_total,
                },
                step=step,
            )

        if (
            rank == 0
            and eval_clips is not None
            and eval_f0 is not None
            and step % cfg.log.val_every == 0
        ):
            metrics = _evaluate(
                module,
                ddp_module,
                eval_clips,
                eval_f0,
                eval_names,
                loss_fn,
                cfg,
                step,
                run_expensive=(step % cfg.log.eval_every == 0),
            )
            wandb.log(metrics, step=step)
            tqdm.write(f"[eval] step={step} loss_ola={metrics['eval/loss_ola']:.4f}")

        if rank == 0 and step % cfg.log.ckpt_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config_json": cfg.model_dump_json(),
                },
                ckpt_path,
            )
            last_path = run_dir / "checkpoints" / "last.pt"
            if last_path.is_symlink() or last_path.exists():
                last_path.unlink()
            last_path.symlink_to(ckpt_path.name)

    pbar.close()
    if rank == 0:
        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
