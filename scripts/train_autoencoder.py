"""Neural autoencoder sanity check: replace Pink Trombone decoder with a SEANet
decoder, train end-to-end with the same MFCC loss + volume normalization the
controller uses. Reconstruction should saturate quickly — the bottleneck is
gone, so any remaining loss is on the loss/encoder side.

Usage:
    uv run python scripts/train_autoencoder.py run.name=ae_smoke
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import hydra
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from plotly.subplots import make_subplots
from tqdm import tqdm

import wandb
from samuel.config import TrainConfig
from samuel.data import (
    DatasetFile,
    _load_resampled,
    build_dataloader,
    load_manifest,
    split_train_val,
)
from samuel.encoder import SEANetDecoder, SEANetEncoder
from samuel.losses import MFCCLoss


class WaveformAutoencoder(torch.nn.Module):
    """SEANet encoder + symmetric SEANet decoder. ``[B, 1, S] -> [B, 1, S]``."""

    def __init__(self, encoder_config):
        super().__init__()
        self.encoder = SEANetEncoder(encoder_config)
        self.decoder = SEANetDecoder(encoder_config)
        self.hop_length = self.encoder.hop_length

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        S = wav.shape[-1]
        hop = self.hop_length
        pad = (hop - S % hop) % hop
        if pad > 0:
            wav = F.pad(wav, (0, pad))
        z = self.encoder(wav)
        out = self.decoder(z)
        # Trim/pad decoder output to original input length S.
        if out.shape[-1] >= S:
            out = out[..., :S]
        else:
            out = F.pad(out, (0, S - out.shape[-1]))
        return out


def _volume_match(pred: torch.Tensor, target: torch.Tensor, hop: int) -> torch.Tensor:
    """Per-frame RMS-match pred to target. Both ``[B, S]``."""
    B = pred.shape[0]
    T = pred.shape[-1] // hop
    pred_f = pred[..., : T * hop].view(B, T, hop)
    tgt_f = target[..., : T * hop].view(B, T, hop)
    pred_rms = pred_f.pow(2).mean(-1).clamp_min(1e-12).sqrt()
    tgt_rms = tgt_f.pow(2).mean(-1).clamp_min(1e-12).sqrt()
    gain = (tgt_rms / pred_rms).unsqueeze(-1)
    return (pred_f * gain).reshape(B, T * hop)


def _norm(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    p = float(np.abs(x).max())
    return x / p * 0.9 if p > 1e-6 else x


def _mel_fig_stacked(audios: list[tuple[str, np.ndarray]], sr: int) -> go.Figure:
    import librosa

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
            go.Heatmap(z=log_mel, colorscale="Viridis", showscale=False),
            row=row,
            col=1,
        )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(height=160 * len(audios), margin=dict(l=10, r=10, t=24, b=10))
    return fig


def _sample_eval_clips(
    files: list[DatasetFile],
    step: int,
    n: int,
    sample_rate: int,
    chunk_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    n = min(n, len(files))
    if n == 0:
        return torch.empty(0, chunk_samples, device=device), []
    indices = np.random.RandomState(step).choice(len(files), size=n, replace=False)
    clips, names = [], []
    for local_idx in indices:
        df = files[int(local_idx)]
        audio = _load_resampled(df.path, sample_rate)
        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))
        audio = audio[:chunk_samples]
        clips.append(torch.from_numpy(audio))
        names.append(df.path.name)
    return torch.stack(clips).to(device), names


@torch.no_grad()
def _evaluate_split(
    model: WaveformAutoencoder,
    files: list[DatasetFile],
    split: str,
    loss_fn: MFCCLoss,
    cfg: TrainConfig,
    step: int,
    samples_per_frame: int,
    log_media: bool,
    device: torch.device,
) -> dict[str, object]:
    chunk_samples = int(round(cfg.data.sample_rate * cfg.data.chunk_seconds))
    chunk_samples = (chunk_samples // samples_per_frame) * samples_per_frame
    clips, names = _sample_eval_clips(
        files,
        step,
        cfg.log.n_audio_samples,
        cfg.data.sample_rate,
        chunk_samples,
        device,
    )
    if clips.numel() == 0:
        return {}
    wav_in = clips.unsqueeze(1)
    pred = model(wav_in).squeeze(1)
    S = min(pred.shape[-1], clips.shape[-1])
    pred_norm = _volume_match(
        pred[..., :S].float(), clips[..., :S].float(), samples_per_frame
    )
    S_norm = pred_norm.shape[-1]
    loss = loss_fn(pred_norm, clips[..., :S_norm].float())
    out: dict[str, object] = {f"eval/{split}/loss_mfcc": loss.item()}

    if not log_media:
        return out

    sr = cfg.data.sample_rate
    gap = np.zeros(int(sr * 0.1), dtype=np.float32)
    audios: list[wandb.Audio] = []
    mel_pairs: list[tuple[str, np.ndarray]] = []
    for i, name in enumerate(names):
        tgt_np = clips[i].detach().cpu().numpy()
        pred_np = pred_norm[i].detach().cpu().numpy()
        combined = np.concatenate([_norm(pred_np), gap, tgt_np.astype(np.float32)])
        audios.append(wandb.Audio(combined, sample_rate=sr, caption=name))
        mel_pairs.append((f"{name}: target", tgt_np))
        mel_pairs.append((f"{name}: pred", pred_np))
    out[f"eval/{split}/audio"] = audios
    out[f"eval/{split}/mel"] = wandb.Plotly(_mel_fig_stacked(mel_pairs, sr))
    return out


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.run.seed)

    # Run dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = cfg.run.runs_root / f"{cfg.run.name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        f.write(cfg.model_dump_json(indent=2))

    samples_per_frame = cfg.model.samples_per_frame
    model = WaveformAutoencoder(cfg.model.encoder).to(device)
    loss_fn = MFCCLoss(
        samples_per_frame=samples_per_frame,
        n_fft=cfg.loss.mfcc_n_fft or samples_per_frame,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
    )

    # Data — pitch cache is not used by the autoencoder but the loader still
    # works with pitch enabled; just ignore the pitch field in the batch.
    loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        sample_rate=cfg.data.sample_rate,
        chunk_seconds=cfg.data.chunk_seconds,
        rank=0,
        world_size=1,
        epoch=0,
        seed=cfg.run.seed,
        pitch_cache_path=cfg.data.pitch_cache_path,
        samples_per_frame=samples_per_frame,
        val_fraction=cfg.data.val_fraction,
    )

    # Eval files
    files = load_manifest(cfg.data.manifest_path)
    train_files, val_files = split_train_val(files, cfg.data.val_fraction)

    wandb.init(
        project=cfg.log.wandb_project,
        entity=cfg.log.wandb_entity,
        name=run_dir.name,
        dir=str(run_dir),
        config=json.loads(cfg.model_dump_json()),
        mode=cfg.log.wandb_mode,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[autoencoder] params: {n_params / 1e6:.2f}M, hop={model.hop_length}")

    step = 0
    pbar = tqdm(total=cfg.optim.max_steps, desc="training")
    data_iter = iter(loader)
    epoch = 0
    throughput_t0 = time.perf_counter()
    throughput_audio_s = 0.0

    while step < cfg.optim.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
            data_iter = iter(loader)
            batch = next(data_iter)

        wav = batch["audio"].to(device, non_blocking=True)  # [B, S]
        target = wav
        wav_in = wav.unsqueeze(1)

        # LR warmup
        warm = (
            (step + 1) / max(cfg.optim.warmup_steps, 1)
            if step < cfg.optim.warmup_steps
            else 1.0
        )
        for g in optimizer.param_groups:
            g["lr"] = cfg.optim.lr * warm

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            pred = model(wav_in).squeeze(1)
        pred = pred.float()
        S = min(pred.shape[-1], target.shape[-1])
        pred_norm = _volume_match(pred[..., :S], target[..., :S], samples_per_frame)
        S_norm = pred_norm.shape[-1]
        loss = loss_fn(pred_norm, target[..., :S_norm])

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.optim.grad_clip
        )
        optimizer.step()

        step += 1
        pbar.update(1)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        throughput_audio_s += wav.shape[0] * wav.shape[-1] / cfg.data.sample_rate

        if step % cfg.log.log_every == 0:
            elapsed = time.perf_counter() - throughput_t0
            throughput = throughput_audio_s / elapsed if elapsed > 0 else 0.0
            throughput_t0 = time.perf_counter()
            throughput_audio_s = 0.0
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm),
                    "train/epoch": epoch,
                    "System/throughput_per_gpu": throughput,
                },
                step=step,
            )

        if step % cfg.log.val_every == 0:
            model.eval()
            metrics: dict[str, object] = {}
            log_media = step % cfg.log.eval_every == 0
            metrics.update(
                _evaluate_split(
                    model,
                    train_files,
                    "train",
                    loss_fn,
                    cfg,
                    step,
                    samples_per_frame,
                    log_media,
                    device,
                )
            )
            metrics.update(
                _evaluate_split(
                    model,
                    val_files,
                    "val",
                    loss_fn,
                    cfg,
                    step,
                    samples_per_frame,
                    log_media,
                    device,
                )
            )
            wandb.log(metrics, step=step)
            tqdm.write(
                f"[eval] step={step} "
                f"train={metrics.get('eval/train/loss_mfcc', float('nan')):.4f} "
                f"val={metrics.get('eval/val/loss_mfcc', float('nan')):.4f}"
            )
            model.train()

        if step % cfg.log.ckpt_every == 0:
            ckpt_path = run_dir / "checkpoints" / f"{step:07d}.pt"
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config_json": cfg.model_dump_json(),
                },
                ckpt_path,
            )

    pbar.close()
    wandb.finish()


if __name__ == "__main__":
    main()
