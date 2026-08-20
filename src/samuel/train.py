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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import plotly.graph_objects as go
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
    split_train_val,
)
from samuel.evals.asr import WhisperEvaluator
from samuel.evals.pitch import pitch_mae_cents
from samuel.losses import MelSpecLoss, MFCCLoss, MultiScaleLogMagSTFTLoss
from samuel.model import PinkTromboneController
from samuel.pink_trombone import PARAM_NAMES, pink_trombone_ola
from samuel.ssl_loss import SSLFeatureLoss


class CombinedReconLoss(nn.Module):
    """Weighted sum of reconstruction losses.

    All components are always evaluated so their values can be logged for
    comparison even when they contribute zero to the gradient. The training
    loss is the weight-weighted sum; zero-weighted components are computed
    but don't backprop. Passing all weights = 0 raises.
    """

    def __init__(self, components: list[tuple[str, float, nn.Module]]) -> None:
        super().__init__()
        if not any(w != 0.0 for _, w, _ in components):
            raise ValueError("CombinedReconLoss needs at least one nonzero weight")
        self.names: list[str] = [n for n, _, _ in components]
        self.weights: list[float] = [w for _, w, _ in components]
        self.fns = nn.ModuleList([fn for _, _, fn in components])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total, _ = self.with_components(pred, target)
        return total

    def with_components(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components: dict[str, torch.Tensor] = {}
        total = pred.new_zeros(())
        for n, w, fn in zip(self.names, self.weights, self.fns):
            v = fn(pred, target)
            components[n] = v
            if w != 0.0:
                total = total + w * v
        return total, components


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
        run_dir = cfg.run.runs_root / f"{cfg.run.name}_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        with open(run_dir / "config.json", "w") as f:
            f.write(cfg.model_dump_json(indent=2))
        path_str = str(run_dir)
    else:
        path_str = ""
    path_str = _broadcast_str(path_str, src=0, is_ddp=is_ddp)
    return Path(path_str)


def _save_checkpoint(
    run_dir: Path,
    step: int,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Path:
    """Write ``<step>.pt`` and point the ``last.pt`` symlink at it."""
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
    return ckpt_path


def _delete_ckpt_artifact(artifact: wandb.Artifact) -> None:
    try:
        # A version can only be deleted once its own upload is committed.
        artifact.wait()
        artifact.delete(delete_aliases=True)
    except Exception as exc:  # noqa: BLE001 - never fail training on this
        tqdm.write(f"[ckpt] could not delete artifact {artifact.name}: {exc}")


def _log_ckpt_artifact(
    run_dir: Path,
    ckpt_path: Path,
    step: int,
    logged: list[wandb.Artifact],
    prune: bool = True,
) -> None:
    """Upload ``ckpt_path`` as the run's model artifact, superseding earlier ones.

    Every upload creates a new artifact version (wandb has no in-place
    overwrite), so older versions are deleted as they are replaced and consumers
    just take the ``latest`` alias. One older version is kept alive on purpose:
    an in-flight upload dedups against its predecessor and fails with "rebase
    required" if that base disappears. ``logged`` is the list of versions this
    process still owns, newest last, and is updated in place. ``prune`` is off
    for offline runs, where there is no server to delete from.
    """
    artifact = wandb.Artifact(
        f"{run_dir.name}-checkpoint", type="model", metadata={"step": step}
    )
    # Store under a stable name so the artifact always exposes the newest
    # checkpoint as last.pt (ckpt_path may be a symlink target like 0016000.pt).
    artifact.add_file(str(ckpt_path), name="last.pt")
    logged.append(wandb.log_artifact(artifact))

    while prune and len(logged) > 2:
        _delete_ckpt_artifact(logged.pop(0))


def _prune_ckpt_artifacts(logged: list[wandb.Artifact]) -> None:
    """Delete every logged checkpoint version but the newest.

    Only safe at the end of a run: it blocks until the newest version has
    finished uploading, so the versions it deletes can no longer be an upload's
    dedup base.
    """
    if not logged:
        return
    newest = logged[-1]
    try:
        newest.wait()
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"[ckpt] final artifact upload did not confirm: {exc}")
        return
    for old in logged[:-1]:
        _delete_ckpt_artifact(old)
    logged[:] = [newest]


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


def _normalized_trainable_params(
    params: torch.Tensor, module: PinkTromboneController
) -> torch.Tensor:
    """Trainable params rescaled to ``[0, 1]``; ``[B, T, n_trainable]``.

    Uses the bucket-center range per parameter so no single wide-range
    parameter dominates. Differentiable.
    """
    train_params = params.index_select(-1, module._trainable_idx).float()
    lo = module.bucket_centers[:, 0].view(1, 1, -1)
    hi = module.bucket_centers[:, -1].view(1, 1, -1)
    return (train_params - lo) / (hi - lo).clamp_min(1e-8)


def _normalized_trainable_diffs(
    params: torch.Tensor, module: PinkTromboneController
) -> torch.Tensor:
    """Per-frame absolute change of trainable params, each rescaled to [0, 1].

    Returns ``[B, T-1, n_trainable]``; differentiable.
    """
    p_norm = _normalized_trainable_params(params, module)
    return (p_norm[:, 1:] - p_norm[:, :-1]).abs()


def _normalized_trainable_accels(
    params: torch.Tensor, module: PinkTromboneController
) -> torch.Tensor:
    """Absolute *second* difference of the trajectories; ``[B, T-2, n_t]``.

    Unlike the first difference this is blind to speed: a steady ramp scores
    zero however fast it moves, while a direction reversal scores twice the
    step size. That is the distinction the first-difference penalty cannot
    make -- turning ``loss.smooth`` up far enough to remove jitter also
    freezes the genuine gestures.
    """
    p_norm = _normalized_trainable_params(params, module)
    d1 = p_norm[:, 1:] - p_norm[:, :-1]
    return (d1[:, 1:] - d1[:, :-1]).abs()


def _weighted_param_penalty(
    per_frame: torch.Tensor,
    module: PinkTromboneController,
    weights: dict[str, float],
) -> torch.Tensor:
    """``sum_p weights[p] * mean_{batch,time} per_frame[..., p]``.

    Params absent from ``weights`` contribute 0.
    """
    coeffs = torch.tensor(
        [weights.get(n, 0.0) for n in module.trainable_names_],
        dtype=per_frame.dtype,
        device=per_frame.device,
    )
    return (per_frame.mean(dim=(0, 1)) * coeffs).sum()


def _smoothness_loss(
    params: torch.Tensor,
    module: PinkTromboneController,
    weights: dict[str, float],
) -> torch.Tensor:
    """L1 penalty on the per-frame change: ``sum_p w_p * mean |Δp_norm|``.

    See LossConfig.smooth_weights.
    """
    return _weighted_param_penalty(
        _normalized_trainable_diffs(params, module), module, weights
    )


def _acceleration_loss(
    params: torch.Tensor,
    module: PinkTromboneController,
    weights: dict[str, float],
) -> torch.Tensor:
    """L1 penalty on the per-frame *acceleration*: ``sum_p w_p * mean |Δ²p_norm|``.

    See LossConfig.accel_weights.
    """
    return _weighted_param_penalty(
        _normalized_trainable_accels(params, module), module, weights
    )


def _rest_targets_norm(
    module: PinkTromboneController, targets: dict[str, float]
) -> torch.Tensor:
    """Rest targets, given in raw units, rescaled to the trajectories' [0, 1]."""
    lo = module.bucket_centers[:, 0]
    hi = module.bucket_centers[:, -1]
    tgt = torch.tensor(
        [targets.get(n, 0.0) for n in module.trainable_names_],
        dtype=lo.dtype,
        device=lo.device,
    )
    return (tgt - lo) / (hi - lo).clamp_min(1e-8)


def _rest_pose_distance(
    params: torch.Tensor,
    module: PinkTromboneController,
    targets: dict[str, float],
) -> torch.Tensor:
    """Per-param mean ``|p_norm - target_norm|``; ``[n_trainable]``, unweighted."""
    tgt_norm = _rest_targets_norm(module, targets)
    mask = torch.tensor(
        [1.0 if n in targets else 0.0 for n in module.trainable_names_],
        dtype=tgt_norm.dtype,
        device=tgt_norm.device,
    )
    p_norm = _normalized_trainable_params(params, module)
    return (p_norm - tgt_norm.view(1, 1, -1)).abs().mean(dim=(0, 1)) * mask


def _rest_pose_loss(
    params: torch.Tensor,
    module: PinkTromboneController,
    targets: dict[str, float],
    weights: dict[str, float],
) -> torch.Tensor:
    """``sum_p w_p * mean |p_norm - target_norm|``; see LossConfig.rest."""
    dist = _rest_pose_distance(params, module, targets)
    coeffs = torch.tensor(
        [weights.get(n, 1.0) for n in module.trainable_names_],
        dtype=dist.dtype,
        device=dist.device,
    )
    return (dist * coeffs).sum()


@torch.no_grad()
def _silent_frame_rest_metrics(
    params: torch.Tensor,
    module: PinkTromboneController,
    target: torch.Tensor,
    cfg: TrainConfig,
    silence_db: float = -20.0,
) -> dict[str, float]:
    """Rest-pose distance on frames quieter than ``silence_db`` re. target_rms."""
    spf = module.samples_per_frame
    T = params.shape[1]
    S = T * spf
    tgt = F.pad(target, (0, max(0, S - target.shape[-1])))[:, :S]
    frame_rms = tgt.view(tgt.shape[0], T, spf).pow(2).mean(-1).sqrt()  # [N, T]
    silent = frame_rms < cfg.data.target_rms * 10 ** (silence_db / 20.0)
    out = {"eval/silent_frac": float(silent.float().mean())}
    if not bool(silent.any()):
        return out

    names = module.trainable_names_
    targets = cfg.loss.rest_targets
    tgt_norm = _rest_targets_norm(module, targets)
    p_norm = _normalized_trainable_params(params, module)
    dist = (p_norm - tgt_norm.view(1, 1, -1)).abs()[silent]  # [n_silent, n_t]
    per_param = dist.mean(dim=0)
    total = 0.0
    for name, v in zip(names, per_param.tolist()):
        if name in targets:
            out[f"eval/rest_dist_silent/{name}"] = v
            total += v
    out["eval/rest_loss_silent"] = total
    return out


def _f0_losses(
    aux: dict[str, torch.Tensor],
    module: PinkTromboneController,
    f0_label: torch.Tensor,
    voiced: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """(f0 cross-entropy, voiced BCE, readouts) for the causal pitch head.

    The f0 term is taken on voiced frames only: elsewhere the pyin label is
    ``fill_unvoiced``'s interpolation, not a measurement. Returns a zero f0
    term for a batch with no voiced frame at all.
    """
    targets = module.f0_bucket_targets(f0_label)  # [B, T]
    per_frame = F.cross_entropy(
        aux["f0_logits"].flatten(0, 1), targets.flatten(), reduction="none"
    ).view_as(targets)
    n_voiced = voiced.sum()
    f0_loss = (per_frame * voiced).sum() / n_voiced.clamp_min(1.0)
    voiced_loss = F.binary_cross_entropy_with_logits(aux["voiced_logits"], voiced)

    with torch.no_grad():
        cents = (aux["f0_hz"] / f0_label.clamp_min(1e-3)).log2() * 1200.0
        mae = (cents.abs() * voiced).sum() / n_voiced.clamp_min(1.0)
        acc = ((aux["voiced_logits"] > 0).float() == voiced).float().mean()
    readouts = {
        "f0_mae_cents": float(mae) if n_voiced > 0 else float("nan"),
        "voiced_acc": float(acc),
        "voiced_frac": float(voiced.mean()),
    }
    return f0_loss, voiced_loss, readouts


@torch.no_grad()
def _reversal_rate(params: torch.Tensor, module: PinkTromboneController) -> float:
    """Fraction of moving frames whose direction flips.

    The jitter metric ``param_variation`` cannot supply: mean ``|Δ|`` scores a
    glide and a zigzag of the same step size identically, so it registers how
    *far* the trajectory travels but not how much it oscillates. A monotone
    glide scores 0 here, a frame-by-frame zigzag scores 1.
    """
    p_norm = _normalized_trainable_params(params, module)
    sign = torch.sign(p_norm[:, 1:] - p_norm[:, :-1])
    moving = (sign[:, 1:] != 0) & (sign[:, :-1] != 0)
    n_moving = moving.sum()
    if n_moving == 0:
        return 0.0
    flips = (sign[:, 1:] != sign[:, :-1]) & moving
    return float(flips.sum() / n_moving)


@torch.no_grad()
def _param_variation(params: torch.Tensor, module: PinkTromboneController) -> float:
    """Mean per-frame absolute change of trainable params, normalised to [0, 1].

    Each trainable parameter is rescaled to ``[0, 1]`` using its bucket-center
    range before differencing, so no single wide-range parameter dominates.
    Returned scalar is the mean over batch, time, and trainable params. Useful
    as an "are the predicted trajectories changing at a humanly-plausible
    rate?" check: a value of 0.1 means the average param moves 10 % of its
    range per control frame.
    """
    return float(_normalized_trainable_diffs(params, module).mean().item())


@torch.no_grad()
def _controller_diagnostics(
    aux: dict[str, torch.Tensor], trainable_names: list[str]
) -> dict[str, float | wandb.Histogram]:
    """Per-step diagnostics for the categorical head + encoder.

    Logits and z come from ``model.forward(..., return_aux=True)``. Logged
    under ``diag/`` to keep the train/eval namespaces clean.

    One bucket histogram per trainable param, ``bucket_usage``: hard argmax
    counts (what eval and hard-Gumbel pick).
    """
    logits = aux["logits"].detach().float()  # [B, T, n_t, n_b]
    z = aux["z"].detach().float()  # [B, dim, T]
    _, _, _, n_b = logits.shape

    top2 = logits.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]  # [B, T, n_t]
    argmax = logits.argmax(-1)  # [B, T, n_t]
    edges = np.arange(n_b + 1) - 0.5

    out: dict[str, float | wandb.Histogram] = {}
    for j, name in enumerate(trainable_names):
        a_j = argmax[..., j].flatten()
        counts = torch.bincount(a_j, minlength=n_b).float()
        out[f"diag/margin/{name}"] = margin[..., j].mean().item()
        out[f"diag/bucket_usage/{name}"] = wandb.Histogram(
            np_histogram=(counts.cpu().numpy(), edges)
        )

    out["diag/margin/mean"] = margin.mean().item()
    out["diag/z_mean_norm"] = z.norm(dim=1).mean().item()
    # Per-feature std across batch+time, averaged. Low ⇒ encoder collapsed.
    out["diag/z_std_per_feat"] = z.std(dim=(0, 2)).mean().item()
    return out


def _rms_normalize(wav: torch.Tensor, target_rms: float) -> torch.Tensor:
    """Scale each clip to ``target_rms`` (one gain per clip). ``wav`` is ``[B, S]``.

    Applied to the audio used as *both* encoder input and loss target, so the
    dataset has a single canonical loudness. The synth output is deliberately
    left alone: the model must produce this absolute level itself, so going
    silent (e.g. driving ``intensity`` to 0) is penalised rather than corrected
    for free. Within-utterance energy dynamics are untouched.

    This replaced a per-frame RMS match of the synth output to the target. That
    match copied the target's energy contour onto the prediction, so the loss
    carried no loudness gradient at all, and with a trainable ``intensity`` it
    opened a constant-output collapse basin (entropy and param_variation to
    zero, WER above 1).
    """
    rms = wav.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return wav * (target_rms / rms)


@dataclass
class EvalSetup:
    """State needed for evaluation.

    Holds a fixed set of held-out val clips preloaded into CPU tensors so
    every eval step sees the same data — only the audio-media subset varies
    by step.
    """

    val_wavs: torch.Tensor  # [N_eval, S] float32, CPU
    val_f0: torch.Tensor  # [N_eval, T_ctrl] float32, CPU
    val_voiced: torch.Tensor  # [N_eval, T_ctrl] float32, CPU
    val_names: list[str]  # caption per clip
    # (index_in_manifest, chunk_start_sample) per clip — stable key for the
    # Whisper target cache. The chunk_start component keeps keys distinct if
    # eval ever draws more than one chunk from the same file.
    val_target_keys: list[tuple[int, int]]
    chunk_samples: int
    T_ctrl: int
    whisper: "WhisperEvaluator | None"


def _eval_setup(
    cfg: TrainConfig,
    samples_per_frame: int,
    device: torch.device,
) -> EvalSetup:
    """Load manifest + pitch cache + ASR model + the fixed eval-clip tensors.

    Called once at rank-0 startup. Loading 100 4-second clips at 44.1 kHz is
    about 17 MB — easily kept in memory for the run.
    """
    files = load_manifest(cfg.data.manifest_path)
    _, val_files = split_train_val(files, cfg.data.val_fraction)
    eval_chunk_seconds = cfg.log.eval_chunk_seconds or cfg.data.chunk_seconds
    chunk_samples = int(round(cfg.data.sample_rate * eval_chunk_seconds))
    # Round chunk_samples down to a multiple of samples_per_frame so eval and
    # training use the same alignment.
    chunk_samples = (chunk_samples // samples_per_frame) * samples_per_frame
    T_ctrl = chunk_samples // samples_per_frame

    if cfg.data.pitch_cache_path is None:
        raise ValueError(
            "data.pitch_cache_path is required (eval needs precomputed f0)"
        )
    pitch = _load_pitch_cache(
        cfg.data.pitch_cache_path, cfg.data.sample_rate, samples_per_frame
    )
    f0_cfg = cfg.model.f0
    if f0_cfg.enabled and (f0_cfg.fmin > pitch.fmin or f0_cfg.fmax < pitch.fmax):
        raise ValueError(
            f"model.f0 buckets span [{f0_cfg.fmin}, {f0_cfg.fmax}] Hz but the "
            f"pitch cache holds [{pitch.fmin}, {pitch.fmax}] Hz — labels "
            f"outside the head's range would all collapse onto its end buckets"
        )

    n_eval = min(cfg.log.n_eval_clips, len(val_files))
    selected = val_files[:n_eval]

    wavs: list[torch.Tensor] = []
    f0s: list[torch.Tensor] = []
    voiceds: list[torch.Tensor] = []
    names: list[str] = []
    target_keys: list[tuple[int, int]] = []
    # Eval always takes the first chunk of each file; if that ever changes,
    # thread the real start offset through here so target keys stay distinct.
    chunk_start = 0
    for df in selected:
        audio = _load_resampled(df.path, cfg.data.sample_rate)
        if len(audio) < chunk_samples:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))
        audio = audio[chunk_start : chunk_start + chunk_samples]

        f0_full, voiced_full = pitch.by_file[df.index_in_manifest]
        f0_chunk = np.zeros(T_ctrl, dtype=np.float32)
        voiced_chunk = np.zeros(T_ctrl, dtype=bool)
        have = min(T_ctrl, len(f0_full))
        if have > 0:
            f0_chunk[:have] = f0_full[:have]
            voiced_chunk[:have] = voiced_full[:have]
        f0_filled = fill_unvoiced(f0_chunk, voiced_chunk, pitch.fmin, pitch.fmax)

        wavs.append(torch.from_numpy(audio))
        f0s.append(torch.from_numpy(f0_filled))
        voiceds.append(torch.from_numpy(voiced_chunk.astype(np.float32)))
        names.append(df.path.name)
        target_keys.append((df.index_in_manifest, chunk_start))

    whisper: WhisperEvaluator | None = None
    if cfg.log.asr_whisper_size:
        whisper = WhisperEvaluator(
            model_size=cfg.log.asr_whisper_size,
            device="cuda" if device.type == "cuda" else "cpu",
        )

    return EvalSetup(
        val_wavs=torch.stack(wavs),
        val_f0=torch.stack(f0s),
        val_voiced=torch.stack(voiceds),
        val_names=names,
        val_target_keys=target_keys,
        chunk_samples=chunk_samples,
        T_ctrl=T_ctrl,
        whisper=whisper,
    )


def _audio_sample_indices(step: int, total: int, k: int) -> list[int]:
    """Pick ``k`` distinct indices in ``[0, total)`` deterministically by step.

    Same step ⇒ same indices (cross-run reproducibility); consecutive evals
    within a run get different subsets so wandb gets fresh listening samples.
    """
    k = min(k, total)
    if k == 0:
        return []
    return [
        int(i) for i in np.random.RandomState(step).choice(total, size=k, replace=False)
    ]


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


_EVAL_BATCH = 16  # cap for forward/synth chunks to keep memory bounded


def _run_eval_batched(
    model: PinkTromboneController,
    wavs: torch.Tensor,  # [N, S] on device
    f0s: torch.Tensor,  # [N, T_ctrl] on device
    cfg: TrainConfig,
    frame_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Forward + OLA synth in chunks. Returns (params, ola, aux)."""
    N = wavs.shape[0]
    params_all: list[torch.Tensor] = []
    ola_all: list[torch.Tensor] = []
    aux_all: list[dict[str, torch.Tensor]] = []
    for start in range(0, N, _EVAL_BATCH):
        end = min(start + _EVAL_BATCH, N)
        wav = wavs[start:end].unsqueeze(1)  # [B, 1, S]
        f0 = f0s[start:end]
        params, aux = model(wav, f0, return_aux=True)
        fixed_seed = torch.arange(start, end, dtype=torch.long)
        ola = pink_trombone_ola(
            params,
            seed=fixed_seed,
            ir_length=cfg.synth.ir_length,
            control_rate=frame_rate,
        )
        params_all.append(params)
        ola_all.append(ola)
        aux_all.append(aux)
    # ``z`` is the only aux entry whose frame count can differ per chunk; it is
    # unused here, so concatenating the rest keeps this cheap.
    aux_cat = {
        k: torch.cat([a[k] for a in aux_all], dim=0) for k in aux_all[0] if k != "z"
    }
    return torch.cat(params_all, dim=0), torch.cat(ola_all, dim=0), aux_cat


@torch.no_grad()
def _evaluate(
    model: PinkTromboneController,
    ddp_module: nn.Module,
    eval_setup: EvalSetup,
    loss_fn: CombinedReconLoss,
    cfg: TrainConfig,
    step: int,
    device: torch.device,
) -> dict[str, object]:
    """Full eval on the fixed held-out val clips.

    Computes losses, pitch MAE, ASR WER/CER, and ``param_variation`` on every
    val clip. Logs audio / params / mel media for ``n_audio_samples`` of them
    (random subset, deterministic by ``step``). Wall-time of the eval call is
    logged under ``eval/duration_s`` so we can keep an eye on it growing.
    """
    if eval_setup.val_wavs.numel() == 0:
        return {}

    t0 = time.perf_counter()
    model_was_training = ddp_module.training
    ddp_module.eval()

    target = eval_setup.val_wavs.to(device)  # [N, S] float32
    target = _rms_normalize(target, cfg.data.target_rms)  # same level as training
    f0 = eval_setup.val_f0.to(device)  # [N, T_ctrl]
    frame_rate = model.config.frame_rate

    params, pred_ola, eval_aux = _run_eval_batched(model, target, f0, cfg, frame_rate)

    # Synth output used as-is: the model is scored on hitting target_rms.
    S = min(pred_ola.shape[-1], target.shape[-1])
    pred_norm = pred_ola[..., :S].float()

    S_norm = pred_norm.shape[-1]
    total, components = loss_fn.with_components(pred_norm, target[..., :S_norm].float())

    out: dict[str, object] = {
        "eval/loss": total.item(),
        "eval/param_variation": _param_variation(params, model),
        "eval/smooth_loss": _smoothness_loss(
            params, model, cfg.loss.smooth_weights
        ).item(),
        "eval/accel_loss": _acceleration_loss(
            params, model, cfg.loss.accel_weights
        ).item(),
        # The two jitter metrics that mean |Δ| cannot express: acceleration is
        # blind to steady movement, reversal rate counts oscillation directly.
        "eval/param_accel": float(_normalized_trainable_accels(params, model).mean()),
        "eval/reversal_rate": _reversal_rate(params, model),
    }
    if model.config.f0.enabled:
        voiced = eval_setup.val_voiced.to(device)
        f0_loss, voiced_loss, readouts = _f0_losses(eval_aux, model, f0, voiced)
        out["eval/f0_loss"] = f0_loss.item()
        out["eval/voiced_loss"] = voiced_loss.item()
        out.update({f"eval/{k}": v for k, v in readouts.items()})
    if cfg.loss.rest_targets:
        rest_dist = _rest_pose_distance(params, model, cfg.loss.rest_targets)
        out["eval/rest_loss"] = _rest_pose_loss(
            params, model, cfg.loss.rest_targets, cfg.loss.rest_weights
        ).item()
        for name, v in zip(model.trainable_names_, rest_dist.tolist()):
            if name in cfg.loss.rest_targets:
                out[f"eval/rest_dist/{name}"] = v
        out.update(_silent_frame_rest_metrics(params, model, target, cfg))

    # Per-param variation too: the scalar mean hides which param is jittery,
    # which is exactly what the smoothing/rate-limit experiments turn on.
    per_param = _normalized_trainable_diffs(params, model).mean(dim=(0, 1))
    for name, v in zip(model.trainable_names_, per_param.tolist()):
        out[f"eval/param_variation/{name}"] = v
    for name, value in components.items():
        out[f"eval/recon/{name}"] = value.item()

    # Pitch MAE + ASR (per clip, CPU).
    pitch_vals: list[float] = []
    miss_vals: list[float] = []
    wer_vals: list[float] = []
    cer_vals: list[float] = []
    sr = cfg.data.sample_rate
    whisper = eval_setup.whisper
    N = target.shape[0]
    for i in range(N):
        tgt_np = target[i].detach().cpu().numpy()
        pred_np = pred_norm[i].detach().cpu().numpy()

        # This eval is slow and doesn't make sense to run if we're not learning the f0
        if "frequency" in model.trainable_names_:
            m = pitch_mae_cents(
                tgt_np,
                pred_np,
                sr,
                fmin=cfg.log.pitch_fmin,
                fmax=cfg.log.pitch_fmax,
                voiced_prob_threshold=cfg.log.pitch_voiced_prob_threshold,
            )
            if not np.isnan(m.mae_cents):
                pitch_vals.append(m.mae_cents)
                miss_vals.append(m.unvoiced_miss_frac)

        if whisper is not None:
            scores = whisper.score(
                tgt_np,
                pred_np,
                sr,
                target_key=eval_setup.val_target_keys[i],
            )
            if not np.isnan(scores.wer):
                wer_vals.append(scores.wer)
            if not np.isnan(scores.cer):
                cer_vals.append(scores.cer)

    if pitch_vals:
        out["eval/pitch_mae_cents"] = float(np.mean(pitch_vals))
        out["eval/unvoiced_miss_frac"] = float(np.mean(miss_vals))
    if wer_vals:
        out["eval/wer"] = float(np.mean(wer_vals))
    if cer_vals:
        out["eval/cer"] = float(np.mean(cer_vals))

    # Audio / params / mel for a random subset (seeded by ``step``).
    media_idx = _audio_sample_indices(step, N, cfg.log.n_audio_samples)
    if media_idx:
        gap = np.zeros(int(sr * 0.1), dtype=np.float32)
        trainable_names = model.trainable_names_
        bounds = {n: model.config.param_spec[n][:2] for n in trainable_names}
        audios: list[wandb.Audio] = []
        param_figs: list[wandb.Plotly] = []
        mel_pairs: list[tuple[str, np.ndarray]] = []
        for i in media_idx:
            name = eval_setup.val_names[i]
            tgt_np = target[i].detach().cpu().numpy()
            pred_np = pred_norm[i].detach().cpu().numpy()
            # pred → tgt so listeners hear the prediction first.
            combined = np.concatenate([_norm(pred_np), gap, tgt_np.astype(np.float32)])
            audios.append(wandb.Audio(combined, sample_rate=sr, caption=name))
            param_figs.append(
                wandb.Plotly(
                    _param_traj_figure(
                        params[i], trainable_names, frame_rate, bounds=bounds
                    )
                )
            )
            mel_pairs.append((f"{name}: target", tgt_np))
            mel_pairs.append((f"{name}: pred", pred_np))
        out["eval/audio"] = audios
        out["eval/params"] = param_figs
        out["eval/mel"] = wandb.Plotly(_mel_fig_stacked(mel_pairs, sr))

    if model_was_training:
        ddp_module.train()
    out["eval/duration_s"] = time.perf_counter() - t0
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    # Synth occasionally emits non-finite samples for pathological controls;
    # sanitize before normalising so wandb.Audio (int16) isn't fed garbage.
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
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
    for field in ("smooth_weights", "accel_weights", "rest_targets", "rest_weights"):
        unknown = set(getattr(cfg.loss, field)) - set(model.trainable_names_)
        if unknown:
            raise ValueError(
                f"loss.{field} names non-trainable params: {sorted(unknown)}"
            )
    for name, value in cfg.loss.rest_targets.items():
        lo, hi = cfg.model.param_spec[name][:2]
        if not lo <= value <= hi:
            raise ValueError(
                f"loss.rest_targets[{name}]={value} outside param range [{lo}, {hi}]"
            )
    frame_rate = cfg.model.frame_rate
    samples_per_frame = cfg.model.samples_per_frame
    loss_components: list[tuple[str, float, nn.Module]] = [
        (
            "mfcc",
            cfg.loss.mfcc,
            MFCCLoss(
                samples_per_frame=samples_per_frame,
                n_fft=cfg.loss.mfcc_n_fft,
            ),
        ),
        ("mel", cfg.loss.mel, MelSpecLoss(samples_per_frame=samples_per_frame)),
        ("stft", cfg.loss.stft, MultiScaleLogMagSTFTLoss()),
    ]
    # Only build the SSL encoder when it's actually weighted — CombinedReconLoss
    # evaluates every component each step (for logging), so a zero-weight SSL
    # entry would run the encoder for nothing.
    if cfg.loss.ssl != 0.0:
        loss_components.append(
            (
                "ssl",
                cfg.loss.ssl,
                SSLFeatureLoss(
                    model_name=cfg.loss.ssl_model,
                    layer=cfg.loss.ssl_layer,
                    distance=cfg.loss.ssl_distance,
                    source_sr=cfg.data.sample_rate,
                ),
            )
        )
    loss_fn = CombinedReconLoss(loss_components).to(device)

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
        cfg.data,
        batch_size=cfg.batch_size,
        rank=rank,
        world_size=world_size,
        epoch=0,
        seed=cfg.run.seed,
        samples_per_frame=samples_per_frame,
    )

    # Eval setup (rank 0 only): preloads the fixed val-clip tensors + ASR model.
    eval_setup: EvalSetup | None = None
    if rank == 0:
        eval_setup = _eval_setup(cfg, samples_per_frame, device)

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

    # Every local checkpoint is mirrored to wandb, keeping only the newest
    # version (see _log_ckpt_artifact).
    upload_ckpts = (
        rank == 0 and cfg.log.ckpt_wandb_artifact and cfg.log.wandb_mode != "disabled"
    )
    prune_ckpts = cfg.log.wandb_mode == "online"
    ckpt_artifacts: list[wandb.Artifact] = []

    if rank == 0 and eval_setup is not None:
        metrics = _evaluate(
            module,
            ddp_module,
            eval_setup,
            loss_fn,
            cfg,
            step,
            device=device,
        )
        wandb.log(metrics, step=step)
        tqdm.write(
            f"[eval] step={step} "
            f"loss={metrics.get('eval/loss', float('nan')):.4f} "
            f"wer={metrics.get('eval/wer', float('nan')):.3f} "
            f"in {metrics.get('eval/duration_s', float('nan')):.1f}s"
        )

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
        voiced = batch["voiced"].to(device, non_blocking=True)  # [B, T_ctrl]
        # One canonical loudness for both what the model hears and what it is
        # scored against; the synth output is not gain-matched below.
        wav = _rms_normalize(wav, cfg.data.target_rms)
        target = wav
        wav_in = wav.unsqueeze(1)  # [B, 1, S]

        # LR warmup
        for g in optimizer.param_groups:
            g["lr"] = cfg.optim.lr * _warmup_lr(step, cfg.optim.warmup_steps)

        tau = _tau_for_step(step, cfg)
        optimizer.zero_grad(set_to_none=True)

        # Encoder + head in bf16; synth & loss in fp32. We always pull aux
        # so we can use the logits for the entropy bonus and re-use them
        # for the diag log every log_every steps.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            params, aux = ddp_module(wav_in, f0, tau, return_aux=True)
        params = params.float()

        pred = pink_trombone_ola(
            params,
            ir_length=cfg.synth.ir_length,
            control_rate=frame_rate,
        )
        S = min(pred.shape[-1], target.shape[-1])
        pred_norm = pred[..., :S]
        S_norm = pred_norm.shape[-1]
        recon_loss, recon_components = loss_fn.with_components(
            pred_norm, target[..., :S_norm]
        )

        # Hinged entropy penalty on softmax(logits) — penalizes only
        # positions whose entropy drops below the floor, so logits can't
        # saturate to one-hot (which kills the soft-Gumbel gradient) but
        # positions above the floor feel no pressure toward uniform.
        # Hinged per position, not on the mean: uncertain positions must
        # not offset saturated ones.
        logits = aux["logits"].float()
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_per_pos = -(log_probs.exp() * log_probs).sum(-1)  # [B, T, P]
        entropy = entropy_per_pos.mean()
        entropy_penalty = F.relu(cfg.loss.entropy_floor - entropy_per_pos).mean()
        # Always computed (it's cheap) so the raw value can be compared across
        # runs; contributes to the gradient only when weighted.
        smooth_loss = _smoothness_loss(params, module, cfg.loss.smooth_weights)
        accel_loss = _acceleration_loss(params, module, cfg.loss.accel_weights)
        rest_loss = _rest_pose_loss(
            params, module, cfg.loss.rest_targets, cfg.loss.rest_weights
        )
        loss = (
            recon_loss
            + cfg.loss.entropy * entropy_penalty
            + cfg.loss.smooth * smooth_loss
            + cfg.loss.accel * accel_loss
            + cfg.loss.rest * rest_loss
        )

        f0_readouts: dict[str, float] = {}
        if module.config.f0.enabled:
            f0_loss, voiced_loss, f0_readouts = _f0_losses(aux, module, f0, voiced)
            loss = loss + cfg.loss.f0 * f0_loss + cfg.loss.voiced * voiced_loss
            f0_readouts["f0_loss"] = f0_loss.item()
            f0_readouts["voiced_loss"] = voiced_loss.item()

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
            log_payload: dict[str, object] = {
                "train/loss": loss.item(),
                "train/recon_loss": recon_loss.item(),
                "train/entropy": entropy.item(),
                "train/entropy_penalty": entropy_penalty.item(),
                "train/smooth_loss": smooth_loss.item(),
                "train/accel_loss": accel_loss.item(),
                **{f"train/{k}": v for k, v in f0_readouts.items()},
                "train/rest_loss": rest_loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/tau": tau,
                "train/grad_norm": float(grad_norm),
                "train/epoch": epoch,
                "System/throughput_per_gpu": throughput_per_gpu,
                "System/throughput_total": throughput_total,
            }
            for name, value in recon_components.items():
                log_payload[f"train/recon/{name}"] = value.item()
            log_payload["train/param_variation"] = _param_variation(params, module)
            log_payload.update(_controller_diagnostics(aux, model.trainable_names_))
            wandb.log(log_payload, step=step)

        if rank == 0 and eval_setup is not None and step % cfg.log.eval_every == 0:
            metrics = _evaluate(
                module,
                ddp_module,
                eval_setup,
                loss_fn,
                cfg,
                step,
                device=device,
            )
            wandb.log(metrics, step=step)
            tqdm.write(
                f"[eval] step={step} "
                f"loss={metrics.get('eval/loss', float('nan')):.4f} "
                f"wer={metrics.get('eval/wer', float('nan')):.3f} "
                f"in {metrics.get('eval/duration_s', float('nan')):.1f}s"
            )

        if rank == 0 and step % cfg.log.ckpt_every == 0:
            ckpt_path = _save_checkpoint(run_dir, step, module, optimizer, cfg)
            if upload_ckpts:
                _log_ckpt_artifact(
                    run_dir, ckpt_path, step, ckpt_artifacts, prune=prune_ckpts
                )

    pbar.close()
    if rank == 0:
        # The final step usually isn't a multiple of ckpt_every, so save and
        # upload it too; a step that was just checkpointed is skipped.
        if step % cfg.log.ckpt_every != 0:
            ckpt_path = _save_checkpoint(run_dir, step, module, optimizer, cfg)
            if upload_ckpts:
                _log_ckpt_artifact(
                    run_dir, ckpt_path, step, ckpt_artifacts, prune=prune_ckpts
                )
        if prune_ckpts:
            _prune_ckpt_artifacts(ckpt_artifacts)
        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
