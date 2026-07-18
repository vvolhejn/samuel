"""Hydra entry point for RL post-training the Pink Trombone controller with GRPO.

Launch example:

    uv run python -m samuel.rl_train run.name=grpo-smoke
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from samuel.config import RLConfig
from samuel.data import build_dataloader
from samuel.evals.asr import WhisperEvaluator
from samuel.model import PinkTromboneController


def _resolve_checkpoint(ref: str) -> Path:
    """Return a local path to the checkpoint file.

    ``ref`` is either a local ``.pt`` path or a wandb artifact reference
    (``entity/project/name:alias``), which is downloaded on demand.
    """
    if Path(ref).exists():
        return Path(ref)

    import wandb

    artifact = wandb.Api().artifact(ref, type="model")
    return Path(artifact.download()) / "last.pt"


def _load_checkpoint(
    model: PinkTromboneController, ref: str, device: torch.device
) -> None:
    ckpt = torch.load(_resolve_checkpoint(ref), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])


@hydra.main(version_base=None, config_path="../../configs", config_name="rl_train")
def main(hydra_cfg: DictConfig) -> None:
    cfg = RLConfig.from_hydra(hydra_cfg)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(cfg.run.seed)

    # Model
    model = PinkTromboneController(cfg.model).to(device)
    samples_per_frame = cfg.model.samples_per_frame
    if cfg.checkpoint is not None:
        _load_checkpoint(model, cfg.checkpoint, device)

    # Data
    _loader = build_dataloader(
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

    _whisper = WhisperEvaluator(
        model_size=cfg.log.asr_whisper_size,
        device="cuda" if device.type == "cuda" else "cpu",
    )

    # GRPO
    raise NotImplementedError


if __name__ == "__main__":
    main()
