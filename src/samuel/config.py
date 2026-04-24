"""Top-level pydantic schemas for the training run.

These are the schemas Hydra's ``DictConfig`` is resolved into. The hydra
groups in ``configs/`` populate each sub-config; pydantic validates the
whole tree on entry to ``train.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

from samuel.model import PinkTromboneControllerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_repo_relative(p: Path) -> Path:
    return p if p.is_absolute() else REPO_ROOT / p


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest_path: Path
    sample_rate: int = 44100
    chunk_seconds: float = 4.0
    num_workers: int = 4

    @field_validator("manifest_path")
    @classmethod
    def _resolve_manifest_path(cls, v: Path) -> Path:
        return _resolve_repo_relative(v)


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: int = 100_000
    warmup_steps: int = 1_000


class SynthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ir_length: int = 256
    ir_impl: Literal["eig", "sequential"] = "eig"
    # frame_rate is the parameter control rate; lives on the model config
    # (it drives T_ctrl) but the synth path reads it from the same field.


class WhisperConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    model_name: str = "openai/whisper-large-v3"
    # Layers (0-indexed) whose activations we match between pred and target.
    # Defaults span low/mid/high-level features.
    perceptual_layers: list[int] = Field(default_factory=lambda: [7, 15, 23, 31])
    # Loss mix weights. 0 disables that term.
    distill_weight: float = 1.0
    perceptual_weight: float = 1.0
    stft_weight: float = 1.0


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb_project: str = "samuel-trombone"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    log_every: int = 50
    val_every: int = 1_000
    eval_every: int = 10_000
    ckpt_every: int = 5_000
    n_audio_samples: int = 4
    pitch_fmin: float = 70.0
    pitch_fmax: float = 500.0
    pitch_voiced_prob_threshold: float = 0.5


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs_root: Path = Path("runs")
    name: str
    seed: int = 0

    @field_validator("runs_root")
    @classmethod
    def _resolve_runs_root(cls, v: Path) -> Path:
        return _resolve_repo_relative(v)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    data: DataConfig
    model: PinkTromboneControllerConfig = Field(
        default_factory=PinkTromboneControllerConfig
    )
    synth: SynthConfig = Field(default_factory=SynthConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    batch_size: int = 8

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "TrainConfig":
        data = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(data, dict)
        return cls.model_validate(data)
