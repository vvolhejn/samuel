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
    pitch_cache_path: Path | None = None
    # Fraction of the manifest reserved as the held-out validation split.
    # Files at the tail of the manifest (after the train cut) are never seen
    # during training; eval samples from them for the val_* metrics.
    val_fraction: float = 0.05

    @field_validator("manifest_path")
    @classmethod
    def _resolve_manifest_path(cls, v: Path) -> Path:
        return _resolve_repo_relative(v)

    @field_validator("pitch_cache_path")
    @classmethod
    def _resolve_pitch_cache(cls, v: Path | None) -> Path | None:
        return _resolve_repo_relative(v) if v is not None else None


class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_steps: int = 100_000
    warmup_steps: int = 1_000
    # Gumbel-softmax temperature: linear anneal from tau_start to tau_end over
    # the first tau_anneal_steps; afterwards held at tau_end. tau_anneal_steps
    # defaults to max_steps when omitted in YAML.
    tau_start: float = 2.0
    tau_end: float = 0.5
    tau_anneal_steps: int | None = None
    # Coefficient on the entropy bonus subtracted from the loss:
    #   loss = mfcc_loss - entropy_weight * mean_entropy(softmax(logits))
    # Higher values keep the per-position softmax distribution closer to
    # uniform → stops the head's logits from running off to one-hot, which
    # otherwise makes the soft Gumbel effectively hard and kills gradients.
    entropy_weight: float = 0.01


class SynthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ir_length: int = 256
    ir_impl: Literal["eig", "sequential"] = "eig"
    # frame_rate is the parameter control rate; lives on the model config
    # (it drives T_ctrl) but the synth path reads it from the same field.


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


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # "mfcc": L1 on first 20 MFCCs (frame-aligned to samples_per_frame).
    # "mel":  L1 on log-mel spectrogram (frame-aligned to samples_per_frame).
    # "stft": Multi-scale log-magnitude STFT, n_ffts (512, 1024, 2048).
    type: Literal["mfcc", "mel", "stft"] = "mfcc"


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
    loss: LossConfig = Field(default_factory=LossConfig)
    batch_size: int = 8

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "TrainConfig":
        data = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(data, dict)
        return cls.model_validate(data)
