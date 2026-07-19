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


class SynthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ir_length: int = 256
    # frame_rate is the parameter control rate; lives on the model config
    # (it drives T_ctrl) but the synth path reads it from the same field.


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wandb_project: str = "samuel-trombone"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    log_every: int = 50
    eval_every: int = 1_000
    ckpt_every: int = 5_000
    # On clean completion, upload the final checkpoint (last.pt) as a wandb
    # artifact so runs are backed up off the training filesystem.
    ckpt_wandb_artifact: bool = True
    # The same clips are used for each eval for stable metrics
    n_eval_clips: int = 100
    # Eval-clip length in seconds. None -> use data.chunk_seconds (match training)
    eval_chunk_seconds: float | None = None
    # Subset of those clips for which we attach audio/params/mel media to
    # wandb. The subset is re-sampled every eval step (deterministic by
    # step) so listeners hear new examples without bloating storage.
    n_audio_samples: int = 10
    pitch_fmin: float = 70.0
    pitch_fmax: float = 500.0
    pitch_voiced_prob_threshold: float = 0.5
    # Whisper model size for the WER/CER eval. Empty string disables ASR eval.
    asr_whisper_size: str = "base"


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

    # Reconstruction-loss weights. Components with weight 0 are skipped.
    # Total training loss is:
    #   sum(w_i * loss_i(pred, target))
    #     + entropy * mean(relu(entropy_floor - H_pos(softmax(logits))))
    # The hinged entropy penalty keeps each position's softmax from
    # saturating to one-hot (which kills the soft-Gumbel gradient), but
    # exerts zero pressure once a position's entropy is above the floor —
    # positions are free to commit down to the floor, and unused buckets
    # are allowed.
    mfcc: float = 1.0  # L1 on first 20 MFCCs (frame-aligned to samples_per_frame)
    mel: float = 0.0  # L1 on log-mel spectrogram (frame-aligned to samples_per_frame)
    stft: float = 0.0  # Multi-scale log-magnitude STFT, n_ffts (512, 1024, 2048)
    entropy: float = 1.0
    # Per-position entropy floor in nats (1.0 ~ spread over e ~ 2.7 buckets).
    entropy_floor: float = 1.0
    # MFCC-loss STFT window size. Default 2048 with samples_per_frame=512 gives
    # 4x window overlap and better-resolved spectra (vs. no overlap at
    # n_fft=samples_per_frame), which improves voicedness and recon. Set to
    # None to revert to n_fft = samples_per_frame.
    mfcc_n_fft: int | None = 2048

    # SSL feature-matching (perceptual) loss on a frozen speech encoder.
    # L1 distance between the encoder's hidden states for pred vs. target audio.
    ssl_distill: float = 0.0
    # HF model id. Others tried: facebook/hubert-base-ls960,
    # facebook/wav2vec2-base-960h (wavlm won; wav2vec2 needs entropy>=0.1).
    ssl_distill_model: str = "microsoft/wavlm-base-plus"
    ssl_distill_layer: int = 6  # mid transformer layer is most phonetic
    ssl_distill_distance: str = "L1"  # "L1" | "L2" | "cosine"

    # Frame-wise KL between CTC character posteriors of a frozen ASR model.
    # Content-only perceptual loss: constrains which character is said when,
    # not timbre/prosody. No CTC marginalization (alignment comes from the
    # target audio).
    asr_distill: float = 0.0
    asr_distill_model: str = "facebook/wav2vec2-base-960h"
    # Softmax temperature on teacher+student logits. T=1 gives near-one-hot
    # teachers whose KL gradient spikes saturate the Gumbel head; T>1 softens.
    asr_distill_temperature: float = 2.0


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


class RLConfig(BaseModel):
    """Top-level schema for RL post-training (``rl_train.py``).

    Deliberately separate from ``TrainConfig``: RL uses rewards rather than
    the supervised reconstruction ``loss``/``optim`` blocks, so those are
    omitted here. The data/model/synth/log building blocks are shared.
    """

    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    data: DataConfig
    model: PinkTromboneControllerConfig = Field(
        default_factory=PinkTromboneControllerConfig
    )
    synth: SynthConfig = Field(default_factory=SynthConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    batch_size: int = 8
    # Warm-start weights. Either a local path to a ``.pt`` checkpoint or a
    # wandb artifact reference (``entity/project/name:alias``). None starts
    # from a freshly initialised model.
    checkpoint: str | None = None

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> "RLConfig":
        data = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(data, dict)
        return cls.model_validate(data)
