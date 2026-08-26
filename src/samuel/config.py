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
    # Which detector produced the pitch cache; validated against the cache's
    # own metadata at load. "pyin" is Viterbi-smoothed and non-causal; "yin"
    # and "pesto" caches are computed causally (see scripts/precompute_pitch*)
    # so the same f0 track is reproducible by a streaming detector at
    # inference, and unvoiced gaps are filled causally (hold-last) rather than
    # by interpolation.
    pitch_source: Literal["pyin", "yin", "pesto"] = "pyin"
    # Every clip is RMS-normalised to this level
    target_rms: float = 0.05
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
    # defaults to max_steps when omitted in YAML. Equal endpoints (the default)
    # hold tau constant -- annealing was measured to add nothing.
    tau_start: float = 2.0
    tau_end: float = 2.0
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
    # Mirror every local checkpoint (and the final step) to a wandb artifact so
    # runs are backed up off the training filesystem. Only the newest version is
    # kept; older ones are deleted as they are replaced.
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

    # Temporal-smoothness penalty: L1 on the per-frame change of the predicted
    # control trajectories, computed on range-normalised params (each trainable
    # param rescaled to [0, 1]. Contribution to the training loss:
    #   smooth * sum_p smooth_weights[p] * mean_{batch,time} |Δp_norm|
    smooth: float = 0.3
    smooth_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tongueIndex": 1.0,
            "constrictionIndex": 1.0,
            "tongueDiameter": 0.3,
            "constrictionDiameter": 0.3,
            "lipDiameter": 0.3,
            "voiceness": 0.1,
            "intensity": 0.1,
        }
    )

    # Acceleration penalty: the same L1 on the *second* difference,
    #   accel * sum_p accel_weights[p] * mean_{batch,time} |Δ²p_norm|
    # This penalises direction changes rather than movement: a steady ramp costs
    # nothing however fast it is, which is the distinction ``smooth`` cannot
    # express. It has a collapse regime all the same -- a constant trajectory
    # has Δ² = 0 too -- so the weight is still bounded above, just less tightly.
    accel: float = 0.3
    accel_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tongueIndex": 1.0,
            "constrictionIndex": 1.0,
            "tongueDiameter": 0.3,
            "constrictionDiameter": 0.3,
            "lipDiameter": 0.3,
            "voiceness": 0.1,
            "intensity": 0.1,
        }
    )

    # Rest-posture prior: a small constant L1 pull of each control trajectory
    # toward a fixed "closed mouth" posture,
    #   rest * sum_p rest_weights[p] * mean_{batch,time} |p_norm - target_p|
    # on the same range-normalised params as ``smooth``/``accel``. Params
    # absent from ``rest_targets`` are unpenalised.
    #
    # The point is the frames where the reconstruction loss has no opinion:
    # during silence nothing pins the tongue, so it parks wherever it happens
    # to be. L1 makes the pull a constant force regardless of distance, so it
    # is a fixed small offset to the recon gradient -- negligible where that
    # gradient is strong (speech), decisive where it is weak or just noise
    # (silence). It has to stay small: these gradients are heavy-tailed, large
    # only in the frames where an articulator does its work, so a force even a
    # few percent above this wins in the low-gradient majority of frames and
    # the parameter stops being used at all.
    rest: float = 0.01
    # Target values in *raw* parameter units (same scale as model.param_spec),
    # normalised internally by the same [lo, hi] range. Empty disables the term.
    rest_targets: dict[str, float] = Field(
        default_factory=lambda: {
            "lipDiameter": 0.3,  # top of the [0, 0.3] closure band: almost closed
            "tongueDiameter": 3.5,  # max of [1.5, 3.5]: tongue body down
            "constrictionDiameter": 3.0,  # max of [-2, 3]: no tip constriction
        }
    )
    # Per-param multipliers; params absent here default to 1.0. The recon
    # gradient differs ~5x across these params, so a flat weight would bias the
    # lips far harder than the tongue.
    rest_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "lipDiameter": 0.3,
            "tongueDiameter": 1.0,
            "constrictionDiameter": 1.0,
        }
    )

    # SSL feature-matching (perceptual) loss on a frozen speech encoder.
    # L1 distance between the encoder's hidden states for pred vs. target audio.
    ssl: float = 1.0
    # HF model id. wav2vec2 is what the tuned recipe uses and every run since
    # has kept; an early comparison favoured microsoft/wavlm-base-plus, so this
    # is worth revisiting. facebook/hubert-base-ls960 also tried. Note wav2vec2
    # needs entropy>=0.1, which is the default -- see LossConfig.entropy.
    ssl_model: str = "facebook/wav2vec2-base-960h"
    ssl_layer: int = 6  # mid transformer layer is most phonetic
    ssl_distance: str = "L1"  # "L1" | "L2" | "cosine"


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
