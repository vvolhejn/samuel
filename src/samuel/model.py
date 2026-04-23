"""1D-CNN controller that turns audio into Pink Trombone parameter trajectories."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from samuel.encoder import SEANetEncoder, SEANetEncoderConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, SAMPLE_RATE

# Defaults mirror notebooks/tract_fit.ipynb (9 trainable, 4 frozen).
_DEFAULT_PARAM_SPEC: dict[str, tuple[float, float, float]] = {
    "noise": (0.0, 0.5, 0.1),
    "frequency": (80.0, 400.0, 140.0),
    "tenseness": (0.0, 1.0, 0.6),
    "intensity": (0.0, 1.0, 1.0),
    "loudness": (0.0, 1.0, 1.0),
    "tongueIndex": (10.0, 35.0, 20.0),
    "tongueDiameter": (1.5, 3.5, 2.4),
    "constrictionIndex": (0.0, 44.0, 30.0),
    "constrictionDiameter": (-0.5, 3.0, 3.0),
}
_DEFAULT_FROZEN_VALUES: dict[str, float] = {
    "vibratoWobble": 0.0,
    "vibratoFrequency": 6.0,
    "vibratoGain": 0.0,
    "tractLength": 44.0,
}


class PinkTromboneControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    encoder: SEANetEncoderConfig = Field(default_factory=SEANetEncoderConfig)
    param_spec: dict[str, tuple[float, float, float]] = Field(
        default_factory=lambda: dict(_DEFAULT_PARAM_SPEC)
    )
    frozen_values: dict[str, float] = Field(
        default_factory=lambda: dict(_DEFAULT_FROZEN_VALUES)
    )
    frame_rate: float = 12.5  # parameter control rate, Hz

    def trainable_names(self) -> list[str]:
        """Trainable parameter names in PARAM_NAMES order."""
        return [n for n in PARAM_NAMES if n in self.param_spec]

    def validate_coverage(self) -> None:
        trainable = set(self.param_spec)
        frozen = set(self.frozen_values)
        overlap = trainable & frozen
        if overlap:
            raise ValueError(f"params in both param_spec and frozen_values: {overlap}")
        missing = set(PARAM_NAMES) - trainable - frozen
        if missing:
            raise ValueError(
                f"params covered by neither param_spec nor frozen_values: {missing}"
            )
        unknown = (trainable | frozen) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"unknown Pink Trombone params: {unknown}")


class PinkTromboneController(nn.Module):
    """SEANet encoder -> linear head -> sigmoid-bounded Pink Trombone params."""

    def __init__(self, config: PinkTromboneControllerConfig):
        super().__init__()
        config.validate_coverage()
        self.config = config

        self.samples_per_frame = int(round(SAMPLE_RATE / config.frame_rate))
        self.encoder = SEANetEncoder(config.encoder)

        trainable = config.trainable_names()
        self.trainable_names_: list[str] = trainable
        n_trainable = len(trainable)

        lo = torch.tensor(
            [config.param_spec[n][0] for n in trainable], dtype=torch.float32
        )
        hi = torch.tensor(
            [config.param_spec[n][1] for n in trainable], dtype=torch.float32
        )
        init = torch.tensor(
            [config.param_spec[n][2] for n in trainable], dtype=torch.float32
        )
        init_norm = ((init - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4)
        bias_init = torch.log(init_norm / (1 - init_norm))
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)

        self.head = nn.Linear(config.encoder.dimension, n_trainable)
        with torch.no_grad():
            nn.init.normal_(self.head.weight, std=0.01)
            self.head.bias.copy_(bias_init)

        trainable_idx = torch.tensor(
            [PARAM_NAMES.index(n) for n in trainable], dtype=torch.long
        )
        self.register_buffer("_trainable_idx", trainable_idx)

        frozen_items = list(config.frozen_values.items())
        frozen_idx = torch.tensor(
            [PARAM_NAMES.index(n) for n, _ in frozen_items], dtype=torch.long
        )
        frozen_vals = torch.tensor([v for _, v in frozen_items], dtype=torch.float32)
        self.register_buffer("_frozen_idx", frozen_idx)
        self.register_buffer("_frozen_vals", frozen_vals)

    def t_ctrl_for(self, n_samples: int) -> int:
        """Number of control frames the model will emit for a given waveform length."""
        return math.ceil(n_samples / self.samples_per_frame)

    def forward(self, wav: Tensor) -> Tensor:
        """``wav [B, 1, S]`` (float32, 44.1 kHz) -> ``params [B, T_ctrl, 13]``."""
        if wav.ndim != 3:
            raise ValueError(f"expected [B, 1, S], got {tuple(wav.shape)}")
        B, _C, S = wav.shape
        T_ctrl = self.t_ctrl_for(S)

        # SEANetEncoder requires S % prod(ratios) == 0; zero-pad on the right.
        hop = self.encoder.hop_length
        pad = (hop - S % hop) % hop
        if pad > 0:
            wav = F.pad(wav, (0, pad))

        z = self.encoder(wav)  # [B, dim, T_enc]
        if z.shape[-1] != T_ctrl:
            z = F.interpolate(z, size=T_ctrl, mode="linear", align_corners=True)

        # Head + sigmoid in fp32 so small bf16-range gradients don't underflow.
        raw = self.head(z.transpose(1, 2)).float()  # [B, T_ctrl, n_trainable]
        constrained = torch.sigmoid(raw) * (self._hi - self._lo) + self._lo

        out = torch.zeros(
            B, T_ctrl, N_PARAMS, device=wav.device, dtype=constrained.dtype
        )
        train_idx = self._trainable_idx.view(1, 1, -1).expand(B, T_ctrl, -1)
        out = out.scatter(2, train_idx, constrained)

        if self._frozen_idx.numel() > 0:
            frozen_idx = self._frozen_idx.view(1, 1, -1).expand(B, T_ctrl, -1)
            frozen_vals = self._frozen_vals.view(1, 1, -1).expand(B, T_ctrl, -1)
            out = out.scatter(2, frozen_idx, frozen_vals.to(out.dtype))

        return out
