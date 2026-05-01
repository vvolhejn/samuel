"""1D-CNN controller that turns audio into Pink Trombone parameter trajectories.

When ``joint=False`` the head emits ``n_buckets`` logits *per* trainable
parameter — a factorized categorical with the per-param distributions
independent. When ``joint=True`` a single categorical over
``n_buckets ** n_trainable`` joint classes replaces the factorized head;
each class encodes a full assignment of buckets across all trainable
params. Training uses a (soft) Gumbel-softmax expectation of bucket
centers; eval takes the argmax class. ``frequency`` is supplied externally
(precomputed pyin) and ``intensity`` is frozen to 1.0.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from samuel.encoder import SEANetEncoder, SEANetEncoderConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, SAMPLE_RATE

# (lo, hi, init) per trainable parameter. ``frequency`` and ``intensity`` are
# intentionally absent — frequency comes from pyin, intensity is frozen.
_DEFAULT_PARAM_SPEC: dict[str, tuple[float, float, float]] = {
    "voiceness": (0.0, 1.0, 0.6),
    "tongueIndex": (10.0, 35.0, 20.0),
    "tongueDiameter": (1.5, 3.5, 2.4),
    "constrictionIndex": (22.0, 44.0, 33.0),
    "constrictionDiameter": (-0.5, 3.0, 1.25),
}
_DEFAULT_FROZEN_VALUES: dict[str, float] = {
    "intensity": 1.0,
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
    samples_per_frame: int = 2048
    n_buckets: int = 8
    # When True, the head predicts a single categorical over
    # ``n_buckets ** n_trainable`` joint classes instead of a factorized
    # categorical (``n_buckets`` per param). Useful to test whether
    # modeling cross-param coupling helps over independent per-param heads.
    joint: bool = False

    @property
    def frame_rate(self) -> float:
        return SAMPLE_RATE / self.samples_per_frame

    def trainable_names(self) -> list[str]:
        """Trainable parameter names in PARAM_NAMES order."""
        return [n for n in PARAM_NAMES if n in self.param_spec]

    def validate_coverage(self) -> None:
        trainable = set(self.param_spec)
        frozen = set(self.frozen_values)
        overlap = trainable & frozen
        if overlap:
            raise ValueError(f"params in both param_spec and frozen_values: {overlap}")
        if "frequency" in trainable or "frequency" in frozen:
            raise ValueError(
                "'frequency' must not appear in param_spec or frozen_values; "
                "it is supplied externally from the pyin cache"
            )
        covered = trainable | frozen | {"frequency"}
        missing = set(PARAM_NAMES) - covered
        if missing:
            raise ValueError(
                f"params covered by neither param_spec, frozen_values, nor "
                f"external frequency: {missing}"
            )
        unknown = (trainable | frozen) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"unknown Pink Trombone params: {unknown}")


class PinkTromboneController(nn.Module):
    """SEANet encoder -> categorical head over ``n_buckets`` per parameter."""

    def __init__(self, config: PinkTromboneControllerConfig):
        super().__init__()
        config.validate_coverage()
        self.config = config

        self.samples_per_frame = config.samples_per_frame
        self.n_buckets = config.n_buckets
        self.joint = config.joint
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
        # Bucket centers cover the full [lo, hi] range including endpoints.
        # ``voiceness=0`` (extreme breathiness) used to produce NaN gradients
        # via ``voiceness**0.25`` in pink_trombone.py; that's now clamped at
        # the synth boundary so endpoints are safe.
        steps = torch.linspace(0.0, 1.0, config.n_buckets, dtype=torch.float32)
        # [n_trainable, n_buckets]
        centers = lo.unsqueeze(1) + steps.unsqueeze(0) * (hi - lo).unsqueeze(1)
        self.register_buffer("bucket_centers", centers)

        if self.joint:
            # Joint head: K = n_buckets ** n_trainable classes. Each class k
            # encodes a tuple of per-param bucket indices via base-n_buckets
            # decomposition. ``joint_centers[k, j]`` is the value of param j
            # for class k — precompute once so the forward is a matmul.
            n_classes = config.n_buckets**n_trainable
            self._n_classes = n_classes
            digits = torch.arange(n_classes)
            joint_idx = torch.stack(
                [
                    (digits // (config.n_buckets**j)) % config.n_buckets
                    for j in range(n_trainable)
                ],
                dim=-1,
            )  # [K, n_trainable]
            joint_centers = torch.gather(
                centers.unsqueeze(0).expand(n_classes, -1, -1),
                dim=-1,
                index=joint_idx.unsqueeze(-1),
            ).squeeze(-1)  # [K, n_trainable]
            self.register_buffer("joint_centers", joint_centers)
            self.register_buffer("joint_idx", joint_idx)
            self.head = nn.Linear(config.encoder.dimension, n_classes)
        else:
            self.head = nn.Linear(
                config.encoder.dimension, n_trainable * config.n_buckets
            )
        # Bias init at zero -> uniform softmax -> mean bucket value at start.
        with torch.no_grad():
            self.head.bias.zero_()

        trainable_idx = torch.tensor(
            [PARAM_NAMES.index(n) for n in trainable], dtype=torch.long
        )
        self.register_buffer("_trainable_idx", trainable_idx)

        self._freq_idx: int = PARAM_NAMES.index("frequency")

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

    def forward(
        self,
        wav: Tensor,
        f0: Tensor,
        tau: float = 1.0,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Predict Pink Trombone parameter trajectories.

        Args:
            wav: ``[B, 1, S]`` audio at 44.1 kHz.
            f0: ``[B, T_ctrl]`` fundamental frequency in Hz per control frame.
                Already interpolated through unvoiced regions and clamped to a
                sane range.
            tau: Gumbel-softmax temperature (training only).
            return_aux: if True, also return a dict with ``logits``
                ``[B, T_ctrl, n_trainable, n_buckets]`` and ``z`` (encoder
                output, ``[B, dim, T_ctrl]``) for diagnostics.

        Returns:
            ``[B, T_ctrl, N_PARAMS]`` parameter tensor (and ``aux`` dict if
            ``return_aux=True``).
        """
        if wav.ndim != 3:
            raise ValueError(f"expected wav [B, 1, S], got {tuple(wav.shape)}")
        B, _C, S = wav.shape
        T_ctrl = self.t_ctrl_for(S)
        if f0.shape != (B, T_ctrl):
            raise ValueError(f"expected f0 [{B}, {T_ctrl}], got {tuple(f0.shape)}")

        hop = self.encoder.hop_length
        pad = (hop - S % hop) % hop
        if pad > 0:
            wav = F.pad(wav, (0, pad))

        z = self.encoder(wav)  # [B, dim, T_enc]
        if z.shape[-1] != T_ctrl:
            z = F.interpolate(z, size=T_ctrl, mode="linear", align_corners=True)

        n_trainable = self.bucket_centers.shape[0]
        head_out = self.head(z.transpose(1, 2)).float()  # [B, T_ctrl, *]

        if self.joint:
            # Joint categorical over K = n_buckets ** n_trainable classes.
            logits = head_out  # [B, T_ctrl, K]
            if self.training:
                weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            else:
                argmax = logits.argmax(dim=-1)
                weights = F.one_hot(argmax, num_classes=self._n_classes).to(
                    logits.dtype
                )
            # weights: [B, T, K], joint_centers: [K, n_trainable]
            constrained = weights @ self.joint_centers
        else:
            logits = head_out.view(B, T_ctrl, n_trainable, self.n_buckets)
            if self.training:
                # hard=False: the forward output is the soft Gumbel-softmax
                # distribution; (weights * centers).sum is then a smooth
                # expectation between bucket centers. Eval still snaps to the
                # argmax bucket, so there's a mild train/eval mismatch — but
                # hard=True (straight-through) tends to lock the argmax in this
                # setup, with eval loss bit-identical across many steps.
                weights = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
            else:
                argmax = logits.argmax(dim=-1)
                weights = F.one_hot(argmax, num_classes=self.n_buckets).to(logits.dtype)
            constrained = (weights * self.bucket_centers).sum(dim=-1)
        # constrained: [B, T_ctrl, n_trainable]

        out = torch.zeros(
            B, T_ctrl, N_PARAMS, device=wav.device, dtype=constrained.dtype
        )
        train_idx = self._trainable_idx.view(1, 1, -1).expand(B, T_ctrl, -1)
        out = out.scatter(2, train_idx, constrained)

        if self._frozen_idx.numel() > 0:
            frozen_idx = self._frozen_idx.view(1, 1, -1).expand(B, T_ctrl, -1)
            frozen_vals = self._frozen_vals.view(1, 1, -1).expand(B, T_ctrl, -1)
            out = out.scatter(2, frozen_idx, frozen_vals.to(out.dtype))

        freq_idx = torch.full(
            (B, T_ctrl, 1), self._freq_idx, device=out.device, dtype=torch.long
        )
        out = out.scatter(2, freq_idx, f0.unsqueeze(-1).to(out.dtype))
        if return_aux:
            aux: dict[str, Tensor] = {"logits": logits, "z": z}
            if self.joint:
                aux["joint_idx"] = self.joint_idx
            return out, aux
        return out
