"""1D-CNN controller that turns audio into Pink Trombone parameter trajectories.

The head emits a categorical distribution over ``n_buckets`` evenly spaced
values per trainable parameter. During training a Gumbel-softmax sample
weights the bucket centers (soft by default; one-hot straight-through with
``gumbel_hard``); at eval time the argmax bucket is used.
The ``frequency`` parameter is supplied externally (precomputed pyin).
``intensity`` is trainable and carries the energy contour: the synth output is
never gain-matched, so the model has to produce the level itself (see
``data.target_rms``).

Trajectories can additionally be passed through a slew-rate limiter
(``rate_limits``) that hard-bounds how fast each parameter may move. It is an
architectural alternative to the L1 smoothness penalty in the loss: instead of
paying for jitter, the model simply cannot emit it.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from samuel.encoder import SEANetEncoder, SEANetEncoderConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, SAMPLE_RATE

# (lo, hi, init) per trainable parameter. ``frequency`` is intentionally
# absent — it comes from pyin.
_DEFAULT_PARAM_SPEC: dict[str, tuple[float, float, float]] = {
    "voiceness": (0.0, 1.0, 0.6),
    # Overall gain / voicing onset. Trainable because nothing downstream
    # corrects the output level.
    "intensity": (0.0, 1.0, 1.0),
    "tongueIndex": (10.0, 35.0, 20.0),
    "tongueDiameter": (1.5, 3.5, 2.4),
    # Capped at _LIP_START (39): this constriction models the tongue tip; the
    # lips get their own fixed-position constriction via lipDiameter.
    "constrictionIndex": (22.0, 39.0, 33.0),
    # constrictionDiameter controls the oral constriction and, past the nose
    # start, the velum. Effect by interval (thresholds from pink_trombone.py):
    #   < -1.65        : velum open, oral tract untouched      -> nasal vowels
    #   [-1.65, -0.8)  : velum open, oral tract clamped shut   -> nasal consonants (m/n/ng)
    #   [-0.8, 0.3]    : velum closed, full oral closure        -> oral stops (b/d/g, p/t/k)
    #   (0.3, 0.7)     : narrow opening + turbulence injected   -> fricatives (s/f, ...)
    #   >= 0.7         : wide opening, no turbulence; once the constriction
    #                    exceeds the local rest diameter it has no effect
    #                                                            -> approximants / open vowels
    "constrictionDiameter": (-2.0, 3.0, 1.25),
    # Second constriction fixed at the last tract index (the lips).
    # Non-negative: [0, 0.3] full closure (b/p), (0.3, 0.7) fricative (f),
    # >= 0.7 open. No nasal range — the velum stays driven by
    # constrictionDiameter only.
    "lipDiameter": (0.0, 3.0, 1.25),
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
    samples_per_frame: int = 2048
    n_buckets: int = 32
    # Straight-through Gumbel-softmax during training: the forward output is a
    # one-hot sample (matching eval's argmax snap exactly), gradients flow
    # through the soft distribution. With False (default), the forward output
    # is the soft distribution and the synth sees a smooth expectation between
    # bucket centers.
    gumbel_hard: bool = False

    # Slew-rate limiter: maximum speed per trainable parameter, in *fractions
    # of that parameter's range per second* (so 2.0 = may traverse the whole
    # [lo, hi] range in 0.5 s). Applied to the emitted trajectory at both train
    # and eval time, before the params reach the synth, so it is a genuine
    # constraint on the model's output rather than a penalty it can pay.
    # Params absent from the dict are unconstrained.
    #
    # Split scalar/dict like loss.smooth and loss.smooth_weights: the effective
    # limit is ``rate_limit_scale * rate_limits[p]``, so a sweep over the
    # tightness of the constraint is a single scalar override.
    # ``rate_limit_scale: null`` (the default) disables the limiter entirely.
    # Ordering vs. loss.smooth_weights is inverse: a *large* limit is the
    # counterpart of a *small* smoothness weight (constrictionDiameter must
    # snap for plosives; tongueIndex should glide).
    rate_limit_scale: float | None = None
    rate_limits: dict[str, float] = Field(default_factory=dict)
    # How the per-frame increment is bounded.
    #   "clamp": delta <- clip(delta, +-d). Exact box constraint; a saturated
    #     frame gets zero gradient to its own logits (the gradient passes
    #     through to the previous frame with weight 1, so nothing vanishes).
    #   "tanh": delta <- d * tanh(delta / d). Same hard bound |delta| < d,
    #     but smooth and everywhere-differentiable; near-linear for small
    #     moves, so it only distorts trajectories that were pushing the limit.
    rate_limit_mode: Literal["clamp", "tanh"] = "tanh"

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
        unknown_limits = set(self.rate_limits) - trainable
        if unknown_limits:
            raise ValueError(
                f"rate_limits names non-trainable params: {sorted(unknown_limits)}"
            )
        nonpositive = {n: v for n, v in self.rate_limits.items() if v <= 0.0}
        if nonpositive:
            raise ValueError(f"rate_limits must be positive, got: {nonpositive}")
        if self.rate_limit_scale is not None and self.rate_limit_scale <= 0.0:
            raise ValueError(
                "rate_limit_scale must be positive or null (null = limiter off), "
                f"got {self.rate_limit_scale}"
            )

    def effective_rate_limits(self) -> dict[str, float]:
        """``rate_limits`` after applying the scale; empty when disabled."""
        if self.rate_limit_scale is None:
            return {}
        return {n: self.rate_limit_scale * v for n, v in self.rate_limits.items()}


def slew_rate_limit(
    values: Tensor, max_delta: Tensor, mode: Literal["clamp", "tanh"] = "clamp"
) -> Tensor:
    """Bound the per-frame change of a trajectory to ``+-max_delta``.

    Args:
        values: ``[B, T, P]`` raw trajectory.
        max_delta: ``[P]`` maximum absolute change per frame, in the same units
            as ``values``. Must be strictly positive.
        mode: ``"clamp"`` (exact box) or ``"tanh"`` (smooth, same hard bound).

    Returns:
        ``[B, T, P]`` with ``|out[:, t] - out[:, t-1]| <= max_delta``. The
        first frame is passed through unchanged.

    The recursion compares against the *limited* previous value, so the bound
    holds on the realised trajectory rather than on the raw one. It is a
    sequential scan over T; T is a few hundred frames, which is negligible next
    to the sample-rate waveguide downstream.
    """
    T = values.shape[1]
    prev = values[:, 0]
    out = [prev]
    for t in range(1, T):
        delta = values[:, t] - prev
        if mode == "clamp":
            delta = torch.clamp(delta, -max_delta, max_delta)
        else:
            delta = max_delta * torch.tanh(delta / max_delta)
        prev = prev + delta
        out.append(prev)
    return torch.stack(out, dim=1)


class PinkTromboneController(nn.Module):
    """SEANet encoder -> categorical head over ``n_buckets`` per parameter."""

    def __init__(self, config: PinkTromboneControllerConfig):
        super().__init__()
        config.validate_coverage()
        self.config = config

        self.samples_per_frame = config.samples_per_frame
        self.n_buckets = config.n_buckets
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

        # Slew-rate limits, converted from range-fractions/second to absolute
        # param units per control frame.
        effective = config.effective_rate_limits()
        limited = [n for n in trainable if n in effective]
        self.rate_limited_names_: list[str] = limited
        rl_idx = torch.tensor([trainable.index(n) for n in limited], dtype=torch.long)
        rl_delta = torch.tensor(
            [
                effective[n]
                * (config.param_spec[n][1] - config.param_spec[n][0])
                / config.frame_rate
                for n in limited
            ],
            dtype=torch.float32,
        )
        # Non-persistent: fully derived from the config, and keeping them out
        # of the state_dict lets pre-rate-limit checkpoints load unchanged.
        self.register_buffer("_rate_limit_idx", rl_idx, persistent=False)
        self.register_buffer("_rate_limit_delta", rl_delta, persistent=False)

        self.head = nn.Linear(config.encoder.dimension, n_trainable * config.n_buckets)
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
                ``[B, T_ctrl, n_trainable, n_buckets]``, ``z`` (encoder
                output, ``[B, dim, T_ctrl]``) and ``raw_trainable``
                (``[B, T_ctrl, n_trainable]``, the trajectory before the
                slew-rate limiter) for diagnostics.

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

        logits = self.head(
            rearrange(z, "b d t -> b t d")
        ).float()  # [B, T_ctrl, n_t*n_b]
        logits = rearrange(logits, "b t (p k) -> b t p k", k=self.n_buckets)

        if self.training:
            # hard=False: the forward output is the soft Gumbel-softmax
            # distribution; (weights * centers).sum is then a smooth
            # expectation between bucket centers. Eval still snaps to the
            # argmax bucket, so there's a mild train/eval mismatch.
            # hard=True (straight-through) removes that mismatch but locked
            # the argmax when tried before the hinged entropy floor existed
            # (eval loss bit-identical across many steps) — watch
            # train/bucket_usage if enabling it.
            weights = F.gumbel_softmax(
                logits, tau=tau, hard=self.config.gumbel_hard, dim=-1
            )
        else:
            argmax = logits.argmax(dim=-1)
            weights = F.one_hot(argmax, num_classes=self.n_buckets).to(logits.dtype)

        constrained = (weights * self.bucket_centers).sum(dim=-1)  # [B, T_ctrl, n_t]

        raw = constrained
        if self._rate_limit_idx.numel() > 0:
            sub = constrained.index_select(2, self._rate_limit_idx)
            sub = slew_rate_limit(
                sub, self._rate_limit_delta, self.config.rate_limit_mode
            )
            constrained = constrained.index_copy(2, self._rate_limit_idx, sub)

        out = torch.zeros(
            B, T_ctrl, N_PARAMS, device=wav.device, dtype=constrained.dtype
        )
        train_idx = repeat(self._trainable_idx, "p -> b t p", b=B, t=T_ctrl)
        out = out.scatter(2, train_idx, constrained)

        if self._frozen_idx.numel() > 0:
            frozen_idx = repeat(self._frozen_idx, "p -> b t p", b=B, t=T_ctrl)
            frozen_vals = repeat(self._frozen_vals, "p -> b t p", b=B, t=T_ctrl)
            out = out.scatter(2, frozen_idx, frozen_vals.to(out.dtype))

        freq_idx = torch.full(
            (B, T_ctrl, 1), self._freq_idx, device=out.device, dtype=torch.long
        )
        out = out.scatter(2, freq_idx, f0.unsqueeze(-1).to(out.dtype))
        if return_aux:
            # ``raw`` is the pre-limiter trajectory; comparing it to the
            # emitted one says how hard the limiter is binding.
            return out, {"logits": logits, "z": z, "raw_trainable": raw}
        return out
