"""1D-CNN controller that turns audio into Pink Trombone parameter trajectories.

The head emits a categorical distribution over ``n_buckets`` evenly spaced
values per trainable parameter. During training a Gumbel-softmax sample
weights the bucket centers (soft by default; one-hot straight-through with
``gumbel_hard``); at eval time the argmax bucket is used.
The ``frequency`` parameter is supplied externally (precomputed pyin).
``intensity`` is trainable and carries the energy contour: the synth output is
never gain-matched, so the model has to produce the level itself (see
``data.target_rms``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from samuel.encoder import SEANetEncoder, SEANetEncoderConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES, SAMPLE_RATE

A4_HZ = 440.0

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


class F0HeadConfig(BaseModel):
    """Causal pitch prediction, supervised by the precomputed pyin track.

    Off by default, in which case ``frequency`` is handed to the model from the
    pyin cache. pyin runs Viterbi over the whole utterance, so a run that wants
    to be streamable end to end has to predict pitch instead of receiving it.

    Bucket centers sit on the equal-tempered grid anchored at A4 = 440 Hz, so a
    bucket is a fixed interval in cents and every center is a nameable pitch.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # 2 -> quarter-tones, 50 cents apart, with every other center an exact
    # semitone. Pitch resolution matters more than the articulator params' and
    # the head is cheap, so this is not tied to ``PinkTromboneControllerConfig.
    # n_buckets``.
    buckets_per_semitone: int = 2
    # The grid is snapped *outward* to cover this range, so no label can fall
    # past the end buckets. Checked against the pyin cache at startup.
    fmin: float = 70.0
    fmax: float = 500.0

    @property
    def cents_per_bucket(self) -> float:
        return 100.0 / self.buckets_per_semitone

    def bucket_steps(self) -> tuple[int, int]:
        """Inclusive grid-step range, in buckets away from A4, covering the range."""
        per_octave = 12 * self.buckets_per_semitone
        k_lo = math.floor(per_octave * math.log2(self.fmin / A4_HZ))
        k_hi = math.ceil(per_octave * math.log2(self.fmax / A4_HZ))
        return k_lo, k_hi

    @property
    def n_buckets(self) -> int:
        k_lo, k_hi = self.bucket_steps()
        return k_hi - k_lo + 1

    def log_centers(self) -> Tensor:
        k_lo, k_hi = self.bucket_steps()
        if k_hi <= k_lo:
            raise ValueError(
                f"f0 range [{self.fmin}, {self.fmax}] Hz spans fewer than two "
                f"buckets at {self.buckets_per_semitone} per semitone"
            )
        steps = torch.arange(k_lo, k_hi + 1, dtype=torch.float32)
        octave = math.log(2.0) / (12 * self.buckets_per_semitone)
        return math.log(A4_HZ) + steps * octave


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
    # Control frames of future input the model may use. Control frame ``t`` is
    # read off encoder frame ``t + lookahead_frames``, so it sees samples up to
    # ``(t + 1 + lookahead_frames) * samples_per_frame``. Synthesis needs one
    # frame beyond that anyway -- the glottis ramps frame ``t`` toward frame
    # ``t + 1`` (see pink_trombone._upsample_params) -- so total algorithmic
    # delay is ``lookahead_frames + 1`` frames, 11.6 ms at the default 0.
    lookahead_frames: int = 0
    f0: F0HeadConfig = Field(default_factory=F0HeadConfig)

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

        self.head = nn.Linear(config.encoder.dimension, n_trainable * config.n_buckets)
        # Bias init at zero -> uniform softmax -> mean bucket value at start.
        with torch.no_grad():
            self.head.bias.zero_()

        if config.f0.enabled:
            dim = config.encoder.dimension
            self.f0_head = nn.Linear(dim, config.f0.n_buckets)
            self.voiced_head = nn.Linear(dim, 1)
            with torch.no_grad():
                self.f0_head.bias.zero_()
                self.voiced_head.bias.zero_()
            self.register_buffer("f0_log_centers", config.f0.log_centers())

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

    def f0_bucket_targets(self, f0_hz: Tensor) -> Tensor:
        """Nearest log-bucket index for each label frequency; ``[B, T]`` long."""
        log_f0 = f0_hz.clamp_min(1e-3).log()
        return (log_f0.unsqueeze(-1) - self.f0_log_centers).abs().argmin(dim=-1)

    def forward(
        self,
        wav: Tensor,
        f0: Tensor | None = None,
        tau: float = 1.0,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Predict Pink Trombone parameter trajectories.

        Args:
            wav: ``[B, 1, S]`` audio at 44.1 kHz.
            f0: ``[B, T_ctrl]`` fundamental frequency in Hz per control frame,
                already interpolated through unvoiced regions and clamped to a
                sane range. Required unless ``config.f0.enabled``, in which case
                the model predicts pitch itself and this argument is ignored
                (it stays the *label*, consumed by the loss, not an input).
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
        predict_f0 = self.config.f0.enabled
        if not predict_f0:
            if f0 is None:
                raise ValueError("f0 is required unless config.f0.enabled")
            if f0.shape != (B, T_ctrl):
                raise ValueError(f"expected f0 [{B}, {T_ctrl}], got {tuple(f0.shape)}")

        hop = self.encoder.hop_length
        k = self.config.lookahead_frames
        # Pad by the lookahead so the encoder emits T_ctrl + k frames, then
        # drop the first k: control frame t becomes encoder frame t + k.
        pad = k * self.samples_per_frame
        pad += (hop - (S + pad) % hop) % hop
        if pad > 0:
            wav = F.pad(wav, (0, pad))

        z = self.encoder(wav)  # [B, dim, T_enc]
        # A no-op when prod(encoder.ratios) == samples_per_frame, which is the
        # default. Otherwise it resamples the latent sequence to control rate,
        # and costs a few ms of lookahead on top of lookahead_frames.
        if z.shape[-1] != T_ctrl + k:
            z = F.interpolate(z, size=T_ctrl + k, mode="linear", align_corners=True)
        if k > 0:
            z = z[..., k:]

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

        out = torch.zeros(
            B, T_ctrl, N_PARAMS, device=wav.device, dtype=constrained.dtype
        )
        train_idx = repeat(self._trainable_idx, "p -> b t p", b=B, t=T_ctrl)
        out = out.scatter(2, train_idx, constrained)

        if self._frozen_idx.numel() > 0:
            frozen_idx = repeat(self._frozen_idx, "p -> b t p", b=B, t=T_ctrl)
            frozen_vals = repeat(self._frozen_vals, "p -> b t p", b=B, t=T_ctrl)
            out = out.scatter(2, frozen_idx, frozen_vals.to(out.dtype))

        aux: dict[str, Tensor] = {"logits": logits, "z": z}
        if predict_f0:
            zt = rearrange(z, "b d t -> b t d")
            f0_logits = self.f0_head(zt).float()  # [B, T_ctrl, K]
            voiced_logits = self.voiced_head(zt).float().squeeze(-1)  # [B, T_ctrl]
            if self.training:
                f0_weights = F.gumbel_softmax(
                    f0_logits, tau=tau, hard=self.config.gumbel_hard, dim=-1
                )
            else:
                f0_weights = F.one_hot(
                    f0_logits.argmax(dim=-1), num_classes=self.config.f0.n_buckets
                ).to(f0_logits.dtype)
            # Mix in the log domain: a soft mixture of log-spaced centers taken
            # in Hz would sit above the perceptual midpoint of its buckets.
            f0 = (f0_weights * self.f0_log_centers).sum(dim=-1).exp()
            aux["f0_logits"] = f0_logits
            aux["voiced_logits"] = voiced_logits
            aux["f0_hz"] = f0
        assert f0 is not None

        freq_idx = torch.full(
            (B, T_ctrl, 1), self._freq_idx, device=out.device, dtype=torch.long
        )
        out = out.scatter(2, freq_idx, f0.unsqueeze(-1).to(out.dtype))
        if return_aux:
            return out, aux
        return out
