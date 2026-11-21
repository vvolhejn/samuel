"""CHIMERA's human voice as a samuel synth: ``chimera(params [B, T, P]) -> audio [B, T·spf]``.

The same call shape as ``samuel.pink_trombone.pink_trombone``. Each frame sets the 17 articulators
(``geometry.ARTIC_NAMES``) and the voice source; frames are ramped linearly like
``pink_trombone._upsample_params`` and the ramp is sampled at each of CHIMERA's control blocks (every
64 output samples). From there on it is the port that matches the Rust engine sample for sample:
articulators → area function (``geometry``) → slew limits, tube targets, noise placement and
glottis settings (``control``) → the airway simulated at 2 × 44.1 kHz (``voice.render``).

What the Rust voice does and this doesn't (the controller is meant to supply it):

- no articulator dynamics: the frames are the positions, not targets the articulators move to (the
  area slew limits, which ease closures into their seal, are kept);
- no f0 glide, no gestural score and no planner;
- no ``enforce_contacts``: the tongue can be placed inside the palate (the area function still
  seals there);
- no output limiter.

The voice starts from the first frame's posture, settled (no transient from a neutral pose); the
lung pressure ramps up from 0 over the first block (1.45 ms). The turbulence noise is drawn
independently per batch item and per source from ``seed``; the glottis's jitter, shimmer and pitch
drift use CHIMERA's fixed streams, the same for every batch item.
"""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from samuel.chimera.airway import HumanAirway, build_airway
from samuel.chimera.control import (
    AirwayControl,
    Source,
    glottis_params,
    lung_pressure,
)
from samuel.chimera.geometry import (
    ARTIC_NAMES,
    ARTIC_RANGES,
    NEUTRAL,
    Geometry,
    VocalTract,
)
from samuel.chimera.voice import VoiceConstants, render

SAMPLE_RATE = 44100
OVERSAMPLE = 2
BLOCK = 64  # output samples per control block
CONTROL_RATE = 12.5

SOURCE_NAMES = [
    "f0",  # Hz
    "pressure",  # lung pressure, cmH₂O
    "abduction",  # the folds' extra opening (cm², adult male), as the score's /h/ and voiceless gestures
    "closure",  # 0..1 holds the folds shut (glottal stop)
    "breathiness",
    "tension",
    "brightness",
    "roughness",
]
PARAM_NAMES = ARTIC_NAMES + SOURCE_NAMES
N_PARAMS = len(PARAM_NAMES)

# The fixed parts of the source (``DEFAULT_SOURCE`` in ``human_voice.rs``).
VIBRATO_DEPTH = 0.0
VIBRATO_RATE = 6.0
DRIFT = 5.0
FRY = 0.0

# (lo, hi, init) per trainable parameter: the articulators' ranges (``ARTIC_SPECS``) and neutral
# posture, the source's ranges as the planner uses them.
_ARTIC = dict(zip(ARTIC_NAMES, zip(*zip(*ARTIC_RANGES), NEUTRAL)))
_FROZEN_ARTIC = [
    "ttCurl",
    "labiodental",
    "epilarynx",
    "ventricular",
    "lateral",
    "dorsum",
]
DEFAULT_PARAM_SPEC: dict[str, tuple[float, float, float]] = {
    **{n: _ARTIC[n] for n in ARTIC_NAMES if n not in _FROZEN_ARTIC},
    "f0": (60.0, 400.0, 110.0),
    "pressure": (0.0, 15.0, 7.2),
    "abduction": (0.0, 0.5, 0.0),
    "closure": (0.0, 1.0, 0.0),
    "breathiness": (0.0, 1.0, 0.15),
    "tension": (0.0, 1.0, 0.5),
    "brightness": (0.0, 1.0, 0.25),
}
DEFAULT_FROZEN_VALUES: dict[str, float] = {
    **{n: _ARTIC[n][2] for n in _FROZEN_ARTIC},
    "roughness": 0.1,
}
assert set(DEFAULT_PARAM_SPEC) | set(DEFAULT_FROZEN_VALUES) == set(PARAM_NAMES)


def default_params(batch: int, frames: int, dtype=torch.float32, device=None) -> Tensor:
    """``[batch, frames, N_PARAMS]`` at every parameter's init: a sustained schwa at 110 Hz."""
    init = {n: v[2] for n, v in DEFAULT_PARAM_SPEC.items()} | DEFAULT_FROZEN_VALUES
    p = torch.tensor([init[n] for n in PARAM_NAMES], dtype=dtype, device=device)
    return p.expand(batch, frames, -1).clone()


def _samples_per_frame(control_rate: float) -> int:
    return round(SAMPLE_RATE / control_rate)


def params_at_blocks(params: Tensor, samples_per_frame: int, n_out: int) -> Tensor:
    """``[B, T, P]`` frames → ``[B, n_blocks, P]``: the linear ramp from frame ``t`` (at sample
    ``t·spf``) to frame ``t + 1`` (the last frame holds), read at each block's first sample."""
    T = params.shape[1]
    s = torch.arange(0, n_out, BLOCK, device=params.device)
    t = torch.div(s, samples_per_frame, rounding_mode="floor")
    frac = (s - t * samples_per_frame).to(params.dtype) / samples_per_frame
    nxt = torch.clamp(t + 1, max=T - 1)
    a, b = params[:, t], params[:, nxt]
    return a + (b - a) * frac[None, :, None]


@dataclasses.dataclass
class Voice:
    """The airway, its control and the voice's constants for one sample rate and body."""

    air: HumanAirway
    k: VoiceConstants
    vt: VocalTract

    @classmethod
    def build(cls, dtype=torch.float64, device=None) -> Voice:
        fs_out = float(SAMPLE_RATE)
        air = build_airway(fs_out * OVERSAMPLE, dtype=dtype, device=device)
        k = VoiceConstants(
            fs_out=fs_out,
            oversample=OVERSAMPLE,
            ctrl_every=BLOCK * OVERSAMPLE,
            output_gain=5.5e-7,
            dc_coef=math.exp(-2.0 * math.pi * 25.0 / fs_out),
            decimator=decimator_taps(OVERSAMPLE, dtype=dtype, device=device),
            lungs=air.lungs,
            lung_port=air.lung_port,
            lung_port_excess=air.lung_port_excess,
            lung_port_coef=air.lung_port_coef,
            glottis_port=air.glottis_port,
            sub_node=air.trachea.last,
            supra_node=air.pharynx.first,
        )
        return cls(air, k, VocalTract(air.body, dtype=dtype, device=device))


def decimator_taps(
    factor: int, taps_per_phase: int = 24, dtype=torch.float64, device=None
):
    """``Decimator::new``: a Kaiser-windowed sinc (β 8.5) cut at 0.46 × the output rate (as a
    fraction of the input rate, 0.46 / ``factor``), normalised to unit DC gain."""

    def i0(x: float) -> float:
        s = term = 1.0
        for k in range(1, 30):
            q = x / (2.0 * k)
            term *= q * q
            s += term
            if term < 1e-12 * s:
                break
        return s

    n = factor * taps_per_phase + 1
    cutoff, beta = 0.46 / factor, 8.5
    mid = (n - 1) / 2.0
    h = []
    for i in range(n):
        x = i - mid
        sinc = (
            2.0 * cutoff
            if x == 0
            else math.sin(2.0 * math.pi * cutoff * x) / (math.pi * x)
        )
        r = 2.0 * i / (n - 1) - 1.0
        h.append(sinc * i0(beta * math.sqrt(max(0.0, 1.0 - r * r))) / i0(beta))
    s = sum(h)
    return torch.tensor([v / s for v in h], dtype=dtype, device=device)


def _geometry_block(g: Geometry, B: int, i: int) -> Geometry:
    """Block ``i`` of a geometry computed for ``[B · n_blocks]`` articulations, as ``[B, ...]``."""

    def pick(v: Tensor) -> Tensor:
        return v.view(B, -1, *v.shape[1:])[:, i]

    return Geometry(
        **{
            f: (
                pick(v) if isinstance(v, Tensor) else {k: pick(x) for k, x in v.items()}
            )
            for f, v in g.__dict__.items()
        }
    )


def render_blocks(
    voice: Voice,
    blocks: Tensor,
    n_out: int,
    start: Tensor | None = None,
    noise: str = "independent",
    generator: torch.Generator | None = None,
    trace: bool = False,
) -> dict[str, Tensor]:
    """Render from ``[B, n_blocks, N_PARAMS]`` per-block parameters.

    ``start``: the ``[B, 17]`` articulation the voice settles in before the first block (default:
    the first block's). ``noise="rust"`` places the noise sources and draws them as the Rust does
    (every batch item needs the same number of active sources per block); ``"independent"``
    keeps all three source slots, faded to gain 0 when off, each with its own noise.
    """
    air, k, vt = voice.air, voice.k, voice.vt
    B, n_blocks, P = blocks.shape
    assert P == N_PARAMS, f"expected {N_PARAMS} parameters, got {P}"
    n_art = len(ARTIC_NAMES)
    artic = blocks[..., :n_art]
    geo = vt.compute(artic.reshape(B * n_blocks, n_art), air.n_ph, air.n_or)
    g0 = (
        _geometry_block(geo, B, 0)
        if start is None
        else vt.compute(start.to(blocks), air.n_ph, air.n_or)
    )
    ac = AirwayControl(air, k.ctrl_every)
    ac.start(g0)
    init = ac.init_state(g0)

    s = {n: blocks[..., n_art + j] for j, n in enumerate(SOURCE_NAMES)}

    def const(v: float) -> Tensor:
        return torch.full_like(s["f0"], v)

    src = Source(
        **s,
        vibrato_depth=const(VIBRATO_DEPTH),
        vibrato_rate=const(VIBRATO_RATE),
        drift=const(DRIFT),
        fry=const(FRY),
    )
    gp = glottis_params(src, air)  # each [B, n_blocks]
    p_lung = lung_pressure(src)

    def block(i, net):
        return ac.block(
            _geometry_block(geo, B, i),
            {key: v[:, i] for key, v in gp.items()},
            p_lung[:, i],
            net,
            exact_noise=noise == "rust",
        )

    glottis_blocks = {
        key: gp[key] for key in ("vibrato_rate", "vibrato_depth", "drift")
    }
    return render(
        air.topo,
        k,
        init,
        block,
        n_out,
        trace=trace,
        glottis_blocks=glottis_blocks,
        noise=noise,
        generator=generator,
    )


_VOICES: dict[tuple, Voice] = {}


def chimera(
    params: Tensor,
    seed: int | None = None,
    control_rate: float = CONTROL_RATE,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Differentiable CHIMERA human voice (adult male).

    Args:
        params: ``[B, T, N_PARAMS]`` at ``control_rate`` Hz, in ``PARAM_NAMES`` order: the 17
            articulators (template coordinates, ``geometry.ARTIC_RANGES``), then ``f0`` (Hz),
            ``pressure`` (lung pressure, cmH₂O; 7.2 is normal speech, 0 silence), ``abduction``
            (0..0.5, extra glottal opening: ~0.36 for voiceless consonants), ``closure`` (0..1,
            glottal stop), ``breathiness``, ``tension``, ``brightness``, ``roughness`` (0..1).
        seed: seeds the turbulence noise; ``None`` draws fresh noise.
        control_rate: frames per second of ``params``.
        dtype: the simulation's precision. Only float64 is checked against the Rust.

    Returns:
        ``[B, T · round(44100 / control_rate)]`` audio at 44.1 kHz, in ``params``'s dtype.
    """
    _, T, P = params.shape
    assert P == N_PARAMS, f"expected {N_PARAMS} parameters, got {P}"
    key = (dtype, params.device)
    if key not in _VOICES:
        _VOICES[key] = Voice.build(dtype=dtype, device=params.device)
    voice = _VOICES[key]
    spf = _samples_per_frame(control_rate)
    n_out = T * spf
    gen = torch.Generator(device=params.device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(seed)
    blocks = params_at_blocks(params.to(dtype), spf, n_out)
    out = render_blocks(voice, blocks, n_out, noise="independent", generator=gen)["out"]
    return out.to(params.dtype)
