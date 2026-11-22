"""The parametric glottis, one sample at a time (``crates/chimera/src/sources/glottis.rs``).

It outputs the glottal *area*; the network turns it into flow. Voicing is gated by the smoothed
transglottal pressure and by abduction. Jitter and shimmer come from CHIMERA's own random stream,
drawn once per glottal cycle; drift and vibrato are precomputed per sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from samuel.chimera import rng
from samuel.chimera.network import CM_H2O

GLOTTIS_SEED = 777
DRIFT_SEED = 4242

PARAM_FIELDS = [
    "f0",
    "amplitude",
    "abduction",
    "chink",
    "open_quotient",
    "open_with_abduction",
    "skew",
    "closure_sharpness",
    "effort",
    "jitter",
    "shimmer",
    "vibrato_depth",
    "vibrato_rate",
    "drift",
    "fry",
    "diplophonia",
    "closure",
]


def smoothstep(e0: float, e1: float, x: Tensor) -> Tensor:
    t = torch.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def shape(ph: Tensor, oq: Tensor, skew: Tensor, sharp: Tensor) -> Tensor:
    """Normalised area pulse (0..1) at cycle phase ``ph``: a raised-cosine opening over ``skew``
    of the open phase, then a closing between a soft and an abrupt cosine."""
    t = ph / oq
    s = torch.clamp(skew, 0.3, 0.9)
    rise = 0.5 - 0.5 * torch.cos(math.pi * torch.clamp(t, max=1.0) / s)
    u = (t - s) / (1.0 - s)
    soft = 0.5 + 0.5 * torch.cos(math.pi * u)
    hard = torch.cos(0.5 * math.pi * u)
    fall = soft + (hard - soft) * sharp
    return torch.where(ph >= oq, torch.zeros_like(ph), torch.where(t < s, rise, fall))


def pitch_factors(
    fs: float,
    n: int,
    vibrato_rate: Tensor,
    vibrato_depth: Tensor,
    drift_cents: Tensor,
    seed: int = DRIFT_SEED,
) -> Tensor:
    """The pitch ratio per sample from vibrato and drift, both updated every 32 samples.

    ``vibrato_rate``, ``vibrato_depth``, ``drift_cents``: ``[B, n]`` per sample (the block's values).
    """
    steps = (n + 31) // 32
    dv = torch.tensor(
        rng.drift(seed, fs / 32.0, steps),
        dtype=vibrato_rate.dtype,
        device=vibrato_rate.device,
    )  # [steps]
    at = torch.arange(steps, device=vibrato_rate.device) * 32  # sample of each update
    rate, depth, cents_d = vibrato_rate[:, at], vibrato_depth[:, at], drift_cents[:, at]
    inc = rate * 32.0 / fs * (1.0 + 0.04 * dv)
    vib = torch.remainder(torch.cumsum(inc, dim=-1), 1.0)
    cents = depth * torch.sin(2 * math.pi * vib) + cents_d * dv
    pf = torch.pow(2.0, cents / 1200.0)  # [B, steps]
    return pf.repeat_interleave(32, dim=-1)[:, :n]


@dataclass
class GlottisState:
    phase: Tensor
    cycle_amp: Tensor
    period_scale: Tensor
    cycle_index: Tensor  # long
    voicing_sm: Tensor
    dp_sm: Tensor
    static_sm: Tensor
    closure_sm: Tensor
    amp_sm: Tensor


class Glottis:
    def __init__(
        self,
        fs: float,
        batch: int,
        dtype=torch.float64,
        device=None,
        max_cycles: int = 4096,
        seed: int = GLOTTIS_SEED,
    ):
        self.fs = fs
        z = torch.zeros(batch, dtype=dtype, device=device)
        self.s = GlottisState(
            phase=z,
            cycle_amp=z + 1.0,
            period_scale=z + 1.0,
            cycle_index=torch.zeros(batch, dtype=torch.long, device=device),
            voicing_sm=z,
            dp_sm=z,
            static_sm=z,
            closure_sm=z,
            amp_sm=z,
        )
        self.dp_coef = 1.0 - math.exp(-2 * math.pi * 30.0 / fs)
        self.voice_coef = 1.0 - math.exp(-2 * math.pi * 25.0 / fs)
        self.sm_coef = 1.0 - math.exp(-1.0 / (0.004 * fs))
        self.threshold = 2.5 * CM_H2O
        # per cycle: a gauss for the period, then one for the amplitude
        g = rng.gauss(seed, 2 * max_cycles).reshape(max_cycles, 2)
        self.g_period = torch.tensor(
            np.concatenate([[0.0], g[:, 0]]), dtype=dtype, device=device
        )
        self.g_amp = torch.tensor(
            np.concatenate([[0.0], g[:, 1]]), dtype=dtype, device=device
        )

    def next(self, dp: Tensor, p: dict[str, Tensor], pitch_factor: Tensor) -> Tensor:
        """One sample. ``dp``: transglottal pressure ``[B]``; ``p``: the block's parameters, each
        ``[B]``; ``pitch_factor``: ``[B]``. Returns the glottal area (cm²)."""
        s = self.s
        s.dp_sm = s.dp_sm + self.dp_coef * (dp - s.dp_sm)
        f0 = p["f0"] * pitch_factor
        f0 = f0 * (1.0 - p["fry"]) + torch.clamp(f0, max=48.0) * p["fry"]
        pressure_gate = smoothstep(self.threshold * 0.8, self.threshold * 1.6, s.dp_sm)
        adduct_gate = 1.0 - smoothstep(0.08, 0.22, s.static_sm - p["chink"])
        target = pressure_gate * adduct_gate * (1.0 - s.closure_sm)
        s.voicing_sm = s.voicing_sm + self.voice_coef * (target - s.voicing_sm)

        phase = s.phase + (f0 / self.fs) / s.period_scale
        wrap = phase >= 1.0
        s.phase = torch.where(wrap, phase - 1.0, phase)
        s.cycle_index = s.cycle_index + wrap.long()
        alt = torch.where(s.cycle_index % 2 == 1, 1.0, -1.0).to(phase.dtype)
        gp, ga = self.g_period[s.cycle_index], self.g_amp[s.cycle_index]
        ps = torch.clamp(
            1.0 + p["jitter"] * gp + p["diplophonia"] * 0.08 * alt, 0.55, 1.8
        )
        ca = torch.clamp(
            1.0
            + p["shimmer"] * ga
            - p["diplophonia"] * 0.35 * (alt > 0).to(phase.dtype),
            min=0.12,
        )
        s.period_scale = torch.where(wrap, ps, s.period_scale)
        s.cycle_amp = torch.where(wrap, ca, s.cycle_amp)

        oq = torch.clamp(p["open_quotient"] - 0.18 * (p["effort"] - 0.5), 0.25, 0.98)
        oq = torch.clamp(
            oq
            + p["open_with_abduction"] * torch.clamp(s.static_sm - p["chink"], min=0.0),
            max=0.98,
        )
        oq = oq * (1.0 - p["fry"]) + 0.28 * p["fry"]
        pulse = shape(s.phase, oq, p["skew"], p["closure_sharpness"])
        # (fry > 0.3 adds a second pulse in the Rust; speech never sets fry)
        effort_scale = 0.45 + 0.75 * p["effort"]
        c = self.sm_coef
        s.amp_sm = s.amp_sm + c * (p["amplitude"] * effort_scale - s.amp_sm)
        s.closure_sm = s.closure_sm + c * (p["closure"] - s.closure_sm)
        s.static_sm = s.static_sm + c * (p["abduction"] + p["chink"] - s.static_sm)
        vib = s.amp_sm * s.cycle_amp * s.voicing_sm * pulse
        statics = s.static_sm * (1.0 - 0.92 * s.closure_sm)
        return torch.clamp(vib + statics, min=0.0)


def cycle_state_at(
    fs: float,
    f0: Tensor,
    jitter: Tensor,
    shimmer: Tensor,
    diplophonia: Tensor,
    at: Tensor,
    seed: int = GLOTTIS_SEED,
    max_cycles: int = 4096,
) -> dict[str, Tensor]:
    """The glottis's open-loop cycle state (``phase``, ``period_scale``, ``cycle_amp``,
    ``cycle_index``) at the start of samples ``at`` (``[A]``, ascending), as ``Glottis.next``
    would have left it after running from sample 0. ``f0``: ``[B, n]`` per sample, pitch factor
    and fry applied; ``jitter``, ``shimmer``, ``diplophonia``: ``[B, n]`` per sample.

    The phase never depends on the acoustics, so this walks it a cycle at a time (a
    ``searchsorted`` on the summed increments) rather than a sample at a time. The same sums in
    another order: equal to ``Glottis.next`` to rounding, and differentiable in ``f0`` and the
    jitter. Returns each field as ``[B, A]``."""
    B, n = f0.shape
    dt, dev = f0.dtype, f0.device
    g = rng.gauss(seed, 2 * max_cycles).reshape(max_cycles, 2)
    g_period = torch.tensor(np.concatenate([[0.0], g[:, 0]]), dtype=dt, device=dev)
    g_amp = torch.tensor(np.concatenate([[0.0], g[:, 1]]), dtype=dt, device=dev)
    # C[j] = the increments of samples 0..j-1 summed; the phase after sample j−1 from a cycle that
    # began after sample s (phase φ, scale ps) is φ + (C[j] − C[s]) / ps
    C = torch.cat(
        [torch.zeros(B, 1, dtype=dt, device=dev), torch.cumsum(f0 / fs, -1)], -1
    )
    rows = torch.arange(B, device=dev)
    s = torch.zeros(
        B, dtype=torch.long, device=dev
    )  # samples done when the cycle began
    phi = torch.zeros(B, dtype=dt, device=dev)
    ps = torch.ones(B, dtype=dt, device=dev)
    ca = torch.ones(B, dtype=dt, device=dev)
    idx = torch.zeros(B, dtype=torch.long, device=dev)
    A = at.shape[0]
    out = {
        "phase": torch.zeros(B, A, dtype=dt, device=dev),
        "period_scale": torch.ones(B, A, dtype=dt, device=dev),
        "cycle_amp": torch.ones(B, A, dtype=dt, device=dev),
        "cycle_index": torch.zeros(B, A, dtype=torch.long, device=dev),
    }

    def record(upto: Tensor):
        # chunk starts in [s, upto): this cycle's state, the phase advanced to the start
        hit = (at[None, :] >= s[:, None]) & (at[None, :] < upto[:, None])
        ph = (
            phi[:, None]
            + (C[:, at] - C[rows, torch.clamp(s, max=n)][:, None]) / ps[:, None]
        )
        for key, v in [
            ("phase", ph),
            ("period_scale", ps[:, None].expand(-1, A)),
            ("cycle_amp", ca[:, None].expand(-1, A)),
            ("cycle_index", idx[:, None].expand(-1, A)),
        ]:
            out[key] = torch.where(hit, v, out[key])

    last = at.max() if A else torch.tensor(0, device=dev)
    while bool((s <= last).any()):
        # the first sample m (1-based count j = m + 1) whose phase reaches 1
        target = C[rows, torch.clamp(s, max=n)] + (1.0 - phi) * ps
        j = torch.searchsorted(C, target[:, None], side="left")[:, 0]
        j = torch.clamp(torch.maximum(j, s + 1), max=n + 1)
        record(j)
        if bool((j > n).all()):
            break
        m = torch.clamp(j - 1, max=n - 1)
        phi = torch.where(
            j <= n,
            phi
            + (C[rows, torch.clamp(j, max=n)] - C[rows, torch.clamp(s, max=n)]) / ps
            - 1.0,
            phi,
        )
        idx = idx + (j <= n).long()
        alt = torch.where(idx % 2 == 1, 1.0, -1.0).to(dt)
        jit, shim, dip = jitter[rows, m], shimmer[rows, m], diplophonia[rows, m]
        new_ps = torch.clamp(1.0 + jit * g_period[idx] + dip * 0.08 * alt, 0.55, 1.8)
        new_ca = torch.clamp(
            1.0 + shim * g_amp[idx] - dip * 0.35 * (alt > 0).to(dt), min=0.12
        )
        ps = torch.where(j <= n, new_ps, ps)
        ca = torch.where(j <= n, new_ca, ca)
        s = torch.where(j <= n, j, torch.full_like(j, n + 1))
    return out
