"""Control rate: area functions and source settings → the airway's and glottis's block inputs.

``HumanAirway::commit`` and ``update_noise`` (``crates/chimera/src/voice/human_airway.rs``) and
``HumanVoice::apply_source`` / ``control`` (``human_voice.rs``):

- the section areas and sub-section minima are slew-limited against the last block's (closures
  ease into their seal, lips part from a point), then become the network's targets, with the
  tubes stretched where a pose would make sections shorter than a time step allows;
- turbulence sources go at the glottis, at the narrowest open point, and at the teeth when that
  jet strikes them, each faded in by smooth functions of the geometry;
- the source settings become the glottis's parameters and the lung pressure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from samuel.chimera.airway import HumanAirway
from samuel.chimera.geometry import Geometry, smoothstep
from samuel.chimera.network import CM_H2O
from samuel.chimera.voice import MAX_NOISE, Block

TEETH_STROUHAL = 1.0
JET_SPREAD = 0.17
FRICATION_FADE_AREA = 0.9
TEETH_LEVEL_HZ = 5000.0
TEETH_HIGHPASS_HZ = 1200.0
NOISE_GAIN = dict(glottis=3.2e-6, constriction=1.364e-6, teeth=3.0e-6)
SLEW = dict(
    max_area_rate=45.0,
    area_rate_gain=40.0,
    seal_rate=5.0,
    seal_lo=0.03,
    seal_hi=0.4,
    seal_ease=0.03,
    lip_open_rate=7.0,
)


def js_round(x: Tensor) -> Tensor:
    return torch.floor(x + 0.5)


@dataclass
class Source:
    """The voice source's settings per block, each ``[B]`` (``VoiceSource`` plus the score's glottal
    gestures). ``pressure`` is the lung pressure applied (cmH₂O, the breath gate included);
    ``abduction`` the gestures' extra opening of the folds (cm² at body scale 1); ``closure`` 0..1
    holds them shut (glottal stop)."""

    f0: Tensor
    pressure: Tensor
    abduction: Tensor
    closure: Tensor
    breathiness: Tensor
    tension: Tensor
    brightness: Tensor
    roughness: Tensor
    vibrato_depth: Tensor
    vibrato_rate: Tensor
    drift: Tensor
    fry: Tensor


def effort_at(pressure: Tensor) -> Tensor:
    return torch.clamp((pressure.abs() - 2.2) / 10.0, 0.0, 1.0)


def glottis_params(src: Source, air: HumanAirway) -> dict[str, Tensor]:
    """``apply_source`` and the score's part of ``control``, for a voiced source."""
    body = air.body
    k2 = body.scale**2
    br = src.breathiness
    t = src.tension
    bright = src.brightness - 0.25
    by_tension = 0.4 * t + 1.375 * torch.clamp(t - 0.6, min=0.0)
    rough = src.roughness
    base_abduction = 0.1 * br * br * k2
    return dict(
        f0=src.f0,
        amplitude=torch.full_like(t, 0.17 * body.fold_length * body.scale),
        abduction=base_abduction + src.abduction * k2,
        chink=(0.008 + 0.03 * br) * k2,
        open_quotient=0.78 - 0.36 * t + body.open_phase,
        open_with_abduction=torch.full_like(t, 0.98),
        skew=0.56 + 0.16 * t + 0.2 * bright,
        closure_sharpness=torch.clamp(by_tension + 0.8 * bright, 0.0, 1.0),
        effort=effort_at(src.pressure),
        jitter=0.0021 + 0.0245 * rough,
        shimmer=0.02 + 0.22 * rough,
        vibrato_depth=src.vibrato_depth,
        vibrato_rate=src.vibrato_rate,
        drift=src.drift,
        fry=src.fry,
        diplophonia=torch.where(
            rough > 0.5, (rough - 0.5) * 1.4, torch.zeros_like(rough)
        ),
        closure=torch.clamp(src.closure, min=0.0),
    )


class AirwayControl:
    """Turns each block's geometry into the network's targets and noise sources."""

    def __init__(self, air: HumanAirway, ctrl_every: int):
        self.air = air
        self.dt = ctrl_every / air.fs
        t = air.topo
        self.n_ph, self.n_or = air.n_ph, air.n_or
        self.ph_nodes = torch.arange(
            air.pharynx.first, air.pharynx.first + air.n_ph, device=t.b_a.device
        )
        self.or_nodes = torch.arange(
            air.oral.first, air.oral.first + air.n_or, device=t.b_a.device
        )
        self.base: tuple[Tensor, Tensor, Tensor, Tensor] | None = None

    def start(self, g0: Geometry):
        """The areas the voice starts from (the network snapped to them, no slew)."""
        self.base = (g0.ph_areas, g0.or_areas, g0.ph_min, g0.or_min)

    def init_state(self, g0: Geometry) -> dict[str, Tensor]:
        """The network's initial areas, minima, lengths and port areas for geometry ``g0``."""
        air = self.air
        B = g0.ph_len.shape[0]
        area = air.area0.expand(B, -1).clone()
        dx = air.dx0.expand(B, -1).clone()
        port = air.port_area0.expand(B, -1).clone()
        area_t, amin_t, dx_t, port_t = self._targets(
            area,
            area.clone(),
            dx,
            port,
            g0.ph_areas,
            g0.or_areas,
            g0.ph_min,
            g0.or_min,
            g0,
        )
        return dict(area=area_t, amin=amin_t, dx=dx_t, port_area=port_t)

    # ------------------------------------------------------------------ slew limits
    def _slew(self, cur, old, mins, old_mins, lips_from):
        k2 = self.air.body.scale**2
        open_r = SLEW["max_area_rate"] * k2
        gain = SLEW["area_rate_gain"]
        seal_r = SLEW["seal_rate"] * k2
        lo, hi, ease = (
            SLEW["seal_lo"] * k2,
            SLEW["seal_hi"] * k2,
            SLEW["seal_ease"] * k2,
        )
        lip_r = SLEW["lip_open_rate"] * k2
        dt = self.dt

        def close_r(a):
            u = torch.clamp((a - lo) / (hi - lo), 0.0, 1.0)
            eased = seal_r * torch.sqrt(torch.clamp(a, min=0.0) / ease)
            return torch.where(
                a < ease, eased, seal_r + (open_r - seal_r) * u * u * (3.0 - 2.0 * u)
            )

        idx = torch.arange(cur.shape[-1], device=cur.device)

        def open_rate(a):
            lips = lip_r + (open_r - lip_r) * smoothstep(lo, 2.0 * lo, a)
            return torch.where(
                idx[None, :] < lips_from[:, None], torch.full_like(a, open_r), lips
            )

        d = cur - old
        up = torch.minimum(cur, old + (open_rate(old) + gain * cur) * dt)
        down = torch.maximum(cur, old - (close_r(old) + gain * old) * dt)
        cur = torch.where(d > 0, up, down)
        dm = mins - old_mins
        up_m = torch.minimum(mins, old_mins + (open_rate(old_mins) + gain * mins) * dt)
        down_m = torch.maximum(
            mins, old_mins - (close_r(old_mins) + gain * old_mins) * dt
        )
        mins = torch.where(dm > 0, up_m, down_m)
        mins = torch.clamp(torch.minimum(mins, cur), min=0.0)
        return cur, mins

    def _targets(self, area, amin, dx, port, ph, orl, ph_min, or_min, g):
        """``set_tube_targets`` for both tubes and ``set_port_target`` for the velum."""
        air = self.air
        ph_sim = torch.clamp(g.ph_len, min=self.n_ph * air.dx_stable)
        or_sim = torch.clamp(g.or_len, min=self.n_or * air.dx_stable)
        area = area.index_copy(1, self.ph_nodes, torch.clamp(ph, min=0.0)).index_copy(
            1, self.or_nodes, torch.clamp(orl, min=0.0)
        )
        amin = amin.index_copy(
            1, self.ph_nodes, torch.clamp(torch.minimum(ph, ph_min), min=0.0)
        ).index_copy(1, self.or_nodes, torch.clamp(torch.minimum(orl, or_min), min=0.0))
        dx = dx.index_copy(
            1, self.ph_nodes, (ph_sim / self.n_ph)[:, None].expand(-1, self.n_ph)
        ).index_copy(
            1, self.or_nodes, (or_sim / self.n_or)[:, None].expand(-1, self.n_or)
        )
        port = port.index_copy(
            1,
            torch.tensor([air.velum_port], device=port.device),
            torch.clamp(g.velum_port, min=0.0)[:, None],
        )
        return area, amin, dx, port

    # ------------------------------------------------------------------ the block
    def block(
        self,
        g: Geometry,
        src: dict[str, Tensor],
        p_lung: Tensor,
        net,
        exact_noise: bool,
    ) -> Block:
        """Block inputs for geometry ``g`` (one block, ``[B, ...]``) from the network's state."""
        air = self.air
        assert self.base is not None, "call start() first"
        or_len = g.or_len
        lips_from = torch.clamp(
            torch.trunc(self.n_or * (1.0 - (g.lip_s - g.teeth_s) / or_len)), min=0
        ).long()
        big = torch.full_like(lips_from, 1 << 30)
        ph, ph_min = self._slew(g.ph_areas, self.base[0], g.ph_min, self.base[2], big)
        orl, or_min = self._slew(
            g.or_areas, self.base[1], g.or_min, self.base[3], lips_from
        )
        self.base = (ph, orl, ph_min, or_min)
        area_t, amin_t, dx_t, port_t = self._targets(
            net.area_target,
            net.amin_target,
            net.dx_target,
            net.port_area,
            ph,
            orl,
            ph_min,
            or_min,
            g,
        )
        port_t = torch.where(
            air.topo.series,
            port_t,
            torch.where(
                torch.arange(port_t.shape[1], device=port_t.device) == air.glottis_port,
                net.port_area,
                port_t,
            ),
        )
        noise = self._noise(g, ph_min, or_min, net, exact_noise)
        return Block(
            area_target=area_t,
            amin_target=amin_t,
            dx_target=dx_t,
            port_target=port_t,
            p_lung=p_lung,
            glottis=src,
            **noise,
        )

    def _noise(
        self, g: Geometry, ph_min, or_min, net, exact: bool
    ) -> dict[str, Tensor]:
        """``update_noise``: aspiration, the narrowest point's jet, the teeth obstacle source."""
        air = self.air
        fs = air.fs
        k = air.body.scale
        k2 = k * k
        n_ph, n_or = self.n_ph, self.n_or
        B = ph_min.shape[0]
        dev = ph_min.device
        L = lambda x: torch.full((B,), int(x), dtype=torch.long, device=dev)  # noqa: E731
        asp = (
            L(air.glottis_port),
            L(air.branch_of(air.pharynx, 1)),
            L(air.branch_of(air.trachea, air.trachea.n - 2)),
        )
        mins = torch.cat([ph_min, or_min], dim=1)
        open_ = mins > 0
        masked = torch.where(open_, mins, torch.full_like(mins, math.inf))
        min_a, min_i = masked.min(-1)
        any_open = open_.any(-1)
        min_a = torch.where(any_open, min_a, torch.zeros_like(min_a))
        w_area = torch.where(
            any_open,
            1.0 - smoothstep(0.35 * k2, FRICATION_FADE_AREA * k2, min_a),
            torch.zeros_like(min_a),
        )
        n_tot = n_ph + n_or
        dx_ph = g.ph_len / n_ph
        dx_or = g.or_len / n_or
        in_ph = min_i < n_ph
        dxm = torch.where(in_ph, dx_ph, dx_or)
        off = torch.clamp(js_round(0.9 * k / dxm), min=1.0).long()
        ph_b0, or_b0 = air.pharynx.branch_base, air.oral.branch_base

        def to_branch(gi):
            return torch.where(
                gi < n_ph,
                ph_b0 + torch.clamp(gi, 0, n_ph - 2),
                or_b0 + torch.clamp(gi - n_ph, 0, n_or - 2),
            )

        src_i = torch.clamp(min_i, max=n_tot - 2)
        fwd_i = torch.clamp(min_i + off, max=n_tot - 2)
        bwd_i = torch.clamp(min_i - off, min=0)
        # the junction between the tubes (the oral tube's first node) is no branch: the Rust
        # registers no constriction source (nor teeth source) when one of its ends lands there
        no_junction = (src_i != n_ph) & (fwd_i != n_ph) & (bwd_i != n_ph)
        src, fwd, bwd = to_branch(src_i), to_branch(fwd_i), to_branch(bwd_i)
        con_gain = NOISE_GAIN["constriction"] * w_area
        s_min = torch.where(
            in_ph, (min_i + 0.5) * dx_ph, g.ph_len + (min_i - n_ph + 0.5) * dx_or
        )
        behind = (g.teeth_s - s_min) / k
        w_dist = smoothstep(1.0, 1.5, behind) * (1.0 - smoothstep(2.6, 4.0, behind))
        w_jet = 1.0 - smoothstep(0.2 * k2, 0.5 * k2, min_a)
        wt = w_dist * w_jet
        teeth_i = (
            n_ph
            + torch.clamp(
                js_round((g.teeth_s - g.ph_len) / dx_or), 0.0, n_or - 2
            ).long()
        )
        tb = to_branch(teeth_i)
        a_jet = torch.clamp(
            torch.gather(net.branch_area, 1, src[:, None])[:, 0], min=1e-6
        )
        v_jet = torch.gather(net.u, 1, src[:, None])[:, 0].abs() / a_jet
        d_jet = 2.0 * torch.sqrt(a_jet / math.pi)
        eddy = torch.maximum(d_jet, JET_SPREAD * (g.teeth_s - s_min))
        top = TEETH_STROUHAL * (v_jet * d_jet / (eddy * eddy))
        spread = torch.sqrt(TEETH_LEVEL_HZ / torch.clamp(top, min=TEETH_HIGHPASS_HZ))
        teeth_gain = NOISE_GAIN["teeth"] * wt * spread
        con_on = (w_area > 0) & no_junction
        teeth_on = con_on & (wt > 1e-3) & (teeth_i != n_ph)
        if exact:
            # the Rust's list: a source only where it is on, in order (active slots are a prefix)
            con_act, teeth_act = con_on, teeth_on
        else:
            # every slot always present, faded by its weight (gain 0 when off)
            con_act = teeth_act = torch.ones_like(con_on)
            con_gain = torch.where(con_on, con_gain, torch.zeros_like(con_gain))
            teeth_gain = torch.where(teeth_on, teeth_gain, torch.zeros_like(teeth_gain))
        lpc = lambda hz: 1.0 - torch.exp(-2 * math.pi * hz / fs)  # noqa: E731
        hpc = lambda hz: torch.exp(-2 * math.pi * hz / fs)  # noqa: E731
        T = lambda x: torch.full((B,), float(x), dtype=ph_min.dtype, device=dev)  # noqa: E731
        stack = lambda *xs: torch.stack(xs, dim=1)  # noqa: E731
        assert MAX_NOISE == 3
        return dict(
            noise_src=stack(asp[0], src, src),
            noise_fwd=stack(asp[1], fwd, tb),
            noise_bwd=stack(asp[2], bwd, bwd),
            noise_gain=stack(T(NOISE_GAIN["glottis"]), con_gain, teeth_gain),
            noise_lp=stack(lpc(T(3500.0)), lpc(T(2600.0)), lpc(top)),
            noise_hp=stack(hpc(T(300.0)), hpc(T(350.0)), hpc(T(TEETH_HIGHPASS_HZ))),
            noise_active=stack(torch.ones_like(con_on), con_act, teeth_act),
        )


def lung_pressure(src: Source) -> Tensor:
    return src.pressure * CM_H2O


# ---------------------------------------------------------------------- replay from articulators
DEFAULT_SOURCE = dict(
    breathiness=0.15, tension=0.5, brightness=0.25, roughness=0.1, fry=0.0
)


def geometry_at(g: Geometry, i: int) -> Geometry:
    """Block ``i`` of a ``[n_blocks, ...]`` geometry, kept as a batch of one."""
    return Geometry(
        **{
            f: (
                v[i : i + 1]
                if isinstance(v, Tensor)
                else {k: x[i : i + 1] for k, x in v.items()}
            )
            for f, v in g.__dict__.items()
        }
    )


def source_from_trace(doc: dict, dtype=torch.float64, device=None) -> list[Source]:
    """Each block's source settings, recovered from a ``chimera dump`` trace (male body, the
    default source, a score's abduction and closure gestures on top)."""
    T = lambda x: torch.tensor([x], dtype=dtype, device=device)  # noqa: E731
    out = []
    for c in doc["ctrl"]:
        g = c["glottis"]
        out.append(
            Source(
                f0=T(g["f0"]),
                pressure=T(c["p_lung"] / CM_H2O),
                abduction=T(g["abduction"] - 0.1 * DEFAULT_SOURCE["breathiness"] ** 2),
                closure=T(g["closure"]),
                vibrato_depth=T(g["vibrato_depth"]),
                vibrato_rate=T(g["vibrato_rate"]),
                drift=T(g["drift"]),
                **{k: T(v) for k, v in DEFAULT_SOURCE.items()},
            )
        )
    return out


def replay_articulated(
    doc: dict,
    seconds: float | None = None,
    trace: bool = True,
    record: list | None = None,
    dtype=torch.float64,
    device=None,
) -> dict[str, Tensor]:
    """Replay a ``chimera dump`` trace from its articulators and source settings alone: the
    geometry, slew limits, tube targets, noise placement and glottis mapping are this port's.
    ``record`` collects each block's inputs, to compare with the trace's."""
    import dataclasses

    from samuel.chimera.airway import build_airway
    from samuel.chimera.geometry import NEUTRAL_START, VocalTract
    from samuel.chimera.voice import from_trace, render

    topo, k, init_ref, _ = from_trace(doc, dtype=dtype, device=device)
    air = dataclasses.replace(build_airway(k.fs, dtype=dtype, device=device), topo=topo)
    n_out = doc["header"]["n_out"]
    if seconds is not None:
        n_out = min(n_out, int(seconds * k.fs_out))
    n_blocks = (n_out * k.oversample + k.ctrl_every - 1) // k.ctrl_every
    ctrl = doc["ctrl"][:n_blocks]
    vt = VocalTract(air.body, dtype=dtype, device=device)
    artic = torch.tensor([c["artic"] for c in ctrl], dtype=dtype, device=device)
    geo = vt.compute(artic, air.n_ph, air.n_or)
    g0 = vt.compute(
        torch.tensor([NEUTRAL_START], dtype=dtype, device=device), air.n_ph, air.n_or
    )
    ac = AirwayControl(air, k.ctrl_every)
    ac.start(g0)
    init = ac.init_state(g0)
    srcs = source_from_trace({"ctrl": ctrl}, dtype=dtype, device=device)
    gps = [glottis_params(s, air) for s in srcs]

    def block(i, net):
        b = ac.block(
            geometry_at(geo, i), gps[i], lung_pressure(srcs[i]), net, exact_noise=True
        )
        if record is not None:
            record.append(b)
        return b

    glottis_blocks = {
        key: torch.cat([gp[key] for gp in gps])[None]
        for key in ("vibrato_rate", "vibrato_depth", "drift")
    }
    res = render(
        topo, k, init, block, n_out, trace=trace, glottis_blocks=glottis_blocks
    )
    res["init"] = init
    res["init_ref"] = init_ref
    return res
