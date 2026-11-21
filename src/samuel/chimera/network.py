"""The branched airway as a staggered-grid FDTD network, one sample at a time.

A line-by-line port of ``crates/chimera/src/airway/network.rs`` (``AirwayNetwork::step``) from the
CHIMERA vocal synth. Nodes carry pressure, branches carry volume velocity; leapfrog in time.
Everything is batched over ``B`` and written with ``torch.where`` instead of branching, so it is
differentiable wherever the Rust is smooth. CGS units throughout (cm, g, s, dyn/cm²).

Where the Rust loops over nodes in order and a node reads a neighbour updated earlier in the same
loop (the squeezed-volume spill, a sealed cell following its open neighbours), the port reads the
neighbour's new value when its index is lower and the old one otherwise, as the Rust does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

RHO = 1.14e-3
C_SOUND = 35000.0
MU = 1.86e-4
NU = MU / RHO
RHO_C2 = RHO * C_SOUND * C_SOUND
CM_H2O = 980.665
RE_CRIT = 1800.0

KIND_SERIES = 0
KIND_PORT = 1


@dataclass
class Topology:
    """The network's fixed structure and constants (``AirwayNetwork`` after ``finalize``)."""

    fs: float
    n_nodes: int
    n_branches: int
    b_a: Tensor  # [NB] long
    b_b: Tensor  # [NB] long
    series: Tensor  # [NB] bool: inertance from the adjacent half-sections only
    port_len: Tensor  # [NB]
    port_kin: Tensor
    port_kex: Tensor
    port_slit: Tensor
    port_half: Tensor  # [NB] bool
    port_extra_r: Tensor
    fixed: Tensor  # [N] bool
    has_wall: Tensor  # [N] bool
    wall_m: Tensor
    wall_r: Tensor
    wall_k: Tensor
    wall_perim: Tensor
    wall_rad: Tensor
    visc_f: Tensor
    rad_nodes: Tensor  # [R] long
    is_junction: Tensor  # [N] bool
    nbr: Tensor  # [N, N] how many branches join node k to node m
    closed_area: float
    min_node_area: float
    squeeze_area: float
    vol_lp_coef: float
    viscous_omega: float
    viscous_scale: float
    noise_scale: float

    @property
    def incidence(self) -> Tensor:
        """[NB, N]: −1 at a branch's start node, +1 at its end (``u @ D`` is the net inflow)."""
        d = torch.zeros(
            self.n_branches,
            self.n_nodes,
            dtype=self.port_len.dtype,
            device=self.port_len.device,
        )
        j = torch.arange(self.n_branches, device=d.device)
        d[j, self.b_a] -= 1.0
        d[j, self.b_b] += 1.0
        return d

    @staticmethod
    def from_dump(net: dict, dtype=torch.float64, device=None) -> Topology:
        """From the ``net`` block of a ``chimera dump`` trace."""

        def f(k):
            return torch.tensor(net[k], dtype=dtype, device=device)

        def b(k):
            return torch.tensor(net[k], dtype=torch.bool, device=device)

        def i(k):
            return torch.tensor(net[k], dtype=torch.long, device=device)

        n = net["n_nodes"]
        nbr = torch.zeros(n, n, dtype=dtype, device=device)
        for k in range(n):
            for e in range(net["nbr_start"][k], net["nbr_start"][k + 1]):
                nbr[k, net["nbr_list"][e]] += 1.0
        return Topology(
            fs=net["fs"],
            n_nodes=n,
            n_branches=net["n_branches"],
            b_a=i("b_a"),
            b_b=i("b_b"),
            series=torch.tensor(
                [k == KIND_SERIES for k in net["b_kind"]], device=device
            ),
            port_len=f("port_len"),
            port_kin=f("port_kin"),
            port_kex=f("port_kex"),
            port_slit=f("port_slit"),
            port_half=b("port_half"),
            port_extra_r=f("port_extra_r"),
            fixed=b("fixed"),
            has_wall=b("has_wall"),
            wall_m=f("wall_m"),
            wall_r=f("wall_r"),
            wall_k=f("wall_k"),
            wall_perim=f("wall_perim"),
            wall_rad=f("wall_rad"),
            visc_f=f("visc_f"),
            rad_nodes=i("rad_nodes"),
            is_junction=b("is_junction"),
            nbr=nbr,
            closed_area=net["closed_area"],
            min_node_area=net["min_node_area"],
            squeeze_area=net["squeeze_area"],
            vol_lp_coef=net["vol_lp_coef"],
            viscous_omega=net["viscous_omega"],
            viscous_scale=net["viscous_scale"],
            noise_scale=net["noise_scale"],
        )


class Network:
    """The network's state for a batch, and its per-sample step.

    Control rate: ``begin_block`` with the next block's targets (areas, sub-section minima,
    section lengths, port areas) and noise sources. Sample rate: set the fixed pressures and the
    glottal port's area, then ``step`` returns the radiated pressure (∝ dU_rad/dt).
    """

    def __init__(
        self,
        topo: Topology,
        batch: int,
        area: Tensor,
        amin: Tensor,
        dx: Tensor,
        port_area: Tensor,
    ):
        self.t = topo
        dt = topo.port_len.dtype
        dev = topo.port_len.device
        n, nb, r = topo.n_nodes, topo.n_branches, len(topo.rad_nodes)
        self.T = 1.0 / topo.fs
        self.inv_T = topo.fs
        self.D = topo.incidence  # [NB, N]
        self.absD = self.D.abs()
        # a sealed node, or a squeezed one spilling its volume, reads its lower-indexed neighbours'
        # values of this sample and its higher-indexed ones' of the last
        self.lower = torch.tril(
            torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=-1
        )  # [k, m]: m < k
        self.visc = math.sqrt(RHO * MU * topo.viscous_omega / 2.0) * topo.viscous_scale

        def z(*shape):
            return torch.zeros(batch, *shape, dtype=dt, device=dev)

        ex = lambda x: x.to(dtype=dt, device=dev).expand(batch, -1).clone()  # noqa: E731
        self.area, self.amin, self.dx = ex(area), ex(amin), ex(dx)
        self.vol = torch.clamp(self.area, min=topo.min_node_area) * self.dx
        self.port_area = ex(port_area)
        self.area_target, self.amin_target, self.dx_target = (
            self.area,
            self.amin,
            self.dx,
        )
        self.area_step = self.amin_step = self.dx_step = z(n)
        self.port_step = z(nb)
        self.p = z(n)
        self.u = z(nb)
        self.mom = z(nb)
        self.l_prev = z(nb)
        self.branch_area = z(nb)
        self.wall_y = z(n)
        self.wall_v = z(n)
        self.vol_src1 = z(n)
        self.vol_src2 = z(n)
        self.rad_ul = z(r)
        self.rad_l_prev = z(r)
        self.rad_prev_total = z()
        # noise sources: up to S per block, active ones first (see ``set_noise``)
        self.noise = None

    # ------------------------------------------------------------------ control rate
    def begin_block(
        self,
        samples: int,
        area_target: Tensor,
        amin_target: Tensor,
        dx_target: Tensor,
        port_target: Tensor,
    ):
        """``set_*_target`` then ``begin_block``: linear ramps to the targets over ``samples``."""
        inv = 1.0 / max(1, samples)
        self.area_target = area_target
        self.amin_target = torch.minimum(
            torch.clamp(amin_target, min=0.0), torch.clamp(area_target, min=0.0)
        )
        self.area_target = torch.clamp(area_target, min=0.0)
        self.dx_target = dx_target
        self.area_step = (self.area_target - self.area) * inv
        self.amin_step = (self.amin_target - self.amin) * inv
        self.dx_step = (dx_target - self.dx) * inv
        port_target = torch.clamp(port_target, min=0.0)
        self.port_step = torch.where(
            self.t.series,
            torch.zeros_like(port_target),
            (port_target - self.port_area) * inv,
        )
        # the Rust zeroes a network fallen silent (|p|, |u| < 1e-9): states that small only
        # matter for speed there, and are left alone here

    def set_noise(
        self,
        src: Tensor,
        fwd: Tensor,
        bwd: Tensor,
        gain: Tensor,
        lp_coef: Tensor,
        hp_coef: Tensor,
        active: Tensor,
        keep_from: Tensor,
    ):
        """The block's turbulence sources, ``[B, S]`` each. ``keep_from[b, i]`` is the slot of the
        last block's sources whose band-filter state source i inherits (same branch and injection
        point), or S for none. ``active`` must be a prefix of the slots, as in the Rust list."""
        B, S = src.shape
        if self.noise is None:
            zero = torch.zeros(B, S, dtype=self.p.dtype, device=self.p.device)
            hp, lp1, lp2 = zero, zero, zero
        else:

            def carry(x):
                xp = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)
                return torch.gather(xp, 1, keep_from)

            hp, lp1, lp2 = (
                carry(self.noise["hp"]),
                carry(self.noise["lp1"]),
                carry(self.noise["lp2"]),
            )
        self.noise = dict(
            src=src,
            fwd=fwd,
            bwd=bwd,
            gain=gain,
            lp_coef=lp_coef,
            hp_coef=hp_coef,
            active=active,
            hp=hp,
            lp1=lp1,
            lp2=lp2,
        )

    # ------------------------------------------------------------------ sample rate
    def set_port_area(self, branch: int, area: Tensor):
        """A port driven per sample (the glottis): its area now, no ramp."""
        a = torch.clamp(area, min=0.0)
        self.port_area = torch.cat(
            [self.port_area[:, :branch], a[:, None], self.port_area[:, branch + 1 :]],
            dim=1,
        )
        self.port_step = torch.cat(
            [
                self.port_step[:, :branch],
                torch.zeros_like(a)[:, None],
                self.port_step[:, branch + 1 :],
            ],
            dim=1,
        )

    def set_fixed_pressure(self, node: int, p: Tensor):
        self.p = torch.cat([self.p[:, :node], p[:, None], self.p[:, node + 1 :]], dim=1)

    def step(self, white: Tensor | None) -> Tensor:
        """Advance one sample. ``white``: ``[B, S]`` uniform bipolar draws for the noise sources
        (only the active ones are used). Returns the radiated pressure ``[B]``."""
        t = self.t
        T, inv_T = self.T, self.inv_T
        min_a = t.min_node_area
        sq = t.squeeze_area
        cl = t.closed_area
        half_rho = 0.5 * RHO

        # 1) geometry ramps and the moving-wall volume source
        area_old, amin_old = self.area, self.amin
        a = torch.clamp(self.area + self.area_step, min=0.0)
        am = torch.clamp(self.amin + self.amin_step, min=0.0)
        amk = torch.minimum(am, a)
        self.area, self.amin = a, amk
        self.dx = self.dx + self.dx_step
        d = self.dx
        ia = 1.0 / torch.clamp(a, min=1e-6)
        inv_amin = 1.0 / torch.clamp(amk, min=1e-6)
        sa = torch.sqrt(a)
        half_l = d * ia
        half_r = t.wall_perim * d * sa * ia * ia * t.visc_f
        v = torch.clamp(a, min=min_a) * d
        raw = -(v - self.vol) * inv_T
        ae = amk
        squeezed = (raw != 0) & (ae < sq)
        tt = torch.clamp(ae / sq, max=1.0)
        keep = torch.where(squeezed, tt * tt * (3.0 - 2.0 * tt), torch.ones_like(tt))
        spill = raw * (1.0 - keep)
        # a neighbour m takes a share if open: its new state when m < k, its last otherwise
        open_new = (torch.minimum(a, amk) >= sq) | (self.amin_target >= sq)
        open_old = (torch.minimum(area_old, amin_old) >= sq) | (self.amin_target >= sq)
        open_km = torch.where(
            self.lower, open_new[:, None, :], open_old[:, None, :]
        ) & (t.nbr > 0)  # [B, k, m]
        w_km = t.nbr * open_km  # [B, k, m]
        n_open = w_km.sum(-1)  # [B, k]
        share = torch.where(
            n_open > 0, spill / torch.clamp(n_open, min=1.0), torch.zeros_like(spill)
        )
        inj = raw * keep + torch.einsum("bk,bkm->bm", share, w_km)
        self.vol = v
        c = t.vol_lp_coef
        self.vol_src1 = self.vol_src1 + c * (inj - self.vol_src1)
        self.vol_src2 = self.vol_src2 + c * (self.vol_src1 - self.vol_src2)
        flow_sum = self.vol_src2

        # 2) turbulence noise → series pressure sources (from the last sample's flows)
        psrc = torch.zeros_like(self.u)
        if self.noise is not None:
            psrc = self._noise(white)

        # 3) branches
        b_a, b_b = t.b_a, t.b_b
        aa, ab = a[:, b_a], a[:, b_b]
        ma, mb = amk[:, b_a], amk[:, b_b]
        iaa, iab = ia[:, b_a], ia[:, b_b]
        imaa, imab = inv_amin[:, b_a], inv_amin[:, b_b]
        hl = half_rho * (half_l[:, b_a] + half_l[:, b_b])
        # series: inertance and viscous loss of the adjacent half-sections, Borda–Carnot jet loss
        s_eff = torch.minimum(ma, mb)
        s_closed = (torch.minimum(aa, ab) < cl) | (s_eff < cl)
        s_l = hl
        s_r = 0.5 * self.visc * (half_r[:, b_a] + half_r[:, b_b])
        ef = torch.clamp(imaa - iab, min=0.0)  # a→b
        eb = torch.clamp(imab - iaa, min=0.0)  # b→a
        s_kf = half_rho * (ef * ef + 0.1 * eb * eb)
        s_kb = half_rho * (eb * eb + 0.1 * ef * ef)
        # ports: their own duct, plus (optionally) the half-sections
        ap = torch.clamp(self.port_area + self.port_step, min=0.0)
        self.port_area = torch.where(t.series, self.port_area, ap)
        p_closed = ap < cl
        ip = 1.0 / torch.where(p_closed, torch.ones_like(ap), ap)
        p_l = RHO * t.port_len * ip + torch.where(t.port_half, hl, torch.zeros_like(hl))
        w = t.port_slit
        p_r = (
            torch.where(
                w > 0,
                12.0 * MU * t.port_len * w * w * ip * ip * ip,
                self.visc
                * 2.0
                * math.sqrt(math.pi)
                * t.port_len
                * torch.sqrt(torch.where(p_closed, torch.ones_like(ap), ap))
                * ip
                * ip,
            )
            + t.port_extra_r
        )
        kin = t.port_kin * half_rho * ip * ip
        p_kf = kin + t.port_kex * half_rho * torch.clamp(ip - iab, min=0.0) ** 2
        p_kb = kin + t.port_kex * half_rho * torch.clamp(ip - iaa, min=0.0) ** 2

        ser = t.series
        closed = torch.where(ser, s_closed, p_closed)
        self.branch_area = torch.where(ser, s_eff, ap)
        ind = torch.where(ser, s_l, p_l)
        r = torch.where(ser, s_r, p_r)
        kf = torch.where(ser, s_kf, p_kf)
        kb = torch.where(ser, s_kb, p_kb)
        # energy-safe momentum when the inertance shrinks (opening): conserve the flow
        m0 = torch.where(ind < self.l_prev, ind * self.u, self.mom)
        p = self.p
        cc = p[:, b_a] - p[:, b_b] + psrc + m0 * inv_T
        bq = ind * inv_T + r
        k = torch.where(cc >= 0, kf, kb)
        # (L/T + R)·U + k·|U|·U = c, solved exactly
        u = 2.0 * cc / (bq + torch.sqrt(bq * bq + 4.0 * k * cc.abs()))
        zero = torch.zeros_like(u)
        u = torch.where(closed, zero, u)
        self.u = u
        self.mom = torch.where(closed, zero, ind * u)
        self.l_prev = torch.where(closed, self.l_prev, ind)
        flow_sum = flow_sum + u @ self.D
        jb = (t.is_junction[b_a] | t.is_junction[b_b]) & ~closed
        inv_lsum = torch.where(jb, 1.0 / ind, zero) @ self.absD

        # 4) walls (velocity at the half step from p(n))
        wm, wr, wk = t.wall_m, t.wall_r, t.wall_k
        has = t.has_wall
        wv = (
            (wm * inv_T - 0.5 * wr) * self.wall_v + p - wk * self.wall_y
        ) / torch.where(has, wm * inv_T + 0.5 * wr, torch.ones_like(wm))
        wv = torch.where(has, wv, torch.zeros_like(wv))
        self.wall_v = wv
        self.wall_y = self.wall_y + T * wv
        sa_w = torch.where(a > min_a, sa, torch.full_like(sa, math.sqrt(min_a)))
        uw = t.wall_perim * sa_w * d * wv
        flow_sum = flow_sum - uw
        wall_rad_sum = (uw * t.wall_rad).sum(-1)

        # 5) nodes; radiation loads semi-implicitly (Crank–Nicolson)
        sealed = (a < cl) | (amk < cl)
        cap = self.vol / RHO_C2
        cap = torch.where(
            t.is_junction, torch.maximum(cap, T * T * inv_lsum / 2.4), cap
        )
        p_new = p + (T / cap) * flow_sum
        rn = t.rad_nodes
        rad_total = wall_rad_sum
        # radiating ends
        a_r = torch.clamp(a[:, rn], min=1e-4)
        lr = 8.0 * RHO / (3.0 * math.pi * torch.sqrt(math.pi * a_r))
        rr = 128.0 * RHO * C_SOUND / (9.0 * math.pi**2 * a_r)
        p_old = p[:, rn]
        lp = self.rad_l_prev
        ul_prev = torch.where((lp > 0) & (lr > lp), self.rad_ul * lp / lr, self.rad_ul)
        h = T / (2.0 * lr)
        g = 0.5 / rr
        c_r = cap[:, rn]
        pr_new = (p_old * (c_r * inv_T - 0.5 * h - g) + flow_sum[:, rn] - ul_prev) / (
            c_r * inv_T + 0.5 * h + g
        )
        ul = ul_prev + h * (p_old + pr_new)
        ur = 0.5 * (ul_prev + ul) + g * (pr_new + p_old)
        r_sealed = sealed[:, rn]
        # a sealed aperture relaxes to the atmosphere and its radiated flow dies away smoothly
        ul_s = self.rad_ul * 0.995
        self.rad_ul = torch.where(r_sealed, ul_s, ul)
        self.rad_l_prev = torch.where(r_sealed, self.rad_l_prev, lr)
        rad_total = rad_total + torch.where(r_sealed, ul_s, ur).sum(-1)
        pr_new = torch.where(r_sealed, p_old * 0.98, pr_new)
        p_new = p_new.index_copy(1, rn, pr_new)
        p_new = torch.where(t.fixed, p, p_new)
        # a sealed interior cell has no air of its own: it follows its open neighbours
        open_n = ~sealed
        p_nb = torch.where(self.lower, p_new[:, None, :], p[:, None, :])  # [B, k, m]
        w_open = t.nbr * open_n[:, None, :]  # [B, k, m]
        cnt = w_open.sum(-1)
        mean = torch.einsum("bkm,bkm->bk", w_open, p_nb) / torch.clamp(cnt, min=1.0)
        p_seal = torch.where(cnt > 0, p + 0.02 * (mean - p), p)
        is_rad = torch.zeros(t.n_nodes, dtype=torch.bool, device=a.device)
        is_rad[rn] = True
        p_new = torch.where(sealed & ~is_rad, p_seal, p_new)
        self.p = torch.where(t.fixed, p, p_new)

        out = (rad_total - self.rad_prev_total) * inv_T
        self.rad_prev_total = rad_total
        return out

    def _noise(self, white: Tensor) -> Tensor:
        """Re²-gated band-filtered noise at each source, injected downstream of its jet."""
        t = self.t
        s = self.noise
        # white noise of variance σ² spreads σ²/fs per Hz: the Rust at this commit normalises the
        # wrong way round, √(96 kHz/fs) (a fix is under review), and the port follows it for now
        scale = t.noise_scale * math.sqrt(96000.0 / t.fs)
        sat = 12000.0**2 - RE_CRIT**2
        area = torch.gather(self.branch_area, 1, s["src"])
        u = torch.gather(self.u, 1, s["src"])
        open_ = area > t.closed_area
        re = (
            u.abs()
            * 2.0
            / (
                NU
                * torch.sqrt(math.pi * torch.where(open_, area, torch.ones_like(area)))
            )
        )
        x = torch.clamp(re * re - RE_CRIT * RE_CRIT, min=0.0)
        amp = torch.where(
            open_ & (re > RE_CRIT),
            s["gain"] * scale * x / (1.0 + x / sat),
            torch.zeros_like(x),
        )
        amp = torch.where(s["active"], amp, torch.zeros_like(amp))
        w = white * 1.732
        act = s["active"]
        hp = torch.where(act, s["hp"] + (1.0 - s["hp_coef"]) * (w - s["hp"]), s["hp"])
        hp_out = w - hp
        lp1 = torch.where(act, s["lp1"] + s["lp_coef"] * (hp_out - s["lp1"]), s["lp1"])
        lp2 = torch.where(act, s["lp2"] + s["lp_coef"] * (lp1 - s["lp2"]), s["lp2"])
        s["hp"], s["lp1"], s["lp2"] = hp, lp1, lp2
        val = amp * lp2
        fwd = u >= 0
        idx = torch.where(fwd, s["fwd"], s["bwd"])
        val = torch.where(fwd, val, -val)
        return torch.zeros_like(self.u).scatter_add(1, idx, val)
