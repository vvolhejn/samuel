"""The human vocal tract's geometry: articulators → area function, batched.

A literal port of ``HumanVocalTract::compute`` and ``resample`` (``crates/chimera/src/anatomy/
human.rs`` and ``geom.rs``). A midsagittal head: fixed outer wall (pharyngeal wall, velum, palate,
upper lip), moving inner wall (epiglottis, tongue root, a circular tongue body, blade, tip, floor,
lower incisors and lip, carried by the rotating jaw). 49 rays (12 pharyngeal horizontals, a fan
of 31 through the mouth, 6 verticals at the teeth and lips) measure the gap between the walls; a
contact model takes over near the palate, the gap becomes an area by a power law per region, and
the area function is resampled into the pharynx and oral tubes (harmonic mean per section).

Every articulation is computed at once (any leading shape). Polylines whose point count depends on
the posture in the Rust (where the pharyngeal wall is cut, where the tongue's back leaves the
epiglottis) are padded with repeats of a neighbouring point: a zero-length segment is never
crossed and never the nearest, so it changes nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from samuel.chimera.airway import BODY_MALE, Body

DEG = math.pi / 180.0
ARTIC_NAMES = [
    "jaw",
    "tbX",
    "tbY",
    "ttX",
    "ttY",
    "ttCurl",
    "root",
    "lipAp",
    "lipProt",
    "lipWidth",
    "labiodental",
    "velum",
    "larynx",
    "epilarynx",
    "ventricular",
    "lateral",
    "dorsum",
]
N_ARTIC = 17
# neutral (schwa-like) articulation, template coordinates
NEUTRAL = [
    0.6,
    2.9,
    6.85,
    6.45,
    7.3,
    0.0,
    0.0,
    0.7,
    0.2,
    0.5,
    0.0,
    -0.15,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]
# (min, max) per articulator (``ARTIC_SPECS``)
ARTIC_RANGES = [
    (-0.1, 2.6),
    (1.0, 4.8),
    (4.8, 9.2),
    (3.5, 8.0),
    (5.0, 10.8),
    (0.0, 1.0),
    (-1.0, 1.0),
    (-0.4, 2.4),
    (-0.2, 1.5),
    (0.0, 1.0),
    (0.0, 1.0),
    (-0.25, 1.4),
    (-1.6, 1.6),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
]

LAB_HANDOVER = 0.5
LAB_GAP = 0.3
LAB_FRONT_OPENS = 0.7
LAB_FRONT_RAMP = 0.1
ARC_POINTS = 32
N_PH_RAYS = 12
N_FAN = 31
N_FRONT = 6
N_RAYS = N_PH_RAYS + N_FAN + N_FRONT
SPLIT = N_PH_RAYS + 7  # the first fan ray at or under 150°
TEETH = N_PH_RAYS + N_FAN
AREA_LAW = dict(pharynx=(2.0, 1.4), oral=(2.6, 1.4), front=(2.0, 1.4), ahead=(1.6, 1.4))


# ---------------------------------------------------------------------- small helpers
def smoothstep(e0, e1, x: Tensor) -> Tensor:
    t = torch.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def vec(x, y) -> Tensor:
    return torch.stack(torch.broadcast_tensors(x, y), dim=-1)


def dist(a: Tensor, b: Tensor) -> Tensor:
    return torch.linalg.vector_norm(a - b, dim=-1)


def safe_div(a: Tensor, b: Tensor, ok: Tensor) -> Tensor:
    return a / torch.where(ok, b, torch.ones_like(b))


def bezier3(p0, p1, p2, p3, count: int) -> Tensor:
    t = torch.linspace(0.0, 1.0, count, dtype=p0.dtype, device=p0.device)[:, None]
    u = 1.0 - t
    return (
        (u**3) * p0[..., None, :]
        + (3 * u * u * t) * p1[..., None, :]
        + (3 * u * t * t) * p2[..., None, :]
        + (t**3) * p3[..., None, :]
    )


def bezier2(p0, p1, p2, count: int) -> Tensor:
    t = torch.linspace(0.0, 1.0, count, dtype=p0.dtype, device=p0.device)[:, None]
    u = 1.0 - t
    return (
        (u * u) * p0[..., None, :]
        + (2 * u * t) * p1[..., None, :]
        + (t * t) * p2[..., None, :]
    )


def catmull_rom(pts: list[tuple[float, float]], per: int) -> list[tuple[float, float]]:
    out = []
    n = len(pts)
    for i in range(n - 1):
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n - 1)]
        for k in range(per):
            t = k / per
            t2, t3 = t * t, t * t * t
            out.append(
                tuple(
                    0.5
                    * (
                        2 * p1[c]
                        + (-p0[c] + p2[c]) * t
                        + (2 * p0[c] - 5 * p1[c] + 4 * p2[c] - p3[c]) * t2
                        + (-p0[c] + 3 * p1[c] - 3 * p2[c] + p3[c]) * t3
                    )
                    for c in range(2)
                )
            )
    out.append(pts[-1])
    return out


def crossings(o: Tensor, d: Tensor, pl: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Rays ``o + t·d`` ``[M, R, 2]`` against a polyline ``[M, P, 2]``: for each ray and segment, the
    crossing's t, whether it crosses, and whether the ray leaves the tissue (the polyline's right).
    ``[M, R, P−1]`` each. A point on the line counts as its positive side, as in the Rust."""
    a, b = pl[:, None, :-1, :], pl[:, None, 1:, :]
    ox, oy = o[..., 0:1], o[..., 1:2]
    dx, dy = d[..., 0:1], d[..., 1:2]
    sa = dx * (a[..., 1] - oy) - dy * (a[..., 0] - ox) >= 0
    sb = dx * (b[..., 1] - oy) - dy * (b[..., 0] - ox) >= 0
    ex, ey = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    den = dx * ey - dy * ex
    ok = (sa != sb) & (den.abs() >= 1e-15)
    t = safe_div((a[..., 0] - ox) * ey - (a[..., 1] - oy) * ex, den, ok)
    return t, ok, den < 0


def project_on(pl: Tensor, cum: Tensor, p: Tensor) -> dict[str, Tensor]:
    """Nearest point of a polyline ``[M, P, 2]`` to points ``[M, Q, 2]``: arc position, distance,
    signed distance (positive on the airway side), segment."""
    a, b = pl[:, None, :-1, :], pl[:, None, 1:, :]
    e = b - a
    q = p[:, :, None, :]
    l2 = (e * e).sum(-1).expand(-1, q.shape[1], -1)
    valid = l2 >= 1e-12
    t = torch.clamp(safe_div(((q - a) * e).sum(-1), l2, valid), 0.0, 1.0)
    near = a + e * t[..., None]
    dd = torch.linalg.vector_norm(q - near + 0.0, dim=-1)
    dd_safe = torch.where(valid, dd, torch.full_like(dd, math.inf))
    seg = dd_safe.argmin(-1)  # first of equals, as the Rust's strict <
    g = lambda x: torch.gather(x, -1, seg[..., None])[..., 0]  # noqa: E731
    best = g(dd_safe)
    s = torch.gather(cum[:, None, :-1].expand(-1, p.shape[1], -1), -1, seg[..., None])[
        ..., 0
    ] + g(t) * torch.sqrt(g(torch.where(valid, l2, torch.ones_like(l2))))
    cross = e[..., 0] * (q[..., 1] - a[..., 1]) - e[..., 1] * (q[..., 0] - a[..., 0])
    sign = torch.where(g(cross) < 0, 1.0, -1.0).to(p.dtype)
    return dict(s=s, dist=best, signed=sign * best, seg=seg)


def point_along(pl: Tensor, cum: Tensor, s: Tensor) -> Tensor:
    """The point at arc length ``s`` ``[M, Q]`` along a polyline ``[M, P, 2]``."""
    P = pl.shape[1]
    i = (cum[:, None, 1 : P - 1] < s[..., None]).sum(
        -1
    )  # the Rust's walk, cum being monotone
    i = torch.clamp(i, max=P - 2)
    c0 = torch.gather(cum, 1, i)
    c1 = torch.gather(cum, 1, i + 1)
    l = c1 - c0  # noqa: E741
    ok = l > 1e-9
    u = torch.where(
        ok, torch.clamp(safe_div(s - c0, l, ok), 0.0, 1.0), torch.zeros_like(s)
    )
    idx = i[..., None].expand(-1, -1, 2)
    a = torch.gather(pl, 1, idx)
    b = torch.gather(pl, 1, idx + 1)
    return a + (b - a) * u[..., None]


def signed_gap(pl: Tensor, p: Tensor) -> Tensor:
    """Distance from points ``[M, Q, 2]`` to a closed polygon ``[M, P, 2]``, negative inside."""
    a, b = pl[:, None, :-1, :], pl[:, None, 1:, :]
    e = b - a
    l2 = (e * e).sum(-1)
    valid = l2 > 1e-12
    q = p[:, :, None, :]
    t = torch.where(
        valid,
        torch.clamp(safe_div(((q - a) * e).sum(-1), l2, valid), 0.0, 1.0),
        torch.zeros_like(l2),
    )
    dv = q - a - e * t[..., None]
    d2 = (dv * dv).sum(-1)
    best = d2.min(-1).values
    py, px = q[..., 1], q[..., 0]
    straddle = (a[..., 1] > py) != (b[..., 1] > py)
    xcross = a[..., 0] + safe_div((py - a[..., 1]) * e[..., 0], e[..., 1], straddle)
    inside = ((straddle & (px < xcross)).sum(-1) % 2) == 1
    dd = torch.sqrt(best)
    return torch.where(inside, -dd, dd)


def cumlen(pl: Tensor) -> Tensor:
    seg = torch.linalg.vector_norm(pl[:, 1:] - pl[:, :-1], dim=-1)
    return torch.cat([torch.zeros_like(seg[:, :1]), torch.cumsum(seg, dim=1)], dim=1)


# ---------------------------------------------------------------------- the tract
@dataclass
class Geometry:
    """A batch of articulations' area functions (leading shape ``[M]``)."""

    ph_areas: Tensor  # [M, n_ph]
    or_areas: Tensor  # [M, n_or]
    ph_min: Tensor
    or_min: Tensor
    ph_len: Tensor  # [M]
    or_len: Tensor
    teeth_s: Tensor
    lip_s: Tensor
    velum_port: Tensor
    contact_tip: Tensor
    contact_body: Tensor
    contact_tip_n: Tensor  # [M, 2]
    contact_body_n: Tensor
    grid: dict[str, Tensor]  # d, area, s, inner, outer, mid: [M, 49(, 2)]


class VocalTract:
    def __init__(self, body: Body = BODY_MALE, dtype=torch.float64, device=None):
        self.body = body
        self.k = k = body.scale
        self.dtype, self.device = dtype, device
        P = self.pf
        T = lambda pts: torch.tensor(pts, dtype=dtype, device=device)  # noqa: E731
        self.phar_wall = T(
            catmull_rom(
                [
                    P(-1.0, -1.5),
                    P(-1.1, 1.5),
                    P(-1.2, 3.6),
                    P(-1.3, 5.6),
                    P(-1.38, 7.5),
                    P(-1.33, 8.9),
                    P(-1.05, 10.1),
                    P(-0.45, 10.9),
                    P(0.4, 11.3),
                ],
                4,
            )
        )
        self.pns = T(P(1.0, 9.75))
        self.palate = T(
            catmull_rom(
                [
                    P(1.0, 9.6),
                    P(1.8, 9.92),
                    P(2.7, 10.18),
                    P(3.6, 10.3),
                    P(4.4, 10.25),
                    P(5.1, 10.05),
                    P(5.7, 9.72),
                    P(6.15, 9.36),
                    P(6.5, 9.05),
                    P(6.78, 8.7),
                    P(6.98, 8.3),
                    P(7.12, 7.95),
                ],
                3,
            )
        )
        self.condyle = T(P(-2.5, 10.5))
        self.polar_c = T(P(2.6, 6.95))
        self.polar_c2 = T(P(5.0, 6.85))
        _ = k

    def pf(self, x: float, y: float) -> tuple[float, float]:
        """Template point → body coordinates (floats)."""
        ph = self.body.pharynx
        xx = 0.5 + (x - 0.5) * 0.86 if x > 0.5 else x
        yy = 7.0 - (7.0 - y) * ph if y < 7.0 else y
        return (xx * self.k, yy * self.k)

    def p(self, x: Tensor, y: Tensor) -> Tensor:
        """Template point → body coordinates (tensors)."""
        ph = self.body.pharynx
        xx = torch.where(x > 0.5, 0.5 + (x - 0.5) * 0.86, x)
        yy = torch.where(y < 7.0, 7.0 - (7.0 - y) * ph, y)
        return vec(xx * self.k, yy * self.k)

    def pt(self, x: float, y: float, like: Tensor) -> Tensor:
        return torch.tensor(self.pf(x, y), dtype=like.dtype, device=like.device).expand(
            *like.shape[:-1], 2
        )

    def compute(self, artic: Tensor, n_ph: int, n_or: int) -> Geometry:
        """``artic``: ``[..., 17]`` in ``ARTIC_NAMES`` order."""
        lead = artic.shape[:-1]
        a = artic.reshape(-1, N_ARTIC).to(self.dtype)
        g = self._compute(a, n_ph, n_or)
        return Geometry(
            **{
                f: (
                    v.reshape(*lead, *v.shape[1:])
                    if isinstance(v, Tensor)
                    else {kk: vv.reshape(*lead, *vv.shape[1:]) for kk, vv in v.items()}
                )
                for f, v in g.__dict__.items()
            }
        )

    def _compute(self, a: Tensor, n_ph: int, n_or: int) -> Geometry:
        k = self.k
        M = a.shape[0]
        dt, dev = a.dtype, a.device
        one = torch.ones(M, dtype=dt, device=dev)
        A = {n: a[:, i] for i, n in enumerate(ARTIC_NAMES)}
        two = torch.zeros(M, 2, dtype=dt, device=dev)
        P = lambda x, y: self.pt(x, y, two)  # noqa: E731
        lar = A["larynx"] * k
        up = lambda y: vec(torch.zeros_like(y), y)  # noqa: E731
        glottis = P(0.0, 0.0) + up(lar)

        # ---- jaw: rotation about the condyle
        jaw_r = dist(
            self.condyle, torch.tensor(self.pf(6.95, 7.75), dtype=dt, device=dev)
        ).item()
        phi = -torch.clamp(A["jaw"], -0.2, 3.0) * k / jaw_r
        cphi, sphi = torch.cos(phi)[:, None], torch.sin(phi)[:, None]

        def j(x, y):
            d = P(x, y) - self.condyle
            return self.condyle + vec(
                d[:, 0] * cphi[:, 0] - d[:, 1] * sphi[:, 0],
                d[:, 0] * sphi[:, 0] + d[:, 1] * cphi[:, 0],
            )

        # ---- velum flap from the PNS
        v_open = torch.clamp(A["velum"] / 1.2, 0.0, 1.0)
        v_ang = (194.0 + 46.0 * v_open) * DEG
        v_len = 2.8 * k
        s = torch.linspace(0.0, 1.0, 11, dtype=dt, device=dev)
        sag = torch.sin(s * math.pi) * k * (0.3 * v_open[:, None] - 0.12)
        c = (
            self.pns
            + v_len
            * s[None, :, None]
            * vec(torch.cos(v_ang), torch.sin(v_ang))[:, None, :]
            - up(sag)
        )
        nrm = vec(-torch.sin(v_ang), torch.cos(v_ang))[:, None, :]
        th = ((0.48 - 0.2 * s) * k)[None, :, None]
        vel_oral = c + nrm * th
        u = s / 0.35
        wpal = torch.where(u < 1.0, (1.0 - u) * (1.0 - u), torch.zeros_like(u))[
            None, :, None
        ]
        vel_oral = lerp(vel_oral, self.palate[0], wpal)

        # ---- outer contour: pharyngeal wall up to under the uvula, velum, palate, upper lip
        uv_y = vel_oral[:, -1, 1] - 0.3 * k
        W = self.phar_wall.shape[0]
        wall = self.phar_wall.expand(M, -1, -1)
        above = wall[:, :, 1] >= uv_y[:, None]
        f = torch.where(
            above.any(1), above.float().argmax(1), torch.full((M,), W, device=dev)
        )  # first point at or over
        fi = torch.clamp(f, 1, W - 1)
        wa = torch.gather(wall, 1, (fi - 1)[:, None, None].expand(-1, -1, 2))[:, 0]
        wb = torch.gather(wall, 1, fi[:, None, None].expand(-1, -1, 2))[:, 0]
        tcut = torch.clamp(
            (uv_y - wa[:, 1]) / torch.clamp(wb[:, 1] - wa[:, 1], min=1e-9), 0.0, 1.0
        )
        cut = lerp(wa, wb, tcut[:, None])
        idx = torch.arange(W, device=dev)[None, :]
        fill = torch.where(
            (f > 0)[:, None], cut, vel_oral[:, -1]
        )  # no wall at all: repeat the uvula's tip
        wall_part = torch.where((idx < f[:, None])[..., None], wall, fill[:, None, :])
        roof = torch.cat([vel_oral.flip(1), self.palate[1:].expand(M, -1, -1)], dim=1)
        prot = torch.clamp(A["lipProt"], -0.2, 1.6) * k
        spread_w = torch.clamp(A["lipWidth"], 0.0, 1.0)
        y_mid = (7.85 - 0.3 * A["jaw"]) * k
        lab = torch.clamp(A["labiodental"], 0.0, 1.0)
        ap = A["lipAp"] * k
        lip_front_x = (
            0.5 + (8.15 + 0.12 * (1.0 - spread_w) - 0.5) * 0.86
        ) * k + prot * 0.95
        ul_edge = vec(lip_front_x - 0.15 * k, y_mid + ap * 0.35 + lab * 0.35 * k)
        ll_edge = vec(lip_front_x - 0.25 * k, y_mid - ap * 0.65)
        incisor = self.palate[-1]
        target = vec(
            incisor[0] + 0.05 * k + 0 * lab,
            incisor[1] - torch.clamp(A["lipAp"] * k, min=-0.1),
        )
        ll_edge = lerp(ll_edge, target, lab[:, None])
        ul_edge = lerp(
            ul_edge,
            (incisor + torch.tensor([0.55 * k, 0.45 * k], dtype=dt, device=dev)).expand(
                M, 2
            ),
            (lab * 0.7)[:, None],
        )
        ul_base = P(7.32, 8.35)
        upper_lip_in = bezier3(
            ul_base,
            ul_base + vec(0.3 * k + prot * 0.4, 0.05 * k * one),
            ul_edge + torch.tensor([-0.35 * k, 0.12 * k], dtype=dt, device=dev),
            ul_edge,
            7,
        )
        outer = torch.cat([wall_part, roof, upper_lip_in], dim=1)

        # ---- tongue
        r = self.body.tongue_radius * k
        b = self.p(A["tbX"], A["tbY"])
        b0 = P(NEUTRAL[1], NEUTRAL[2])
        vallecula = P(0.75, 3.3) + vec(
            0.45 * (b[:, 0] - b0[:, 0]) + A["root"] * k,
            0.3 * (b[:, 1] - b0[:, 1]) + 0.55 * lar,
        )
        epi_base = P(0.3, 2.4) + up(lar)
        epi_tip = P(0.0, 4.5) + vec(
            0.35 * (vallecula[:, 0] - 0.75 * k), lar * 0.8 + 0.15 * (b[:, 1] - b0[:, 1])
        )
        th_b = 200.0 * DEG
        p_b = b + r * torch.tensor(
            [math.cos(th_b), math.sin(th_b)], dtype=dt, device=dev
        )
        tan_b = torch.tensor([math.sin(th_b), -math.cos(th_b)], dtype=dt, device=dev)
        h_r = dist(vallecula, p_b) / 3.0
        root = bezier3(
            vallecula,
            vallecula + vec(0.08 * k * one, h_r),
            p_b - tan_b * h_r[:, None],
            p_b,
            9,
        )
        tip = self.p(A["ttX"], A["ttY"])
        d_t = dist(tip, b)
        tip = torch.where(
            (d_t < r * 1.05)[:, None],
            b + (tip - b) / torch.clamp(d_t, min=1e-12)[:, None] * (r * 1.05),
            tip,
        )
        dtb = tip - b
        d = torch.linalg.vector_norm(dtb, dim=-1)
        base = torch.atan2(dtb[:, 1], dtb[:, 0])
        th_f = torch.where(
            d > r * 1.0001, base + torch.acos(torch.clamp(r / d, max=1.0)), base
        )
        twopi = 2 * math.pi
        th_f = th_f - twopi * torch.clamp(torch.ceil((th_f - th_b) / twopi), min=0.0)
        th_f = th_f + twopi * torch.clamp(
            torch.ceil((th_b - twopi - th_f) / twopi), min=0.0
        )
        frac = torch.arange(ARC_POINTS, dtype=dt, device=dev) / (ARC_POINTS - 1)
        ang = th_b + (th_f[:, None] - th_b) * frac
        arc_pts = b[:, None, :] + r * vec(torch.cos(ang), torch.sin(ang))
        p_f = arc_pts[:, -1]
        tan_f = vec(torch.sin(th_f), -torch.cos(th_f))
        bl = dist(p_f, tip)
        curl = torch.clamp(A["ttCurl"], 0.0, 1.0)
        ctrl = p_f + tan_f * (bl * 0.5)[:, None] + up((0.18 + 0.5 * curl) * k)
        blade = bezier2(p_f, ctrl, tip, 7)
        # dorsum depression (clicks)
        dors = A["dorsum"]
        x0, x1 = b[:, 0:1], tip[:, 0:1]

        def dip(pts):
            uu = (pts[..., 0] - x0) / torch.clamp(x1 - x0, min=1e-6)
            inside = (uu > 0) & (uu < 1) & (dors[:, None] > 0)
            w = torch.sin(math.pi * torch.clamp(uu, 1e-9, 1.0) ** 0.8)
            dy = torch.where(
                inside, -dors[:, None] * 0.9 * k * w * w, torch.zeros_like(uu)
            )
            return pts + up(dy)

        arc_pts = dip(arc_pts)
        blade = torch.cat([blade[:, :1], dip(blade[:, 1:-1]), blade[:, -1:]], dim=1)
        fren = j(5.7, 6.15)
        under = bezier2(
            tip, tip + vec(-0.3 * k - curl * 0.4 * k, -0.6 * k * one), fren, 6
        )
        floor = torch.stack([fren, j(6.2, 6.25), j(6.45, 6.55)], dim=1)
        lower_teeth = torch.stack(
            [j(6.45, 6.55), j(6.72, 7.15), j(6.95, 7.75), j(7.12, 7.45)], dim=1
        )
        ll_base = j(7.12, 7.45)
        lower_lip_in = bezier3(
            ll_base,
            ll_base + vec(0.25 * k + prot * 0.3, -0.05 * k * one),
            ll_edge + torch.tensor([-0.3 * k, -0.12 * k], dtype=dt, device=dev),
            ll_edge,
            7,
        )

        # ---- inner wall: epiglottis, then the tongue's back from where it first rises above the
        # epiglottis's tip (see the module docstring on padding)
        y_root = epi_tip[:, 1] + 0.05 * k
        back = torch.cat([root, arc_pts[:, 1:]], dim=1)  # [M, 40, 2]
        NBk = back.shape[1]
        rl = root.shape[1]
        rises = back[:, :, 1] > y_root[:, None]
        has = rises.any(1)
        st = torch.where(
            has, rises.float().argmax(1), torch.full((M,), NBk - 1, device=dev)
        )
        si = torch.clamp(st, 1, NBk - 1)
        ba = torch.gather(back, 1, (si - 1)[:, None, None].expand(-1, -1, 2))[:, 0]
        bb = torch.gather(back, 1, si[:, None, None].expand(-1, -1, 2))[:, 0]
        tc = torch.clamp(
            (y_root - ba[:, 1]) / torch.clamp(bb[:, 1] - ba[:, 1], min=1e-9), 0.0, 1.0
        )
        bcut = lerp(ba, bb, tc[:, None])
        # the points kept: back[i] for i ≥ start (and i > start once the start is on the arc, a quirk
        # of the Rust's indexing); the rest repeat the cut point, or the epiglottis's tip if none
        ii = torch.arange(NBk, device=dev)[None, :]
        keep = (ii >= st[:, None]) & ~((st[:, None] >= rl) & (ii == st[:, None]))
        keep = keep & has[:, None]
        lead = torch.where(
            (has & (st > 0))[:, None],
            bcut,
            torch.where(has[:, None], back[:, 0], epi_tip),
        )
        xs = torch.where(keep[..., None], back, lead[:, None, :])
        inner = torch.cat(
            [
                epi_base[:, None],
                epi_tip[:, None],
                lead[:, None],
                xs,
                blade[:, 1:],
                under[:, 1:],
                floor[:, 1:],
                lower_teeth[:, 1:],
                lower_lip_in,
            ],
            dim=1,
        )

        # ---- grid rays
        y_epi_top = glottis[:, 1] + 2.4 * k * self.body.pharynx
        y_phar_top = self.polar_c[1]
        fr = torch.arange(N_PH_RAYS, dtype=dt, device=dev) / N_PH_RAYS
        o_ph = vec(
            4.6 * k * torch.ones(M, N_PH_RAYS, dtype=dt, device=dev),
            lerp((y_epi_top + 0.12 * k)[:, None], y_phar_top, fr),
        )
        d_ph = torch.tensor([-1.0, 0.0], dtype=dt, device=dev).expand(M, N_PH_RAYS, 2)
        t = torch.arange(N_FAN, dtype=dt, device=dev) / (N_FAN - 1)
        fang = lerp(180.0, 36.0, t) * DEG
        w = torch.clamp((t - 0.45) / 0.55, min=0.0)
        o_fan = lerp(
            self.polar_c, self.polar_c2, (w * w * (3 - 2 * w))[:, None]
        ).expand(M, -1, -1)
        d_fan = vec(torch.cos(fang), torch.sin(fang)).expand(M, -1, -1)
        x_start = incisor[0] + 0.03 * k
        x_end = torch.clamp(
            torch.minimum(ul_edge[:, 0], ll_edge[:, 0]) - 0.04 * k,
            min=(x_start + 0.3 * k).item(),
        )
        fx = lerp(
            x_start,
            x_end[:, None],
            torch.arange(N_FRONT, dtype=dt, device=dev) / (N_FRONT - 1),
        )
        o_fr = vec(fx, torch.full_like(fx, 4.0 * k))
        d_fr = torch.tensor([0.0, 1.0], dtype=dt, device=dev).expand(M, N_FRONT, 2)
        o = torch.cat([o_ph, o_fan, o_fr], dim=1)
        dirs = torch.cat([d_ph, d_fan, d_fr], dim=1)
        max_len = torch.tensor(
            [9.0 * k] * N_PH_RAYS + [8.0 * k] * (N_FAN + N_FRONT), dtype=dt, device=dev
        )
        fan = torch.tensor(
            [False] * N_PH_RAYS + [True] * N_FAN + [False] * N_FRONT, device=dev
        )

        t_o, ok_o, _ = crossings(o, dirs, outer)
        ok_o = ok_o & (t_o >= 0) & (t_o <= max_len[None, :, None])
        assert bool(ok_o.any(-1).all()), (
            "a ray met no outer wall (the Rust drops it from the grid)"
        )
        t_out = torch.where(ok_o, t_o, torch.full_like(t_o, math.inf)).min(-1).values
        lim = t_out + 1.5 * k
        t_i, ok_i, ex_i = crossings(o, dirs, inner)
        ok_i = ok_i & (t_i >= 0) & (t_i <= lim[..., None])
        # rays that start below or in front of all tissue: the last inner crossing before the roof
        t_last = (
            torch.where(
                ok_i & (t_i < lim[..., None]), t_i, torch.full_like(t_i, -math.inf)
            )
            .max(-1)
            .values
        )
        t_in_plain = torch.where(
            torch.isfinite(t_last), t_last, torch.clamp(t_out - 3.5 * k, min=0.0)
        )
        t_in_fan = self._fan_inner(o, dirs, t_i, ok_i, ex_i, t_out, inner)
        t_in = torch.where(fan[None, :], t_in_fan, t_in_plain)
        gd = torch.clamp(t_out - t_in, min=0.0)
        p_in = o + dirs * torch.minimum(t_in, t_out)[..., None]
        p_out = o + dirs * t_out[..., None]
        # in front of the lower lip's edge (tucked for /f v/) the floor starts as the lip's face
        fx_ = p_out[:, TEETH:, 0]
        ahead_ll = fx_ > ll_edge[:, 0:1]
        face = torch.minimum(
            ll_edge[:, 1:2] - (fx_ - ll_edge[:, 0:1]) * 4.0 / 3.0, p_out[:, TEETH:, 1]
        )
        opn = smoothstep(LAB_FRONT_OPENS, 1.0, lab)[:, None] * smoothstep(
            0.0, LAB_FRONT_RAMP * k, fx_ - ll_edge[:, 0:1]
        )
        y_in = lerp(face, p_out[:, TEETH:, 1] - 3.5 * k, opn)
        fr_in = torch.where(ahead_ll[..., None], vec(fx_, y_in), p_in[:, TEETH:])
        fr_d = torch.where(ahead_ll, p_out[:, TEETH:, 1] - y_in, gd[:, TEETH:])
        p_in = torch.cat([p_in[:, :TEETH], fr_in], dim=1)
        gd = torch.cat([gd[:, :TEETH], fr_d], dim=1)
        mid = lerp(p_in, p_out, 0.5)
        ray_d = gd
        region = torch.tensor(
            [1] * SPLIT + [2] * (TEETH - SPLIT) + [3] * N_FRONT, device=dev
        )

        # ---- contact model near the oral roof
        cum = cumlen(roof)
        tip_pts = torch.cat([blade[:, 2:], under[:, 1:3]], dim=1)
        pt_ = project_on(roof, cum, tip_pts)
        pb_ = project_on(roof, cum, arc_pts)

        def closest(pr):
            i = pr["signed"].argmin(-1)
            dmin = torch.gather(pr["signed"], 1, i[:, None])[:, 0]
            seg = torch.gather(pr["seg"], 1, i[:, None])[:, 0]
            e = (
                torch.gather(roof, 1, (seg + 1)[:, None, None].expand(-1, -1, 2))[:, 0]
                - torch.gather(roof, 1, seg[:, None, None].expand(-1, -1, 2))[:, 0]
            )
            ln = torch.linalg.vector_norm(e, dim=-1)
            ln = torch.where(ln == 0, torch.ones_like(ln), ln)
            return dmin, vec(e[:, 1] / ln, -e[:, 0] / ln)

        c_tip, n_tip = closest(pt_)
        c_body, n_body = closest(pb_)
        outline = torch.cat(
            [
                root,
                arc_pts[:, 1:],
                blade[:, 1:],
                under[:, 1:],
                floor[:, 1:],
                root[:, :1],
            ],
            dim=1,
        )
        press = 1.0 - smoothstep(-0.06 * k, 0.0, torch.minimum(c_tip, c_body))
        half = 0.1 * k * (1.0 + press)
        roof_len = cum[:, -1]
        pr = project_on(roof, cum, p_out)
        on_roof = 1.0 - smoothstep(0.15 * k, 0.25 * k, pr["dist"])
        dc = signed_gap(outline, p_out)
        search = (dc - half[:, None] - pr["dist"]) <= 0.6 * k
        for frc in (-1.0, -0.5, 0.5, 1.0):
            sq = torch.minimum(
                torch.clamp(pr["s"] + frc * half[:, None], min=0.0), roof_len[:, None]
            )
            q = point_along(roof, cum, sq)
            dc = torch.where(search, torch.minimum(dc, signed_gap(outline, q)), dc)
        wgt = (1.0 - smoothstep(0.15 * k, 0.6 * k, dc)) * on_roof
        cap = (1.0 - smoothstep(0.45 * k, 0.6 * k, dc)) * on_roof
        dp = torch.clamp(dc, min=0.0)
        dm = gd + (torch.minimum(gd, dp) - gd) * cap
        gd = dm + (dp - dm) * wgt

        # ---- lip contact
        edge_x = torch.minimum(ul_edge[:, 0], ll_edge[:, 0])
        rl_ = 0.6 * k
        lab_open = smoothstep(0.0, LAB_HANDOVER, lab)
        full = lab_open >= 1.0
        ap_eff = ap + LAB_GAP * k * lab_open / torch.where(
            full, torch.ones_like(lab_open), 1.0 - lab_open
        )
        dxl = torch.clamp(edge_x[:, None] - mid[..., 0], min=0.0)
        lip_gap = torch.clamp(ap_eff[:, None] + dxl * dxl / (2.0 * rl_), min=0.0)
        lipped = (region[None, :] == 3) & (mid[..., 0] >= incisor[0]) & ~full[:, None]
        gd = torch.where(lipped, torch.minimum(gd, lip_gap), gd)

        # ---- areas
        lip_w = (1.1 + (self.body.lip_width - 1.1) * spread_w) * k
        dpos = torch.clamp(gd, min=0.0)
        pw = dpos**1.4
        ahead = smoothstep(0.0, 0.4 * k, mid[..., 0] - tip[:, 0:1])
        a_ph = AREA_LAW["pharynx"][0] * pw
        a_or = lerp(AREA_LAW["oral"][0] * pw, AREA_LAW["ahead"][0] * pw, ahead)
        uu = torch.clamp((mid[..., 0] - incisor[0]) / (0.45 * k), 0.0, 1.0)
        a_fr = lerp(
            AREA_LAW["front"][0] * pw, (math.pi / 4.0) * gd * lip_w[:, None], uu
        )
        area = torch.where(region == 1, a_ph, torch.where(region == 2, a_or, a_fr))
        # isolated spikes: each sample at most 1.35× the larger of its neighbours (in order)
        cols = list(area.unbind(1))
        for i in range(1, N_RAYS - 1):
            cols[i] = torch.minimum(
                cols[i], 1.35 * torch.maximum(cols[i - 1], cols[i + 1])
            )
        area = torch.stack(cols, dim=1)
        lat = A["lateral"][:, None]
        lat_on = (
            (lat > 0) & (region[None, :] >= 2) & (dist(mid, tip[:, None, :]) < 1.4 * k)
        )
        area = torch.where(lat_on, torch.maximum(area, lat * 0.35 * k * k), area)

        # ---- centreline
        epi_len = 2.4 * k * self.body.pharynx
        steps = dist(mid[:, 1:], mid[:, :-1])
        gs = epi_len + torch.cat(
            [torch.zeros_like(steps[:, :1]), torch.cumsum(steps, dim=1)], dim=1
        )
        teeth_s, lip_s, sp = gs[:, TEETH], gs[:, -1], gs[:, SPLIT]
        grid = dict(d=gd, ray=ray_d, area=area, s=gs, inner=p_in, outer=p_out, mid=mid)

        ph, ph_min = self._fill(A, grid, epi_len, torch.zeros_like(sp), sp, n_ph)
        orl, or_min = self._fill(A, grid, epi_len, sp, lip_s, n_or)
        return Geometry(
            ph_areas=ph,
            or_areas=orl,
            ph_min=ph_min,
            or_min=or_min,
            ph_len=sp,
            or_len=lip_s - sp,
            teeth_s=teeth_s,
            lip_s=lip_s,
            velum_port=torch.clamp(A["velum"], min=0.0) * k * k,
            contact_tip=c_tip,
            contact_body=c_body,
            contact_tip_n=n_tip,
            contact_body_n=n_body,
            grid=grid,
        )

    def _fan_inner(self, o, dirs, t_i, ok_i, ex_i, t_out, inner) -> Tensor:
        """A fan ray's inner surface: where it leaves the tongue (or the surface behind its origin),
        plus the thickness of any tissue it passes through before the roof."""
        k = self.k
        big = torch.full_like(t_i, math.inf)
        ts, order = torch.sort(torch.where(ok_i, t_i, big), dim=-1, stable=True)
        ex = torch.gather(ex_i, -1, order)
        n = ok_i.sum(-1)
        valid = torch.arange(ts.shape[-1], device=ts.device) < n[..., None]
        first_exit = (n > 0) & ex[..., 0]
        # behind the origin: the nearest inner crossing within 2.5 cm
        tb, okb, _ = crossings(o, -dirs, inner)
        okb = okb & (tb >= 0) & (tb <= 2.5 * k)
        t_back = torch.where(okb, tb, torch.full_like(tb, 2.5 * k)).min(-1).values
        t_back = torch.minimum(t_back, torch.full_like(t_back, 2.5 * k))
        t_in = torch.where(first_exit, ts[..., 0], -t_back)
        tissue = torch.zeros_like(t_out)
        skip = first_exit.clone()  # the first crossing was used
        done = torch.zeros_like(first_exit)
        K = int(n.max().item()) if n.numel() else 0
        tail = t_out + 1.5 * k
        for jj in range(K):
            live = valid[..., jj] & ~done
            if jj == 0:
                live = live & ~skip
                skip = torch.zeros_like(skip)
            else:
                use_skip = live & skip
                skip = skip & ~live
                live = live & ~use_skip
            tj = ts[..., jj]
            beyond = live & (tj >= t_out)
            done = done | beyond
            live = live & ~beyond
            entry = live & ~ex[..., jj]
            nxt = (
                ts[..., jj + 1]
                if jj + 1 < ts.shape[-1]
                else torch.full_like(tj, math.inf)
            )
            has_next = (jj + 1) < n
            bb = torch.where(has_next, nxt, tail)
            tissue = tissue + torch.where(
                entry,
                torch.minimum(bb, t_out) - torch.where(entry, tj, torch.zeros_like(tj)),
                torch.zeros_like(tj),
            )
            skip = skip | entry
        return t_in + tissue

    def _fill(self, A, grid, epi_len, s0, s1, n) -> tuple[Tensor, Tensor]:
        """``resample``'s ``fill``: n sections between s0 and s1, harmonic mean of 8 sub-samples
        each, closed if any sub-sample or grid sample inside is closed."""
        k = self.k
        gs, ga = grid["s"], grid["area"]
        d = (s1 - s0) / n
        sub = 8
        frac = (
            torch.arange(n, dtype=gs.dtype, device=gs.device)[:, None]
            + (torch.arange(sub, dtype=gs.dtype, device=gs.device)[None, :] + 0.5) / sub
        ).reshape(-1)
        ss = s0[:, None] + d[:, None] * frac[None, :]  # [M, n*sub]
        area = self._area_at(A, gs, ga, epi_len, ss).reshape(-1, n, sub)
        closed = (area <= 1e-5).any(-1)
        inv = (1.0 / torch.where(area <= 1e-5, torch.ones_like(area), area)).sum(-1)
        mn = area.min(-1).values
        i = torch.arange(n, dtype=gs.dtype, device=gs.device)
        a0 = s0[:, None] + d[:, None] * i
        a1 = s0[:, None] + d[:, None] * (i + 1) + torch.where(i == n - 1, 1e-6, 0.0)
        inside = (gs[:, None, :] >= a0[..., None]) & (
            gs[:, None, :] <= a1[..., None]
        )  # [M, n, 49]
        gmin = (
            torch.where(
                inside,
                ga[:, None, :],
                torch.full_like(inside, math.inf, dtype=ga.dtype),
            )
            .min(-1)
            .values
        )
        mn = torch.minimum(mn, gmin)
        closed = closed | (inside & (ga[:, None, :] <= 1e-5)).any(-1)
        out = torch.where(closed, torch.zeros_like(inv), sub / inv)
        mins = torch.where(closed, torch.zeros_like(inv), torch.minimum(mn, out))
        _ = k
        return out, mins

    def _area_at(self, A, gs, ga, epi_len, s) -> Tensor:
        k = self.k
        flare = 0.1 * epi_len
        epi = self._epilarynx(A, s, epi_len)
        g0s, g0a = gs[:, :1], ga[:, :1]
        hi = torch.searchsorted(
            gs.contiguous(), s.contiguous(), right=True
        )  # first grid s > s
        lo = torch.clamp(hi - 1, 0, gs.shape[1] - 2)
        hi = lo + 1
        sl, sh = torch.gather(gs, 1, lo), torch.gather(gs, 1, hi)
        al, ah = torch.gather(ga, 1, lo), torch.gather(ga, 1, hi)
        tt = (s - sl) / torch.clamp(sh - sl, min=1e-9)
        interp = lerp(al, ah, tt)
        last = ga[:, -1:]
        out = torch.where(s >= gs[:, -1:], last, interp)
        out = torch.where(s <= g0s, g0a.expand_as(s), out)
        out = torch.where(
            s <= epi_len, lerp(epi, g0a, (s - (epi_len - flare)) / flare), out
        )
        out = torch.where(s <= epi_len - flare, epi, out)
        _ = k
        return out

    def _epilarynx(self, A, s, epi_len) -> Tensor:
        u = s / epi_len
        vent = (0.95 * (1.0 - 0.75 * torch.clamp(A["ventricular"], 0.0, 1.0)))[:, None]
        top = lerp(1.9, 0.35, torch.clamp(A["epilarynx"], 0.0, 1.0))[:, None]

        def f(x):
            return torch.where(
                x < 0.3,
                torch.full_like(x, 1.3),
                torch.where(
                    x < 0.5,
                    vent.expand_as(x),
                    torch.minimum(lerp(1.2, top, (x - 0.5) / 0.5), top),
                ),
            )

        r = 0.03
        area = f(u)
        for u0 in (0.3, 0.5):
            ramp = lerp(
                f(torch.full_like(u, u0 - r)),
                f(torch.full_like(u, u0 + r)),
                (u - (u0 - r)) / (2 * r),
            )
            area = torch.where((u - u0).abs() < r, ramp, area)
        return area * self.k * self.k
