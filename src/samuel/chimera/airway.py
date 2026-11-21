"""The human airway's network (``HumanAirway::new`` in ``crates/chimera/src/voice/human_airway.rs``).

lungs → trachea → glottis → pharynx ─┬─ oral → lips
                                      └─ velum port → nasal (+ maxillary sinus) → nostrils
plus the piriform fossae, a closed side branch above the epilarynx. Nodes and branches are numbered
as in the Rust, so a trace from ``chimera dump`` lines up index for index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from samuel.chimera.network import (
    C_SOUND,
    CM_H2O,
    KIND_PORT,
    KIND_SERIES,
    RHO,
    Topology,
)


def js_round(x: float) -> float:
    """``Math.round``: halves go up."""
    return math.floor(x + 0.5)


@dataclass(frozen=True)
class Body:
    name: str
    scale: float
    pharynx: float
    tongue_radius: float
    lip_width: float
    fold_length: float
    open_phase: float


BODY_MALE = Body("Adult male", 1.0, 1.0, 2.25, 4.6, 1.0, 0.0)
BODY_FEMALE = Body("Adult female", 0.9, 0.88, 2.2, 4.4, 0.67, 0.13)
BODY_CHILD = Body("Child", 0.74, 0.8, 2.15, 4.0, 0.35, 0.13)

# wall mass, resistance, stiffness per unit area (Ishizaka, French & Flanagan 1975)
WALL_SOFT = (1.5, 1600.0, 3.0e5)
WALL_TRACHEA = (0.6, 1000.0, 1.0e6)
WALL_NASAL = (1.0, 3000.0, 1.0e6)

LOWER_AIRWAY_R = 0.8
LOWER_AIRWAY_L = 5.2e-3


@dataclass
class Tube:
    first: int
    n: int
    branch_base: int

    def node(self, i: int) -> int:
        return self.first + i

    @property
    def last(self) -> int:
        return self.first + self.n - 1

    def branch(self, i: int) -> int:
        return self.branch_base + i


@dataclass
class _Builder:
    """``AirwayNetwork``'s construction API, in plain lists."""

    nodes: list[dict] = field(
        default_factory=list
    )  # area, dx, wall, perim, visc, rad, fixed
    branches: list[dict] = field(default_factory=list)  # a, b, kind, port spec
    rad_nodes: list[int] = field(default_factory=list)

    def tube(
        self,
        n: int,
        length: float,
        areas,
        wall=None,
        perimeter=1.0,
        viscous=1.0,
        radiating=0.0,
    ) -> Tube:
        t = Tube(len(self.nodes), n, len(self.branches))
        areas = areas if isinstance(areas, list) else [areas] * n
        for i in range(n):
            self.nodes.append(
                dict(
                    area=areas[i],
                    dx=length / n,
                    wall=wall,
                    perim=perimeter * 2.0 * math.sqrt(math.pi),
                    visc=viscous,
                    rad=radiating,
                    fixed=False,
                )
            )
        for i in range(n - 1):
            self.connect(t.node(i), t.node(i + 1))
        return t

    def connect(self, a: int, b: int) -> int:
        self.branches.append(dict(a=a, b=b, kind=KIND_SERIES))
        return len(self.branches) - 1

    def port(
        self,
        a: int,
        b: int,
        area: float,
        length: float,
        k_entrance=0.4,
        k_exit=1.0,
        slit=0.0,
        half=True,
        extra_r=0.0,
    ) -> int:
        self.branches.append(
            dict(
                a=a,
                b=b,
                kind=KIND_PORT,
                area=area,
                len=length,
                kin=k_entrance,
                kex=k_exit,
                slit=slit,
                half=half,
                extra_r=extra_r,
            )
        )
        return len(self.branches) - 1


@dataclass
class HumanAirway:
    """The built network and the indices the voice needs."""

    topo: Topology
    body: Body
    fs: float
    trachea: Tube
    pharynx: Tube
    oral: Tube
    nasal: Tube
    fossa: Tube
    lungs: int
    lung_port: int
    lung_port_excess: float
    lung_port_coef: float
    glottis_port: int
    velum_port: int
    area0: torch.Tensor  # [N] the areas the network was built with
    dx0: torch.Tensor
    port_area0: torch.Tensor  # [NB]

    @property
    def n_ph(self) -> int:
        return self.pharynx.n

    @property
    def n_or(self) -> int:
        return self.oral.n

    @property
    def dx_stable(self) -> float:
        """Shortest section the time step allows (Courant number 0.97)."""
        return C_SOUND / self.fs / 0.97

    def branch_of(self, t: Tube, i: int) -> int:
        return t.branch(min(max(i, 0), t.n - 2))


def build_airway(
    fs: float, body: Body = BODY_MALE, dtype=torch.float64, device=None
) -> HumanAirway:
    k = body.scale
    g = _Builder()
    dx_min = (C_SOUND / fs) / 0.93
    ph_min_len = 8.7 * k * (0.55 + 0.45 * body.pharynx) - 1.6 * k
    or_min_len = 8.3 * k - 1.0 * k
    n_ph = int(max(8.0, math.floor(ph_min_len / dx_min)))
    n_or = int(max(8.0, math.floor(or_min_len / dx_min)))
    tr_len = 12.0 * k
    trachea = g.tube(
        int(max(6.0, math.floor(tr_len / max(dx_min, 0.9 * k)))),
        tr_len,
        2.5 * k * k,
        wall=WALL_TRACHEA,
    )
    pharynx = g.tube(
        n_ph, 8.8 * k, 3.0 * k * k, wall=WALL_SOFT, radiating=0.25, perimeter=1.25
    )
    oral = g.tube(
        n_or, 8.3 * k, 3.0 * k * k, wall=WALL_SOFT, radiating=0.25, perimeter=1.5
    )
    na_len = 11.4 * k
    n_na = int(max(8.0, math.floor(na_len / max(dx_min, 0.5 * k))))

    def na_area(i):
        u = (i + 0.5) / n_na
        a = (
            2.6
            if u < 0.18
            else 2.0 - 0.6 * (u - 0.18)
            if u < 0.75
            else 1.4
            if u < 0.9
            else 1.0
        )
        return a * k * k

    nasal = g.tube(
        n_na,
        na_len,
        [na_area(i) for i in range(n_na)],
        wall=WALL_NASAL,
        perimeter=3.2,
        viscous=2.5,
    )
    n_fo = int(max(1.0, min(3.0, math.floor(2.0 * k / dx_min))))
    fossa = g.tube(n_fo, 2.0 * k, 0.9 * k * k, wall=WALL_SOFT, perimeter=1.3)
    lungs = g.tube(1, 1.0, 100.0).first
    g.nodes[lungs]["fixed"] = True
    tr_a = 2.5 * k * k
    r_ac = 0.5 * RHO * C_SOUND / tr_a
    lung_port = g.port(lungs, trachea.first, tr_a, 1.0, 0.0, 0.0, extra_r=r_ac)
    lung_port_excess = r_ac - LOWER_AIRWAY_R * (CM_H2O / 1000.0) / (k * k * k)
    lung_port_coef = 1.0 - math.exp((-lung_port_excess * k / LOWER_AIRWAY_L) / fs)
    glottis_port = g.port(
        trachea.last, pharynx.first, 0.0, 0.3 * k, 0.37, 1.0, slit=1.4 * k
    )
    g.connect(pharynx.last, oral.first)
    velum_port = g.port(pharynx.last, nasal.first, 0.0, 1.0 * k, 0.3, 0.8)
    fossa_node = pharynx.node(int(min(n_ph - 1, js_round(2.6 / 8.8 * n_ph))))
    g.connect(fossa_node, fossa.first)
    vol = 14.0 * k**3
    s_len = vol ** (1.0 / 3.0)
    sinus = g.tube(1, s_len, vol / s_len, perimeter=1.2).first
    g.port(
        nasal.node(int(js_round(n_na * 0.45))),
        sinus,
        0.12 * k * k,
        0.5 * k,
        0.0,
        0.0,
        half=False,
    )
    g.rad_nodes = [oral.last, nasal.last]

    # finalize
    N, NB = len(g.nodes), len(g.branches)
    T = lambda x: torch.tensor(x, dtype=dtype, device=device)  # noqa: E731
    deg = [0] * N
    nbr = torch.zeros(N, N, dtype=dtype, device=device)
    for br in g.branches:
        deg[br["a"]] += 1
        deg[br["b"]] += 1
        nbr[br["a"], br["b"]] += 1.0
        nbr[br["b"], br["a"]] += 1.0
    pget = lambda key, d: [br.get(key, d) for br in g.branches]  # noqa: E731
    wall = lambda i: [nd["wall"][i] if nd["wall"] else 0.0 for nd in g.nodes]  # noqa: E731
    topo = Topology(
        fs=fs,
        n_nodes=N,
        n_branches=NB,
        b_a=torch.tensor([br["a"] for br in g.branches], device=device),
        b_b=torch.tensor([br["b"] for br in g.branches], device=device),
        series=torch.tensor(
            [br["kind"] == KIND_SERIES for br in g.branches], device=device
        ),
        port_len=T(pget("len", 0.0)),
        port_kin=T(pget("kin", 0.0)),
        port_kex=T(pget("kex", 0.0)),
        port_slit=T(pget("slit", 0.0)),
        port_half=torch.tensor(pget("half", False), device=device),
        port_extra_r=T(pget("extra_r", 0.0)),
        fixed=torch.tensor([nd["fixed"] for nd in g.nodes], device=device),
        has_wall=torch.tensor(
            [nd["wall"] is not None for nd in g.nodes], device=device
        ),
        wall_m=T(wall(0)),
        wall_r=T(wall(1)),
        wall_k=T(wall(2)),
        wall_perim=T([nd["perim"] for nd in g.nodes]),
        wall_rad=T([nd["rad"] for nd in g.nodes]),
        visc_f=T([nd["visc"] for nd in g.nodes]),
        rad_nodes=torch.tensor(g.rad_nodes, device=device),
        is_junction=torch.tensor([d >= 3 for d in deg], device=device),
        nbr=nbr,
        closed_area=2e-4,
        min_node_area=0.02,
        squeeze_area=0.12,
        vol_lp_coef=1.0 - math.exp(-2.0 * math.pi * 60.0 / fs),
        viscous_omega=2.0 * math.pi * 700.0,
        viscous_scale=1.0,
        noise_scale=1.0,
    )
    return HumanAirway(
        topo=topo,
        body=body,
        fs=fs,
        trachea=trachea,
        pharynx=pharynx,
        oral=oral,
        nasal=nasal,
        fossa=fossa,
        lungs=lungs,
        lung_port=lung_port,
        lung_port_excess=lung_port_excess,
        lung_port_coef=lung_port_coef,
        glottis_port=glottis_port,
        velum_port=velum_port,
        area0=T([nd["area"] for nd in g.nodes]),
        dx0=T([nd["dx"] for nd in g.nodes]),
        port_area0=T(pget("area", 0.0)),
    )
