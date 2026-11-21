"""The human voice's sample loop: lungs → glottis → airway → decimator → DC blocker.

``HumanVoice::render`` from ``crates/chimera/src/voice/human_voice.rs``, driven by control blocks
(``ctrl_every`` internal samples each): per block the airway's targets, the noise sources, the
glottis's parameters and the lung pressure to ramp to. ``replay`` takes those from a
``chimera dump`` trace, which makes the port checkable against the Rust sample for sample.
The output limiter is left out (it is off in the trace too).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch import Tensor

from samuel.chimera import rng
from samuel.chimera.glottis import PARAM_FIELDS, Glottis, pitch_factors
from samuel.chimera.network import Network, Topology

NETWORK_SEED = 12345
MAX_NOISE = 3


@dataclass
class VoiceConstants:
    fs_out: float
    oversample: int
    ctrl_every: int
    output_gain: float
    dc_coef: float
    decimator: Tensor  # FIR taps
    lungs: int
    lung_port: int
    lung_port_excess: float
    lung_port_coef: float
    glottis_port: int
    sub_node: int
    supra_node: int

    @property
    def fs(self) -> float:
        return self.fs_out * self.oversample


@dataclass
class Block:
    """One control block's inputs, each with a leading batch dimension."""

    area_target: Tensor  # [B, N]
    amin_target: Tensor
    dx_target: Tensor
    port_target: Tensor  # [B, NB]
    p_lung: Tensor  # [B]
    glottis: dict[str, Tensor]  # each [B]
    noise_src: Tensor  # [B, S] long
    noise_fwd: Tensor
    noise_bwd: Tensor
    noise_gain: Tensor  # [B, S]
    noise_lp: Tensor
    noise_hp: Tensor
    noise_active: Tensor  # [B, S] bool, a prefix


def render(
    topo: Topology,
    k: VoiceConstants,
    init: dict[str, Tensor],
    blocks: list[Block],
    n_out: int,
    trace: bool = False,
) -> dict[str, Tensor]:
    """Run the voice over ``blocks`` and return ``out`` ``[B, n_out]`` (and, with ``trace``, the
    per-internal-sample ``net_out``, ``glottal_area`` and ``glottal_flow``)."""
    B = blocks[0].p_lung.shape[0]
    dt = topo.port_len.dtype
    dev = topo.port_len.device
    m, ce = k.oversample, k.ctrl_every
    n_int = n_out * m
    net = Network(topo, B, init["area"], init["amin"], init["dx"], init["port_area"])
    glo = Glottis(k.fs, B, dtype=dt, device=dev)
    # per-sample block parameters for the pitch modulation (which steps every 32 samples)
    n_blocks = (n_int + ce - 1) // ce
    assert len(blocks) >= n_blocks, (len(blocks), n_blocks)

    def per_sample(key):
        v = torch.stack(
            [blocks[i].glottis[key] for i in range(n_blocks)], dim=-1
        )  # [B, n_blocks]
        return v.repeat_interleave(ce, dim=-1)[:, :n_int]

    pf = pitch_factors(
        k.fs,
        n_int,
        per_sample("vibrato_rate"),
        per_sample("vibrato_depth"),
        per_sample("drift"),
    )
    # the network's noise stream: one draw per active source per sample, in list order
    n_act = [int(blocks[i].noise_active[0].sum()) for i in range(n_blocks)]
    assert all(
        bool((blocks[i].noise_active.sum(-1) == n_act[i]).all())
        for i in range(n_blocks)
    ), "the noise stream is shared: every batch item needs the same source count"
    lens = [min(ce, n_int - i * ce) for i in range(n_blocks)]
    white_all = torch.tensor(
        rng.bipolar(NETWORK_SEED, sum(a * n for a, n in zip(n_act, lens))),
        dtype=dt,
        device=dev,
    )

    p_lung_now = torch.zeros(B, dtype=dt, device=dev)
    lung_flow = torch.zeros(B, dtype=dt, device=dev)
    ys, areas, flows = [], [], []
    prev_noise = None
    pos = 0
    wpos = 0
    for bi in range(n_blocks):
        b = blocks[bi]
        p_lung_step = (b.p_lung - p_lung_now) / ce
        net.begin_block(ce, b.area_target, b.amin_target, b.dx_target, b.port_target)
        S = b.noise_src.shape[1]
        keep = torch.full((B, S), S, dtype=torch.long, device=dev)
        if prev_noise is not None:
            ps, pf_, pa = prev_noise
            same = (
                (b.noise_src[:, :, None] == ps[:, None, :])
                & (b.noise_fwd[:, :, None] == pf_[:, None, :])
                & pa[:, None, :]
                & b.noise_active[:, :, None]
            )
            first = torch.where(
                same.any(-1), same.float().argmax(-1), torch.full_like(keep, S)
            )
            keep = first
        net.set_noise(
            b.noise_src,
            b.noise_fwd,
            b.noise_bwd,
            b.noise_gain,
            b.noise_lp,
            b.noise_hp,
            b.noise_active,
            keep,
        )
        prev_noise = (b.noise_src, b.noise_fwd, b.noise_active)
        na = n_act[bi]
        L = lens[bi]
        white_blk = white_all[wpos : wpos + na * L].view(L, na)
        wpos += na * L
        white_blk = F.pad(white_blk, (0, S - na))
        for i in range(L):
            p_lung_now = p_lung_now + p_lung_step
            lung_flow = lung_flow + k.lung_port_coef * (
                net.u[:, k.lung_port] - lung_flow
            )
            net.set_fixed_pressure(k.lungs, p_lung_now + k.lung_port_excess * lung_flow)
            a = glo.next(
                net.p[:, k.sub_node] - net.p[:, k.supra_node], b.glottis, pf[:, pos]
            )
            net.set_port_area(k.glottis_port, a)
            y = net.step(white_blk[i].expand(B, -1))
            ys.append(y)
            if trace:
                areas.append(a)
                flows.append(net.u[:, k.glottis_port])
            pos += 1
    y = torch.stack(ys, dim=-1)  # [B, n_int]
    out = decimate(y, k.decimator, m) * k.output_gain
    out = AF.lfilter(
        out,
        torch.tensor([1.0, -k.dc_coef], dtype=dt, device=dev),
        torch.tensor([1.0, -1.0], dtype=dt, device=dev),
        clamp=False,
    )
    res = {"out": out}
    if trace:
        res.update(
            net_out=y,
            glottal_area=torch.stack(areas, -1),
            glottal_flow=torch.stack(flows, -1),
        )
    return res


def decimate(y: Tensor, h: Tensor, m: int) -> Tensor:
    """The Rust ``Decimator``: a causal FIR, sampled at every m-th input (the m−1-th, 2m−1-th, …)."""
    n = h.shape[0]
    x = F.pad(y, (n - 1, 0))[:, None, :]
    full = F.conv1d(x, h.flip(0)[None, None, :].to(y.dtype))[
        :, 0, :
    ]  # full[t] = Σ h[k]·y[t−k]
    return full[:, m - 1 :: m]


# ---------------------------------------------------------------------- trace replay
def load_trace(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def from_trace(
    doc: dict, dtype=torch.float64, device=None
) -> tuple[Topology, VoiceConstants, dict[str, Tensor], list[Block]]:
    h, net = doc["header"], doc["net"]
    topo = Topology.from_dump(net, dtype=dtype, device=device)
    T = lambda x: torch.tensor(x, dtype=dtype, device=device)  # noqa: E731
    k = VoiceConstants(
        fs_out=h["fs_out"],
        oversample=h["oversample"],
        ctrl_every=h["ctrl_every"],
        output_gain=h["output_gain"],
        dc_coef=h["dc_coef"],
        decimator=T(h["decimator"]),
        lungs=h["lungs"],
        lung_port=h["lung_port"],
        lung_port_excess=h["lung_port_excess"],
        lung_port_coef=h["lung_port_coef"],
        glottis_port=h["glottis_port"],
        sub_node=h["sub_node"],
        supra_node=h["supra_node"],
    )
    init = {
        "area": T(net["area"])[None],
        "amin": T(net["amin"])[None],
        "dx": T(net["dx"])[None],
        "port_area": T(net["port_area"])[None],
    }
    assert all(
        c["glottis"]["chaos"] == 0 and c["glottis"]["fry"] <= 0.3 for c in doc["ctrl"]
    ), "chaos and fry's second pulse aren't ported"
    blocks = []
    for c in doc["ctrl"]:
        ns = c["noise"][:MAX_NOISE]
        S = MAX_NOISE
        pad = lambda key, fill: [s[key] for s in ns] + [fill] * (S - len(ns))  # noqa: E731
        L = lambda x: torch.tensor(x, dtype=torch.long, device=device)[None]  # noqa: E731
        blocks.append(
            Block(
                area_target=T(c["area_target"])[None],
                amin_target=T(c["amin_target"])[None],
                dx_target=T(c["dx_target"])[None],
                port_target=T(c["port_area_target"])[None],
                p_lung=T([c["p_lung"]]),
                glottis={f: T([c["glottis"][f]]) for f in PARAM_FIELDS},
                noise_src=L(pad("src", 0)),
                noise_fwd=L(pad("fwd", 0)),
                noise_bwd=L(pad("bwd", 0)),
                noise_gain=T(pad("gain", 0.0))[None],
                noise_lp=T(pad("lp_coef", 0.0))[None],
                noise_hp=T(pad("hp_coef", 1.0))[None],
                noise_active=torch.tensor(
                    [[i < len(ns) for i in range(S)]], device=device
                ),
            )
        )
    return topo, k, init, blocks


def replay(
    doc: dict,
    seconds: float | None = None,
    trace: bool = True,
    dtype=torch.float64,
    device=None,
) -> dict[str, Tensor]:
    topo, k, init, blocks = from_trace(doc, dtype=dtype, device=device)
    n_out = (
        doc["header"]["n_out"]
        if seconds is None
        else min(doc["header"]["n_out"], int(seconds * k.fs_out))
    )
    return render(topo, k, init, blocks, n_out, trace=trace)
