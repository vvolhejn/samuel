"""The CHIMERA port against the Rust engine.

``tests/fixtures/chimera_peter_200ms.json.gz`` is the first 200 ms of "Peter picked." from
``chimera dump`` (branch ``experiment/torch-port-dump`` of vvolhejn/chimera-vocal-synth): voicing
onset, the /p/ closure and its release, with turbulence noise on.
"""

import dataclasses
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from samuel.chimera import rng
from samuel.chimera.airway import build_airway
from samuel.chimera.network import Topology
from samuel.chimera.voice import from_trace, render

TRACE = Path(__file__).parent / "fixtures" / "chimera_peter_200ms.json.gz"


@pytest.fixture(scope="module")
def doc():
    with gzip.open(TRACE, "rt") as f:
        return json.load(f)


def test_rng_is_mulberry32():
    # Rng::new(12345): the first draws, computed by hand from the Rust's three lines
    def ref(seed, n):
        out, s = [], seed
        for _ in range(n):
            s = (s + 0x6D2B79F5) & 0xFFFFFFFF
            t = s
            t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
            t = (
                t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)
            ) & 0xFFFFFFFF
            out.append((t ^ (t >> 14)) / 4294967296.0)
        return out

    np.testing.assert_array_equal(rng.uniform(12345, 50), ref(12345, 50))
    np.testing.assert_array_equal(rng.uniform(777, 10, start=40), ref(777, 50)[40:])


def test_airway_is_built_as_in_rust(doc):
    ref = Topology.from_dump(doc["net"])
    air = build_airway(doc["net"]["fs"])
    for f in dataclasses.fields(Topology):
        a, b = getattr(ref, f.name), getattr(air.topo, f.name)
        if f.name == "noise_scale":
            continue
        if isinstance(a, torch.Tensor):
            assert a.shape == b.shape and torch.equal(a.double(), b.double()), f.name
        else:
            assert a == pytest.approx(b, rel=1e-12), f.name
    h = doc["header"]
    assert (air.lungs, air.lung_port, air.glottis_port, air.velum_port) == (
        h["lungs"],
        h["lung_port"],
        h["glottis_port"],
        h["velum_port"],
    )


def test_replay_matches_rust_sample_for_sample(doc):
    topo, k, init, blocks = from_trace(doc)
    with torch.no_grad():
        r = render(topo, k, init, blocks, doc["header"]["n_out"], trace=True)
    for key, rtol in [
        ("glottal_area", 1e-12),
        ("glottal_flow", 1e-12),
        ("net_out", 1e-10),
        ("out", 1e-6),
    ]:
        x = r[key][0].numpy()
        ref = np.asarray(doc[key])
        assert len(x) == len(ref), key
        err = np.abs(x - ref).max() / np.abs(ref).max()
        assert err < rtol, f"{key}: relative error {err:.2e}"


def test_gradients_reach_the_controls(doc):
    topo, k, init, blocks = from_trace(doc)
    lung = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    open_ = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    for b in blocks:
        b.p_lung = b.p_lung * lung
        b.area_target = b.area_target * open_
    n = int(0.06 * k.fs_out)  # voicing has started by then
    out = render(topo, k, init, blocks, n)["out"]
    loss = out.pow(2).sum()
    loss.backward()
    for g in [lung.grad, open_.grad]:
        assert g is not None and torch.isfinite(g) and g != 0


def test_geometry_matches_rust():
    """Articulations of "Peter picked." and copies jittered by up to 60 % of each range
    (``chimera dump-geometry`` with ``CHIMERA_GEO_JITTER=0.6``)."""
    from samuel.chimera.geometry import VocalTract

    with gzip.open(
        Path(__file__).parent / "fixtures" / "chimera_geometry.json.gz", "rt"
    ) as f:
        d = json.load(f)
    items = d["items"]
    art = torch.tensor(
        [i["artic"] for i in items], dtype=torch.float64, requires_grad=True
    )
    g = VocalTract().compute(art, d["n_ph"], d["n_or"])
    for key in [
        "ph_areas",
        "or_areas",
        "ph_min",
        "or_min",
        "ph_len",
        "or_len",
        "teeth_s",
        "lip_s",
        "velum_port",
        "contact_tip",
        "contact_body",
    ]:
        np.testing.assert_allclose(
            getattr(g, key).detach().numpy(),
            [i[key] for i in items],
            rtol=0,
            atol=1e-11,
            err_msg=key,
        )
    for key in ["d", "area", "s"]:
        np.testing.assert_allclose(
            g.grid[key].detach().numpy(),
            [i["grid"][key] for i in items],
            rtol=0,
            atol=1e-11,
            err_msg=key,
        )
    (g.ph_areas.sum() + g.or_areas.sum() + g.or_len.sum()).backward()
    assert torch.isfinite(art.grad).all() and (art.grad.abs().sum(0) > 0).sum() >= 12


def test_replay_from_articulators_matches_rust(doc):
    """The same 200 ms driven by the trace's articulators and source settings only: this port's
    geometry, slew limits, tube targets, noise placement and glottis mapping make every block."""
    from samuel.chimera.control import replay_articulated
    from samuel.chimera.glottis import PARAM_FIELDS

    rec = []
    with torch.no_grad():
        r = replay_articulated(doc, record=rec)
    assert len(rec) == len(doc["ctrl"])
    for key in ("area", "amin", "dx", "port_area"):
        np.testing.assert_allclose(
            r["init"][key], r["init_ref"][key], rtol=1e-13, atol=1e-14
        )
    gp = doc["header"]["glottis_port"]  # the dump records its target after the block
    for i, (b, c) in enumerate(zip(rec, doc["ctrl"])):
        for got, key in [
            (b.area_target, "area_target"),
            (b.amin_target, "amin_target"),
            (b.dx_target, "dx_target"),
        ]:
            np.testing.assert_allclose(
                got[0], c[key], rtol=1e-12, atol=1e-14, err_msg=f"{key} {i}"
            )
        port = np.delete(b.port_target[0].numpy(), gp)
        np.testing.assert_allclose(
            port, np.delete(c["port_area_target"], gp), rtol=1e-12, atol=1e-14
        )
        for f in PARAM_FIELDS:
            assert b.glottis[f].item() == pytest.approx(
                c["glottis"][f], rel=1e-12, abs=1e-15
            ), (f, i)
        ns = c["noise"]
        assert int(b.noise_active[0].sum()) == len(ns), i
        for j, s in enumerate(ns):
            assert [
                int(b.noise_src[0, j]),
                int(b.noise_fwd[0, j]),
                int(b.noise_bwd[0, j]),
            ] == [s["src"], s["fwd"], s["bwd"]], (i, j)
            for got, key in [
                (b.noise_gain, "gain"),
                (b.noise_lp, "lp_coef"),
                (b.noise_hp, "hp_coef"),
            ]:
                assert got[0, j].item() == pytest.approx(s[key], rel=1e-11), (key, i, j)
    for key, rtol in [("glottal_area", 1e-12), ("net_out", 1e-10), ("out", 1e-6)]:
        x, ref = r[key][0].numpy(), np.asarray(doc[key])
        assert len(x) == len(ref), key
        err = np.abs(x - ref).max() / np.abs(ref).max()
        assert err < rtol, f"{key}: relative error {err:.2e}"


def _trace_blocks(doc):
    """The trace's per-block articulators and source settings as ``synth`` parameters."""
    from samuel.chimera.control import DEFAULT_SOURCE
    from samuel.chimera.network import CM_H2O

    rows = []
    for c in doc["ctrl"]:
        g = c["glottis"]
        rows.append(
            c["artic"]
            + [
                g["f0"],
                c["p_lung"] / CM_H2O,
                g["abduction"] - 0.1 * DEFAULT_SOURCE["breathiness"] ** 2,
                g["closure"],
            ]
            + [
                DEFAULT_SOURCE[n]
                for n in ("breathiness", "tension", "brightness", "roughness")
            ]
        )
    return torch.tensor(rows, dtype=torch.float64)[None]


def test_synth_renders_the_trace_as_rust(doc):
    """``synth``'s renderer (constants built without a trace), started from the Rust's neutral pose
    with the Rust's noise, reproduces the trace."""
    from samuel.chimera.geometry import NEUTRAL_START
    from samuel.chimera.synth import Voice, render_blocks

    voice = Voice.build()
    _, k, _, _ = from_trace(doc)
    for f in dataclasses.fields(k):
        a, b = getattr(voice.k, f.name), getattr(k, f.name)
        if isinstance(a, torch.Tensor):
            np.testing.assert_allclose(a, b, atol=1e-15, err_msg=f.name)
        else:
            assert a == pytest.approx(b, rel=1e-12), f.name
    with torch.no_grad():
        r = render_blocks(
            voice,
            _trace_blocks(doc),
            doc["header"]["n_out"],
            start=torch.tensor([NEUTRAL_START], dtype=torch.float64),
            noise="rust",
        )
    ref = np.asarray(doc["out"])
    err = np.abs(r["out"][0].numpy() - ref).max() / np.abs(ref).max()
    assert err < 1e-6, err


def test_synth_params_ramp_between_frames():
    from samuel.chimera.synth import params_at_blocks

    p = torch.tensor([[[0.0], [1.0], [3.0]]])  # 3 frames, 256 samples each
    got = params_at_blocks(p, 256, 3 * 256)[0, :, 0]
    expect = torch.tensor([0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3, 3, 3])
    torch.testing.assert_close(got, expect)


def test_synth_api():
    """``chimera(params)``: shape, dtype, a voiced schwa, noise seeded per call and per item
    (the draws depend on the batch size: item 0 of a batch of 1 sounds different)."""
    from samuel.chimera.synth import N_PARAMS, chimera, default_params

    p = default_params(2, 4)  # 40 ms at 100 Hz
    assert p.shape == (2, 4, N_PARAMS)
    with torch.no_grad():
        a = chimera(p, seed=1, control_rate=100.0)
        b = chimera(p, seed=1, control_rate=100.0)
    assert a.shape == (2, 4 * 441) and a.dtype == torch.float32
    assert torch.isfinite(a).all() and a[:, 441:].std() > 1e-3
    torch.testing.assert_close(a, b, rtol=0, atol=0)  # same seed, same noise
    diff = (a[0] - a[1]).std() / a[0].std()
    assert 0 < diff < 0.1, diff  # independent noise, the same voice


def test_synth_gradients_through_a_lip_closure():
    """The lips seal (the last oral node's area reaches exactly 0): every trainable parameter but
    the shut velum gets a finite, nonzero gradient. ``sqrt(area)`` at a sealed node used to send
    inf·0 = NaN back to every articulator."""
    from samuel.chimera.synth import (
        DEFAULT_PARAM_SPEC,
        PARAM_NAMES,
        chimera,
        default_params,
    )

    p = default_params(1, 8, dtype=torch.float64)  # 80 ms at 100 Hz
    p[0, :, PARAM_NAMES.index("lipAp")] = torch.tensor([0.6] + [-0.4] * 7)
    p.requires_grad_(True)
    (chimera(p, seed=0, control_rate=100.0) ** 2).mean().backward()
    assert torch.isfinite(p.grad).all()
    reached = {n for i, n in enumerate(PARAM_NAMES) if p.grad[0, :, i].abs().sum() > 0}
    # velum < 0 is a seal with no slope (port area = max(velum, 0))
    assert set(DEFAULT_PARAM_SPEC) - reached == {"velum"}


def test_synth_gradients_match_finite_differences():
    """From a moving pose, per frame. (At a perfectly static pose, e.g. ``default_params``, the
    network's branch conditions sit on their switch points: the gradient is one-sided there and a
    central difference averages both sides.)"""
    from samuel.chimera.synth import PARAM_NAMES, chimera, default_params

    p0 = default_params(1, 3, dtype=torch.float64)
    for n, vals in {"lipAp": [0.5, 0.9, 0.7], "tbY": [6.6, 7.1, 6.9]}.items():
        p0[0, :, PARAM_NAMES.index(n)] = torch.tensor(vals, dtype=torch.float64)

    def loss(p):
        return (chimera(p, seed=0, control_rate=100.0) ** 2).mean()

    p = p0.clone().requires_grad_(True)
    loss(p).backward()
    eps = 1e-6
    for name, frame in [("lipAp", 1), ("tbY", 2), ("pressure", 0)]:
        d = torch.zeros_like(p0)
        d[0, frame, PARAM_NAMES.index(name)] = 1.0
        with torch.no_grad():
            fd = (loss(p0 + eps * d) - loss(p0 - eps * d)) / (2 * eps)
        assert (p.grad * d).sum().item() == pytest.approx(fd.item(), rel=1e-6), name


def test_glottis_cycle_state_matches_the_sample_loop():
    """``cycle_state_at`` (a cycle at a time) against ``Glottis.next`` (a sample at a time), with
    f0, jitter, shimmer and diplophonia changing every block."""
    from samuel.chimera.glottis import Glottis, cycle_state_at

    torch.manual_seed(0)
    fs, B, n = 88200.0, 2, 8820

    def per(x):
        return x.repeat_interleave(128, -1)[:, :n]

    def rand(lo, hi):
        return per(lo + (hi - lo) * torch.rand(B, n // 128 + 1, dtype=torch.float64))

    f0, jit, shim, dip = rand(90, 300), rand(0.0, 0.03), rand(0.02, 0.2), rand(0.0, 0.3)
    at = [0, 1, 128, 3000, 8819, 8820]
    glo = Glottis(fs, B)
    full = lambda v: torch.full((B,), v, dtype=torch.float64)  # noqa: E731
    fixed = dict(fry=0.0, chink=0.0, closure=0.0, open_quotient=0.6, effort=0.5,
                 open_with_abduction=0.98, skew=0.6, closure_sharpness=0.5, amplitude=0.17,
                 abduction=0.0)  # fmt: skip
    ref = []
    for i in range(n + 1):
        if i in at:
            s = glo.s
            ref.append([s.phase, s.period_scale, s.cycle_amp, s.cycle_index.double()])
        if i < n:
            p = {k: full(v) for k, v in fixed.items()}
            p.update(
                f0=f0[:, i], jitter=jit[:, i], shimmer=shim[:, i], diplophonia=dip[:, i]
            )
            glo.next(full(1e5), p, full(1.0))
    got = cycle_state_at(fs, f0, jit, shim, dip, torch.tensor(at))
    for j, key in enumerate(["phase", "period_scale", "cycle_amp", "cycle_index"]):
        r = torch.stack([row[j] for row in ref], 1)
        torch.testing.assert_close(got[key].double(), r, rtol=0, atol=1e-11, msg=key)
    assert ref[-1][3].min() > 10  # cycles were crossed


def test_synth_chunked_is_close_to_sequential(doc):
    """Chunks side by side in the batch (each after 25 ms of warm-up), overlap-added, against the
    sequential render: near the floor two noise seeds set (~43 dB SNR here), gradients finite."""
    from samuel.chimera.synth import chimera

    p = _trace_blocks(doc)
    rate = 44100 / 64
    with torch.no_grad():
        full = chimera(p, seed=0, control_rate=rate)[0, 2205:]
    pc = p.clone().requires_grad_(True)
    y = chimera(pc, seed=0, control_rate=rate, chunk_ms=50, warmup_ms=25)
    (y**2).mean().backward()
    assert torch.isfinite(pc.grad).all() and pc.grad.abs().sum() > 0
    y = y.detach()[0, 2205:]
    snr = 10 * torch.log10((full**2).sum() / ((y - full) ** 2).sum())
    assert snr > 30, snr


def test_synth_compiled_step_matches_eager():
    from samuel.chimera.synth import PARAM_NAMES, chimera, default_params

    p = default_params(1, 2, dtype=torch.float64)
    p[0, :, PARAM_NAMES.index("lipAp")] = torch.tensor([0.5, 0.9])
    with torch.no_grad():
        a = chimera(p, seed=0, control_rate=100.0)
        b = chimera(p, seed=0, control_rate=100.0, compile=True)
    torch.testing.assert_close(a, b, rtol=0, atol=1e-12 * a.abs().max().item())
