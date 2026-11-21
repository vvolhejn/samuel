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
