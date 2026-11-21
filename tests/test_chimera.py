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
