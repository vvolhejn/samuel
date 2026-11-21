"""CHIMERA's random streams, reproduced exactly (``crates/chimera/src/math.rs``).

``Rng`` is mulberry32: its state only ever adds a constant, so draw n is a pure function of n and a
whole stream is computed at once. None of this is differentiated.
"""

import numpy as np

_INC = 0x6D2B79F5


def uniform(seed: int, n: int, start: int = 0) -> np.ndarray:
    """Draws ``start .. start+n`` of ``Rng::new(seed).next()``, in [0, 1)."""
    seed = seed or 1
    s = (
        np.uint64(seed)
        + np.uint64(_INC)
        * (np.arange(start, start + n, dtype=np.uint64) + np.uint64(1))
    ) & np.uint64(0xFFFFFFFF)
    t = s.astype(np.uint64)
    m = np.uint64(0xFFFFFFFF)
    t = ((t ^ (t >> np.uint64(15))) * (t | np.uint64(1))) & m
    t = (t ^ ((t + (((t ^ (t >> np.uint64(7))) * (t | np.uint64(61))) & m)) & m)) & m
    return ((t ^ (t >> np.uint64(14))) & m).astype(np.float64) / 4294967296.0


def bipolar(seed: int, n: int, start: int = 0) -> np.ndarray:
    return uniform(seed, n, start) * 2.0 - 1.0


def gauss(seed: int, n: int, start: int = 0) -> np.ndarray:
    """``Rng::gauss``: four uniforms summed, centred and scaled (n draws use 4n uniforms)."""
    u = uniform(seed, 4 * n, 4 * start).reshape(n, 4)
    return (u.sum(1) - 2.0) * 1.7320508


def drift(
    seed: int, control_rate: float, steps: int, base_hz: float = 0.3
) -> np.ndarray:
    """``Drift::step`` ``steps`` times: three one-pole bands of unit variance, summed 1, ½, ⅓."""
    a = np.exp(
        -2 * np.pi * np.array([base_hz, 3 * base_hz, 9 * base_hz]) / control_rate
    )
    g = gauss(seed, 3 * steps).reshape(steps, 3)
    y = np.zeros(3)
    out = np.empty(steps)
    b = np.sqrt(1.0 - a * a)
    w = np.array([1.0, 0.5, 1.0 / 3.0])
    for i in range(steps):
        y = a * y + b * g[i]
        out[i] = (y * w).sum() / 1.4
    return out
