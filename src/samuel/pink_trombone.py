"""Differentiable Pink Trombone vocal synthesizer in PyTorch."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

SAMPLE_RATE = 24000
CONTROL_RATE = 12.5
SAMPLES_PER_FRAME = int(SAMPLE_RATE / CONTROL_RATE)  # 1920
BLOCK_SIZE = 128

PARAM_NAMES = [
    "noise",
    "frequency",
    "tenseness",
    "intensity",
    "loudness",
    "tongueIndex",
    "tongueDiameter",
    "vibratoWobble",
    "vibratoFrequency",
    "vibratoGain",
    "tractLength",
    "constriction0index",
    "constriction0diameter",
    "constriction1index",
    "constriction1diameter",
    "constriction2index",
    "constriction2diameter",
    "constriction3index",
    "constriction3diameter",
]
N_PARAMS = len(PARAM_NAMES)

# fmt: off
_P_TABLE = [
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,
    169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,
    124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,
    28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,251,
    34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,49,192,
    214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,138,236,205,
    93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
]

# Gradient vectors for 2D simplex noise (only x,y components used in dot2)
_GRAD3_XY = [
    [+1,+1], [-1,+1], [+1,-1], [-1,-1],
    [+1, 0], [-1, 0], [+1, 0], [-1, 0],
    [ 0,+1], [ 0,-1], [ 0,+1], [ 0,-1],
]
# fmt: on


class SimplexNoise:
    """Simplex noise matching Pink Trombone's JS implementation.

    All outputs are detached (no gradient flows through noise).
    """

    def __init__(self, device: torch.device = torch.device("cpu"), seed: int = 0):
        p = _P_TABLE
        seed = int(seed)
        if seed < 256:
            seed |= seed << 8

        perm = [0] * 512
        grad_idx = [0] * 512
        for i in range(256):
            shift = 0 if (i & 1) else 8
            v = p[i] ^ ((seed >> shift) & 255)
            perm[i] = perm[i + 256] = v
            grad_idx[i] = grad_idx[i + 256] = v % 12

        self.perm = torch.tensor(perm, dtype=torch.long, device=device)
        self.grad3 = torch.tensor(_GRAD3_XY, dtype=torch.float32, device=device)
        self.grad_idx = torch.tensor(grad_idx, dtype=torch.long, device=device)
        self.F2 = 0.5 * (math.sqrt(3) - 1)
        self.G2 = (3 - math.sqrt(3)) / 6

    @torch.no_grad()
    def simplex2(self, xin: Tensor, yin: Tensor) -> Tensor:
        s = (xin + yin) * self.F2
        i = torch.floor(xin + s).long()
        j = torch.floor(yin + s).long()

        t = (i + j).float() * self.G2
        x0 = xin - i.float() + t
        y0 = yin - j.float() + t

        i1 = (x0 > y0).long()
        j1 = 1 - i1

        x1 = x0 - i1.float() + self.G2
        y1 = y0 - j1.float() + self.G2
        x2 = x0 - 1.0 + 2.0 * self.G2
        y2 = y0 - 1.0 + 2.0 * self.G2

        ii = i & 255
        jj = j & 255

        gi0 = self.grad3[self.grad_idx[ii + self.perm[jj]]]
        gi1 = self.grad3[self.grad_idx[ii + i1 + self.perm[jj + j1]]]
        gi2 = self.grad3[self.grad_idx[ii + 1 + self.perm[jj + 1]]]

        d0 = gi0[..., 0] * x0 + gi0[..., 1] * y0
        d1 = gi1[..., 0] * x1 + gi1[..., 1] * y1
        d2 = gi2[..., 0] * x2 + gi2[..., 1] * y2

        t0 = 0.5 - x0**2 - y0**2
        n0 = torch.where(t0 < 0, torch.zeros_like(t0), t0**4 * d0)

        t1 = 0.5 - x1**2 - y1**2
        n1 = torch.where(t1 < 0, torch.zeros_like(t1), t1**4 * d1)

        t2 = 0.5 - x2**2 - y2**2
        n2 = torch.where(t2 < 0, torch.zeros_like(t2), t2**4 * d2)

        return 70.0 * (n0 + n1 + n2)

    def simplex1(self, x: Tensor) -> Tensor:
        return self.simplex2(x * 1.2, -x * 0.7)


# ---------------------------------------------------------------------------
# Parameter upsampling
# ---------------------------------------------------------------------------


def _upsample_params(params: Tensor) -> Tensor:
    """[B, T, P] at 12.5 Hz -> [B, T_samples, P] at 24 kHz."""
    B, T, P = params.shape
    T_samples = T * SAMPLES_PER_FRAME
    if T == 1:
        return params.expand(B, T_samples, P)
    # F.interpolate wants [B, C, L]
    up = F.interpolate(
        params.permute(0, 2, 1), size=T_samples, mode="linear", align_corners=True
    )
    return up.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Glottis (LF-model waveform + aspiration noise)
# ---------------------------------------------------------------------------


def _glottis(
    noise_param: Tensor,  # [B, S]
    frequency: Tensor,
    tenseness: Tensor,
    intensity: Tensor,
    loudness: Tensor,
    vibrato_wobble: Tensor,
    vibrato_freq: Tensor,
    vibrato_gain: Tensor,
    simplex: SimplexNoise,
) -> tuple[Tensor, Tensor]:
    """Returns (glottis_output, noise_modulator), both [B, S]."""
    B, S = frequency.shape
    device = frequency.device
    t = torch.arange(S, device=device, dtype=torch.float32) / SAMPLE_RATE  # [S]

    # ---- vibrato ----
    vibrato = vibrato_gain * torch.sin(
        2 * math.pi * t.unsqueeze(0) * vibrato_freq
    )
    sn_vib = 0.02 * simplex.simplex1(t * 4.07) + 0.04 * simplex.simplex1(t * 2.15)
    vibrato = vibrato + sn_vib.unsqueeze(0)

    wobble = 0.2 * simplex.simplex1(t * 0.98) + 0.4 * simplex.simplex1(t * 0.5)
    vibrato = vibrato + wobble.unsqueeze(0) * vibrato_wobble

    freq_mod = frequency * (1 + vibrato)

    # ---- tenseness (with simplex jitter) ----
    sn_tens = 0.1 * simplex.simplex1(t * 0.46) + 0.05 * simplex.simplex1(t * 0.36)
    tenseness = tenseness + sn_tens.unsqueeze(0)
    tenseness = tenseness + (3 - tenseness) * (1 - intensity)

    # ---- phase via cumulative frequency integration ----
    phase_inc = freq_mod / SAMPLE_RATE
    phase = torch.cumsum(phase_inc, dim=-1)
    phase = phase - phase.floor()  # position within glottal period [0, 1)

    # ---- LF-model coefficients (vectorised, per-sample) ----
    Rd = torch.clamp(3 * (1 - tenseness), 0.5, 2.7)
    Ra = -0.01 + 0.048 * Rd
    Rk = 0.224 + 0.118 * Rd
    Rg = (Rk / 4 * (0.5 + 1.2 * Rk)) / (0.11 * Rd - Ra * (0.5 + 1.2 * Rk) + 1e-10)

    Tp = 1 / (2 * Rg + 1e-10)
    Te = Tp + Tp * Rk

    epsilon = 1 / (Ra + 1e-10)
    shift = torch.exp(-epsilon * (1 - Te))
    Delta = 1 - shift + 1e-10

    omega = math.pi / (Tp + 1e-10)

    # integral terms for E0 / alpha
    RHS = ((1 / epsilon) * (shift - 1) + (1 - Te) * shift) / Delta
    total_lower = -(Te - Tp) / 2 + RHS
    total_upper = -total_lower

    s_val = torch.sin(omega * Te)
    y_val = -math.pi * s_val * total_upper / (Tp * 2 + 1e-10)
    z_val = torch.log(torch.abs(y_val) + 1e-10)

    alpha = z_val / (Tp / 2 - Te + 1e-10)
    E0 = -1 / (s_val * torch.exp(alpha * Te) + 1e-10)

    # ---- waveform ----
    decay = (-torch.exp(-epsilon * (phase - Te)) + shift) / Delta
    sine = E0 * torch.exp(alpha * phase) * torch.sin(omega * phase)
    voice = torch.where(phase > Te, decay, sine)
    voice = voice * intensity * loudness

    # ---- noise modulator ----
    pos_amp = torch.clamp(torch.sin(2 * math.pi * phase), min=0)
    noise_mod = pos_amp * 0.2 + 0.1
    noise_mod = noise_mod + (1 - tenseness * intensity) * 3

    # ---- aspiration noise ----
    sn_asp = (0.02 * simplex.simplex1(t * 1.99) + 0.2).unsqueeze(0)
    noise = noise_param * noise_mod
    noise = noise * intensity * intensity
    noise = noise * (1 - torch.sqrt(torch.clamp(tenseness, min=0)))
    noise = noise * sn_asp

    output = (noise + voice) * intensity
    return output, noise_mod


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pink_trombone(params: Tensor) -> Tensor:
    """Differentiable Pink Trombone vocal synthesizer.

    Args:
        params: [B, T, 19] at 12.5 Hz control rate.
            Parameters (in order): noise, frequency, tenseness, intensity,
            loudness, tongueIndex, tongueDiameter, vibratoWobble,
            vibratoFrequency, vibratoGain, tractLength,
            constriction{0-3}index, constriction{0-3}diameter.

    Returns:
        [B, T_samples] audio at 24 kHz.  T_samples = T * 1920.
    """
    B, T, P = params.shape
    assert P == N_PARAMS, f"Expected {N_PARAMS} parameters, got {P}"

    simplex = SimplexNoise(device=params.device)
    params_up = _upsample_params(params)  # [B, T*1920, P]

    p = {name: params_up[..., i] for i, name in enumerate(PARAM_NAMES)}

    glottis_out, noise_mod = _glottis(
        noise_param=p["noise"],
        frequency=p["frequency"],
        tenseness=p["tenseness"],
        intensity=p["intensity"],
        loudness=p["loudness"],
        vibrato_wobble=p["vibratoWobble"],
        vibrato_freq=p["vibratoFrequency"],
        vibrato_gain=p["vibratoGain"],
        simplex=simplex,
    )

    # TODO: tract waveguide simulation
    return glottis_out
