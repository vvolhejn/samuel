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
    "constrictionIndex",
    "constrictionDiameter",
]
N_PARAMS = 13

_TRACT_N = 44
_NOSE_N = int(28 / 44 * _TRACT_N)  # 28
_NOSE_START = _TRACT_N - _NOSE_N + 1  # 17
_BLADE_START = int(10 / 44 * _TRACT_N)  # 10
_TIP_START = int(32 / 44 * _TRACT_N)  # 32
_LIP_START = int(39 / 44 * _TRACT_N)  # 39
_NOSE_OFFSET = 0.8

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


def _make_nose_r() -> Tensor:
    M = _NOSE_N
    t = torch.arange(M, dtype=torch.float32) / M
    d = torch.where(t < 0.5, 0.4 + 1.6 * (2 * t), 0.5 + 1.5 * (2 - 2 * t))
    d = d.clamp(max=1.9)
    A = d**2
    r = torch.zeros(M)
    r[1:] = (A[:-1] - A[1:]) / (A[:-1] + A[1:] + 1e-10)
    return r  # [M]


_NOSE_R_CPU = _make_nose_r()


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
    vibrato = vibrato_gain * torch.sin(2 * math.pi * t.unsqueeze(0) * vibrato_freq)
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
    noise = noise * (1 - torch.sqrt(torch.clamp(tenseness, min=1e-10)))
    noise = noise * sn_asp

    output = (noise + voice) * intensity
    return output, noise_mod


# ---------------------------------------------------------------------------
# Vocal tract waveguide
# ---------------------------------------------------------------------------


def _compute_diameter_profile(
    tongue_index: Tensor,  # [B, S]
    tongue_diameter: Tensor,  # [B, S]
    constriction_index: Tensor,  # [B, S]
    constriction_diameter: Tensor,  # [B, S]
    N: int = _TRACT_N,
) -> Tensor:  # [B, S, N]
    B, S = tongue_index.shape
    device = tongue_index.device

    blade = _BLADE_START  # 10
    tip = _TIP_START  # 32
    lip = _LIP_START  # 39

    # 1. Base rest profile
    j = torch.arange(N, device=device, dtype=torch.float32)  # [N]
    base = torch.where(
        j < (7.0 / 44) * N - 0.5,
        torch.full_like(j, 0.6),
        torch.where(
            j < (12.0 / 44) * N, torch.full_like(j, 1.1), torch.full_like(j, 1.5)
        ),
    )  # [N]
    diameter = base.unsqueeze(0).unsqueeze(0).expand(B, S, N).contiguous()

    # 2. Tongue shape (blade_start:lip_start)
    L = lip - blade
    j_tongue = torch.arange(blade, lip, device=device, dtype=torch.float32)  # [L]
    interp = (tongue_index.unsqueeze(-1) - j_tongue) / (tip - blade)  # [B, S, L]
    angle = 1.1 * math.pi * interp
    # diameter_param = 2 + (tongue_diameter - 2) / 1.5
    td = (2 + (tongue_diameter - 2) / 1.5).unsqueeze(-1)  # [B, S, 1]
    curve = (1.5 - td + 1.7) * torch.cos(angle)  # [B, S, L]  (grid.offset=1.7)

    # Edge scale factors: blade+0 → 0.94, lip-2 → 0.94, lip-1 → 0.8
    scale = torch.ones(L, device=device)
    scale[0] *= 0.94
    if L >= 2:
        scale[L - 2] *= 0.94
    if L >= 1:
        scale[L - 1] *= 0.8
    curve = curve * scale  # [B, S, L]
    tongue_vals = 1.5 - curve  # [B, S, L]

    diameter = torch.cat(
        [
            diameter[:, :, :blade],
            tongue_vals,
            diameter[:, :, lip:],
        ],
        dim=2,
    )

    # 3. Constriction kernel
    j_all = torch.arange(N, device=device, dtype=torch.float32)  # [N]
    norm_idx = constriction_index / N  # [B, S]
    lower_bound = 25.0 / 44
    upper_bound = float(tip) / N

    idx_range = torch.where(
        norm_idx < lower_bound,
        torch.full_like(norm_idx, 10.0),
        torch.where(
            norm_idx >= upper_bound,
            torch.full_like(norm_idx, 5.0),
            10.0
            - (5.0 * (norm_idx - lower_bound)) / (upper_bound - lower_bound + 1e-10),
        ),
    )  # [B, S]

    c_idx = constriction_index.unsqueeze(-1)  # [B, S, 1]
    c_diam = constriction_diameter.unsqueeze(-1)  # [B, S, 1]
    ir = idx_range.unsqueeze(-1)  # [B, S, 1]

    offset = (j_all - c_idx).abs() - 0.5  # [B, S, N]
    scalar = torch.where(
        offset <= 0,
        torch.zeros_like(offset),
        torch.where(
            offset > ir,
            torch.ones_like(offset),
            0.5 * (1 - torch.cos(math.pi * offset / (ir + 1e-10))),
        ),
    )  # [B, S, N]

    new_diam = (c_diam - 0.3).clamp(min=0)  # [B, S, 1]
    diff = diameter - new_diam  # [B, S, N]
    new_diameter = new_diam + diff * scalar  # [B, S, N]
    diameter = torch.where(diff > 0, new_diameter, diameter)

    return diameter  # [B, S, N]


def _waveguide_step(
    right: Tensor,  # [B, N]
    left: Tensor,  # [B, N]
    nose_right: Tensor,  # [B, M]
    nose_left: Tensor,  # [B, M]
    glottis_s: Tensor,  # [B]
    r_s: Tensor,  # [B, N]  oral reflection coefficients (r[:,0] unused)
    r_L_s: Tensor,  # [B]
    r_R_s: Tensor,  # [B]
    r_N_s: Tensor,  # [B]
    nose_r: Tensor,  # [M]  fixed nose reflections
    turb_s: Tensor,  # [B, N]
    N: int,
    M: int,
    ns: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # 1. Turbulence injection
    right = right + turb_s
    left = left + turb_s

    # 2. Oral junctions
    # Convention: rj[:,i] = right.junction[i] (0-indexed)
    #             lj[:,i] = left.junction[i+1] (1-indexed shift, so lj[:,N-1]=junction[N])
    rj0 = left[:, 0] * 0.75 + glottis_s  # [B]  right.junction[0]
    lj_last = right[:, -1] * (-0.85)  # [B]  left.junction[N]

    offsets = r_s[:, 1:] * (right[:, :-1] + left[:, 1:])  # [B, N-1]
    rj_inner = right[:, :-1] - offsets  # [B, N-1]  right.junction[1..N-1]
    lj_inner = left[:, 1:] + offsets  # [B, N-1]  left.junction[1..N-1] → lj[:,0..N-2]

    rj = torch.cat([rj0.unsqueeze(1), rj_inner], dim=1)  # [B, N]
    lj = torch.cat([lj_inner, lj_last.unsqueeze(1)], dim=1)  # [B, N]

    # 3. Override velum junction at ns
    velum_common = nose_left[:, 0] + right[:, ns - 1]  # [B]
    rj_ns = r_R_s * left[:, ns] + (r_R_s + 1) * velum_common
    lj_ns = r_L_s * right[:, ns - 1] + (r_L_s + 1) * (nose_left[:, 0] + left[:, ns])

    rj = torch.cat([rj[:, :ns], rj_ns.unsqueeze(1), rj[:, ns + 1 :]], dim=1)
    lj = torch.cat([lj[:, : ns - 1], lj_ns.unsqueeze(1), lj[:, ns:]], dim=1)

    # 4. Nose entry: nose_right.junction[0]
    nose_rj0 = r_N_s * nose_left[:, 0] + (r_N_s + 1) * (left[:, ns] + right[:, ns - 1])

    # 5. Nose junctions
    # nose_rj[:,i] = nose_right.junction[i], nose_lj[:,i] = nose_left.junction[i+1]
    nose_lj_last = nose_right[:, -1] * (-0.85)  # nose_left.junction[M]
    nose_offsets = nose_r[1:] * (nose_left[:, 1:] + nose_right[:, :-1])  # [B, M-1]
    nose_rj_inner = nose_right[:, :-1] - nose_offsets  # [B, M-1]
    nose_lj_inner = nose_left[:, 1:] + nose_offsets  # [B, M-1]

    nose_rj = torch.cat([nose_rj0.unsqueeze(1), nose_rj_inner], dim=1)  # [B, M]
    nose_lj = torch.cat([nose_lj_inner, nose_lj_last.unsqueeze(1)], dim=1)  # [B, M]

    # 6. State update (nose fade=1.0)
    right_new = rj * 0.999
    left_new = lj * 0.999
    nose_right_new = nose_rj
    nose_left_new = nose_lj

    # 7. Output
    out = right_new[:, -1] + nose_right_new[:, -1]

    return right_new, left_new, nose_right_new, nose_left_new, out


def _tract(
    glottis_out: Tensor,  # [B, S]
    noise_mod: Tensor,  # [B, S]
    tongue_index: Tensor,  # [B, S]
    tongue_diameter: Tensor,  # [B, S]
    constriction_index: Tensor,  # [B, S]
    constriction_diameter: Tensor,  # [B, S]
) -> Tensor:  # [B, S]
    B, S = glottis_out.shape
    N = _TRACT_N
    M = _NOSE_N
    ns = _NOSE_START  # 17
    device = glottis_out.device

    # 1. Diameter profile [B, S, N] and oral reflections [B, S, N]
    diameter = _compute_diameter_profile(
        tongue_index, tongue_diameter, constriction_index, constriction_diameter, N
    )
    amplitude = diameter**2  # [B, S, N]
    A_prev = amplitude[:, :, :-1]
    A_curr = amplitude[:, :, 1:]
    r_inner = (A_prev - A_curr) / (A_prev + A_curr + 1e-10)  # [B, S, N-1]
    r = torch.cat([torch.zeros(B, S, 1, device=device), r_inner], dim=2)  # [B, S, N]

    # 2. Velum and 3-way junction reflections [B, S]
    velum = torch.where(
        (constriction_index > ns) & (constriction_diameter < -_NOSE_OFFSET),
        torch.full_like(constriction_index, 0.4),
        torch.full_like(constriction_index, 0.01),
    )
    A_L = amplitude[:, :, ns]
    A_R = amplitude[:, :, ns + 1]
    A_N = velum**2
    sum_A = A_L + A_R + A_N + 1e-10
    r_L = (2 * A_L - sum_A) / sum_A
    r_R = (2 * A_R - sum_A) / sum_A
    r_N = (2 * A_N - sum_A) / sum_A

    # 3. Turbulence injection [B, S, N] with STE
    c_idx = constriction_index
    c_diam = constriction_diameter
    thinness = torch.clamp(8 * (0.7 - c_diam), 0, 1)
    openness = torch.clamp(30 * (c_diam - 0.3), 0, 1)
    noise_amount = noise_mod * glottis_out * 0.66 * (thinness * openness) / 2  # [B, S]
    valid_mask = ((c_idx >= 2) & (c_idx <= N) & (c_diam > 0)).float()  # [B, S]

    lo_float = c_idx.detach().floor()
    frac = c_idx - lo_float  # STE: d/dc_idx = 1
    lo = lo_float.long().clamp(0, N - 2)

    # mask1: position lo+1 (always in [1, N-1])
    # mask2: position lo+2 (may be N, use N+1 classes and slice)
    mask1 = F.one_hot(lo + 1, N).float()  # [B, S, N]
    mask2 = F.one_hot((lo + 2).clamp(max=N), N + 1).float()[:, :, :N]  # [B, S, N]
    turb = (
        mask1 * (noise_amount * frac).unsqueeze(-1)
        + mask2 * (noise_amount * (1 - frac)).unsqueeze(-1)
    ) * valid_mask.unsqueeze(-1)  # [B, S, N]

    # 4. Fixed nose reflections
    nose_r = _NOSE_R_CPU.to(device=device, dtype=glottis_out.dtype)  # [M]

    # 5. Sample loop (BPTT)
    right = torch.zeros(B, N, device=device, dtype=glottis_out.dtype)
    left = torch.zeros(B, N, device=device, dtype=glottis_out.dtype)
    nose_right = torch.zeros(B, M, device=device, dtype=glottis_out.dtype)
    nose_left = torch.zeros(B, M, device=device, dtype=glottis_out.dtype)

    outputs = []
    for s in range(S):
        right, left, nose_right, nose_left, out = _waveguide_step(
            right,
            left,
            nose_right,
            nose_left,
            glottis_out[:, s],
            r[:, s, :],
            r_L[:, s],
            r_R[:, s],
            r_N[:, s],
            nose_r,
            turb[:, s, :],
            N,
            M,
            ns,
        )
        outputs.append(out)

    return torch.stack(outputs, dim=1) * 0.125  # [B, S]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pink_trombone(params: Tensor) -> Tensor:
    """Differentiable Pink Trombone vocal synthesizer.

    Args:
        params: [B, T, 13] at 12.5 Hz control rate.
            Parameters (in order): noise, frequency, tenseness, intensity,
            loudness, tongueIndex, tongueDiameter, vibratoWobble,
            vibratoFrequency, vibratoGain, tractLength,
            constrictionIndex, constrictionDiameter.

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

    return _tract(
        glottis_out=glottis_out,
        noise_mod=noise_mod,
        tongue_index=p["tongueIndex"],
        tongue_diameter=p["tongueDiameter"],
        constriction_index=p["constrictionIndex"],
        constriction_diameter=p["constrictionDiameter"],
    )
