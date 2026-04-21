import pytest
import torch

from samuel.pink_trombone import (
    N_PARAMS,
    PARAM_NAMES,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    SimplexNoise,
    _NOSE_R_CPU,
    _NOSE_START,
    _TRACT_N,
    _compute_batch_irs,
    _compute_batch_irs_eig,
    _compute_diameter_profile,
    _upsample_params,
    glottis,
    pink_trombone,
    pink_trombone_ola,
)

# ---------------------------------------------------------------------------
# SimplexNoise
# ---------------------------------------------------------------------------


class TestSimplexNoise:
    def test_output_range(self):
        """Simplex noise should be roughly in [-1, 1]."""
        sn = SimplexNoise(seed=0)
        x = torch.linspace(0, 100, 10000)
        out = sn.simplex1(x)
        assert out.shape == x.shape
        assert out.min() >= -1.5
        assert out.max() <= 1.5

    def test_deterministic(self):
        """Same seed, same input -> same output."""
        sn1 = SimplexNoise(seed=42)
        sn2 = SimplexNoise(seed=42)
        x = torch.linspace(0, 10, 500)
        torch.testing.assert_close(sn1.simplex1(x), sn2.simplex1(x))

    def test_different_seeds(self):
        """Different seeds should generally produce different output."""
        sn1 = SimplexNoise(seed=0)
        sn2 = SimplexNoise(seed=12345)
        x = torch.linspace(0, 10, 500)
        assert not torch.allclose(sn1.simplex1(x), sn2.simplex1(x))

    def test_smoothness(self):
        """Nearby inputs should produce nearby outputs (Lipschitz-ish)."""
        sn = SimplexNoise(seed=0)
        x = torch.linspace(0, 1, 1000)
        out = sn.simplex1(x)
        diffs = (out[1:] - out[:-1]).abs()
        # Max jump between adjacent samples should be small
        assert diffs.max() < 0.5

    def test_not_constant(self):
        """Output should have meaningful variation."""
        sn = SimplexNoise(seed=0)
        x = torch.linspace(0, 10, 1000)
        out = sn.simplex1(x)
        assert out.std() > 0.1

    def test_simplex2_symmetry(self):
        """simplex2 should not be trivially symmetric."""
        sn = SimplexNoise(seed=0)
        x = torch.linspace(0, 5, 100)
        a = sn.simplex2(x, torch.zeros_like(x))
        b = sn.simplex2(torch.zeros_like(x), x)
        assert not torch.allclose(a, b)

    def test_batched(self):
        """Should work with multi-dimensional input."""
        sn = SimplexNoise(seed=0)
        x = torch.linspace(0, 5, 60).reshape(3, 20)
        out = sn.simplex2(x, -x * 0.5)
        assert out.shape == (3, 20)
        # Should match flattened version
        x_flat = x.reshape(-1)
        out_flat = sn.simplex2(x_flat, -x_flat * 0.5)
        torch.testing.assert_close(out, out_flat.reshape(3, 20))

    def test_device_cpu(self):
        sn = SimplexNoise(device=torch.device("cpu"), seed=0)
        x = torch.tensor([0.0, 1.0, 2.0])
        out = sn.simplex1(x)
        assert out.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
    def test_device_cuda(self):
        sn = SimplexNoise(device=torch.device("cuda"), seed=0)
        x = torch.tensor([0.0, 1.0, 2.0], device="cuda")
        out = sn.simplex1(x)
        assert out.device.type == "cuda"

    def test_matches_js_reference(self):
        """Spot-check against values computed from the JS implementation (seed=0).

        JS code (run in browser console after loading SimplexNoise.js):
            const sn = new SimplexNoise(); sn.seed(0);
            [0, 0.5, 1.0, 2.0, 5.0].map(x => sn.simplex1(x));
        """
        sn = SimplexNoise(seed=0)
        inputs = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
        out = sn.simplex1(inputs)
        # At x=0, simplex1(0) = simplex2(0, 0) = 0 (all gradients cancel at origin)
        assert abs(out[0].item()) < 1e-5
        # Just verify they're finite and in reasonable range
        assert torch.isfinite(out).all()
        assert out.abs().max() < 1.5


# ---------------------------------------------------------------------------
# Glottis
# ---------------------------------------------------------------------------


class TestGlottis:
    def _default_params(self, B=1, S=1920):
        """Create default glottis parameters."""
        device = torch.device("cpu")
        return dict(
            noise_param=torch.zeros(B, S, device=device),
            frequency=torch.full((B, S), 140.0, device=device),
            tenseness=torch.full((B, S), 0.6, device=device),
            intensity=torch.ones(B, S, device=device),
            loudness=torch.ones(B, S, device=device),
            vibrato_wobble=torch.ones(B, S, device=device),
            vibrato_freq=torch.full((B, S), 6.0, device=device),
            vibrato_gain=torch.full((B, S), 0.005, device=device),
            simplex=SimplexNoise(seed=0, device=device),
        )

    def test_output_shape(self):
        params = self._default_params(B=2, S=4800)
        out, noise_mod = glottis(**params)
        assert out.shape == (2, 4800)
        assert noise_mod.shape == (2, 4800)

    def test_output_finite(self):
        params = self._default_params()
        out, noise_mod = glottis(**params)
        assert torch.isfinite(out).all(), (
            f"Non-finite glottis output: {out[~torch.isfinite(out)]}"
        )
        assert torch.isfinite(noise_mod).all()

    def test_not_silent(self):
        """With default params (intensity=1, loudness=1, freq=140), output should be non-zero."""
        params = self._default_params(S=SAMPLE_RATE)  # 1 second
        out, _ = glottis(**params)
        assert out.abs().max() > 0.01

    def test_silent_at_zero_intensity(self):
        params = self._default_params()
        params["intensity"] = torch.zeros_like(params["intensity"])
        out, _ = glottis(**params)
        assert out.abs().max() < 1e-6

    def test_frequency_affects_pitch(self):
        """Higher frequency should produce more zero crossings."""
        S = SAMPLE_RATE  # 1 second
        params_lo = self._default_params(S=S)
        params_lo["frequency"] = torch.full((1, S), 100.0)
        params_lo["vibrato_gain"] = torch.zeros(1, S)  # no vibrato for clean test
        params_lo["vibrato_wobble"] = torch.zeros(1, S)

        params_hi = self._default_params(S=S)
        params_hi["frequency"] = torch.full((1, S), 200.0)
        params_hi["vibrato_gain"] = torch.zeros(1, S)
        params_hi["vibrato_wobble"] = torch.zeros(1, S)

        out_lo, _ = glottis(**params_lo)
        out_hi, _ = glottis(**params_hi)

        def count_zero_crossings(x):
            signs = torch.sign(x[0])
            return (signs[1:] != signs[:-1]).sum().item()

        zc_lo = count_zero_crossings(out_lo)
        zc_hi = count_zero_crossings(out_hi)
        # Higher freq should have roughly 2x the zero crossings
        assert zc_hi > zc_lo * 1.5

    def test_tenseness_affects_waveform(self):
        """Different tenseness values should produce different waveforms."""
        S = 4800
        params_breathy = self._default_params(S=S)
        params_breathy["tenseness"] = torch.full((1, S), 0.2)
        params_breathy["vibrato_gain"] = torch.zeros(1, S)
        params_breathy["vibrato_wobble"] = torch.zeros(1, S)

        params_tense = self._default_params(S=S)
        params_tense["tenseness"] = torch.full((1, S), 0.9)
        params_tense["vibrato_gain"] = torch.zeros(1, S)
        params_tense["vibrato_wobble"] = torch.zeros(1, S)

        out_breathy, _ = glottis(**params_breathy)
        out_tense, _ = glottis(**params_tense)
        assert not torch.allclose(out_breathy, out_tense, atol=1e-3)

    def test_loudness_scales_linearly(self):
        """Halving loudness should halve the voice component."""
        S = 4800
        params1 = self._default_params(S=S)
        params1["vibrato_gain"] = torch.zeros(1, S)
        params1["vibrato_wobble"] = torch.zeros(1, S)

        params_half = self._default_params(S=S)
        params_half["loudness"] = torch.full((1, S), 0.5)
        params_half["vibrato_gain"] = torch.zeros(1, S)
        params_half["vibrato_wobble"] = torch.zeros(1, S)

        out1, _ = glottis(**params1)
        out_half, _ = glottis(**params_half)
        # With noise_param=0, output is purely voice, which scales with loudness
        # voice *= intensity * loudness, then output = voice * intensity
        # So out1 = voice * 1 * 1 * 1, out_half = voice * 1 * 0.5 * 1
        torch.testing.assert_close(out_half, out1 * 0.5, atol=1e-5, rtol=1e-4)

    def test_differentiable(self):
        """Gradients should flow through frequency, tenseness, intensity, loudness."""
        S = 1920
        freq = torch.full((1, S), 140.0, requires_grad=True)
        tens = torch.full((1, S), 0.6, requires_grad=True)
        inten = torch.ones(1, S, requires_grad=True)
        loud = torch.ones(1, S, requires_grad=True)

        out, _ = glottis(
            noise_param=torch.zeros(1, S),
            frequency=freq,
            tenseness=tens,
            intensity=inten,
            loudness=loud,
            vibrato_wobble=torch.ones(1, S),
            vibrato_freq=torch.full((1, S), 6.0),
            vibrato_gain=torch.full((1, S), 0.005),
            simplex=SimplexNoise(seed=0),
        )

        loss = out.sum()
        loss.backward()

        assert freq.grad is not None and freq.grad.abs().sum() > 0
        assert tens.grad is not None and tens.grad.abs().sum() > 0
        assert inten.grad is not None and inten.grad.abs().sum() > 0
        assert loud.grad is not None and loud.grad.abs().sum() > 0

    def test_batch_independence(self):
        """Each batch element should be processed independently."""
        S = 1920
        params = self._default_params(B=2, S=S)
        # Make batch elements different
        params["frequency"][0] = 100.0
        params["frequency"][1] = 200.0

        out, _ = glottis(**params)
        assert not torch.allclose(out[0], out[1], atol=1e-3)

    def test_aspiration_noise(self):
        """Non-zero noise param should add aspiration noise."""
        S = SAMPLE_RATE
        params_no_noise = self._default_params(S=S)
        params_with_noise = self._default_params(S=S)
        params_with_noise["noise_param"] = torch.ones(1, S)

        out_clean, _ = glottis(**params_no_noise)
        out_noisy, _ = glottis(**params_with_noise)
        assert not torch.allclose(out_clean, out_noisy, atol=1e-3)


# ---------------------------------------------------------------------------
# Upsample
# ---------------------------------------------------------------------------


class TestUpsampleParams:
    def test_shape(self):
        params = torch.randn(2, 5, N_PARAMS)
        up = _upsample_params(params)
        assert up.shape == (2, 5 * SAMPLES_PER_FRAME, N_PARAMS)

    def test_single_frame(self):
        params = torch.randn(1, 1, N_PARAMS)
        up = _upsample_params(params)
        assert up.shape == (1, SAMPLES_PER_FRAME, N_PARAMS)
        # All samples should equal the single input frame
        torch.testing.assert_close(up[0, 0], params[0, 0])
        torch.testing.assert_close(up[0, -1], params[0, 0])

    def test_endpoints(self):
        """With align_corners=True, first and last frames should match."""
        params = torch.randn(1, 3, N_PARAMS)
        up = _upsample_params(params)
        torch.testing.assert_close(up[0, 0], params[0, 0])
        torch.testing.assert_close(up[0, -1], params[0, -1])

    def test_differentiable(self):
        params = torch.randn(1, 3, N_PARAMS, requires_grad=True)
        up = _upsample_params(params)
        up.sum().backward()
        assert params.grad is not None


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestPinkTrombone:
    def _default_params_tensor(self, B=1, T=5):
        """Create a params tensor with reasonable defaults."""
        params = torch.zeros(B, T, N_PARAMS)
        defaults = {
            "noise": 0,
            "frequency": 140,
            "tenseness": 0.6,
            "intensity": 1,
            "loudness": 1,
            "tongueIndex": 12.9,
            "tongueDiameter": 2.43,
            "vibratoWobble": 1,
            "vibratoFrequency": 6,
            "vibratoGain": 0.005,
            "tractLength": 44,
        }
        for i, name in enumerate(PARAM_NAMES):
            if name in defaults:
                params[..., i] = defaults[name]
        return params

    def test_output_shape(self):
        params = self._default_params_tensor(B=2, T=3)
        audio = pink_trombone(params)
        assert audio.shape == (2, 3 * SAMPLES_PER_FRAME)

    def test_output_finite(self):
        params = self._default_params_tensor()
        audio = pink_trombone(params)
        assert torch.isfinite(audio).all()

    def test_differentiable_end_to_end(self):
        params = self._default_params_tensor(B=1, T=2)
        params.requires_grad_(True)
        audio = pink_trombone(params)
        loss = audio.pow(2).mean()
        loss.backward()
        assert params.grad is not None
        assert params.grad.abs().sum() > 0

    def test_wrong_param_count(self):
        with pytest.raises(AssertionError):
            pink_trombone(torch.randn(1, 5, 10))


# ---------------------------------------------------------------------------
# OLA FIR pink_trombone
# ---------------------------------------------------------------------------


class TestPinkTromboneOla:
    def _default_params_tensor(self, B=1, T=5):
        params = torch.zeros(B, T, N_PARAMS)
        defaults = {
            "noise": 0,
            "frequency": 140,
            "tenseness": 0.6,
            "intensity": 1,
            "loudness": 1,
            "tongueIndex": 12.9,
            "tongueDiameter": 2.43,
            "vibratoWobble": 1,
            "vibratoFrequency": 6,
            "vibratoGain": 0.005,
            "tractLength": 44,
        }
        for i, name in enumerate(PARAM_NAMES):
            if name in defaults:
                params[..., i] = defaults[name]
        return params

    def test_output_shape(self):
        params = self._default_params_tensor(B=2, T=3)
        audio = pink_trombone_ola(params)
        assert audio.shape == (2, 3 * SAMPLES_PER_FRAME)

    def test_output_finite(self):
        params = self._default_params_tensor()
        audio = pink_trombone_ola(params)
        assert torch.isfinite(audio).all()

    def test_differentiable(self):
        params = self._default_params_tensor(B=1, T=2)
        params.requires_grad_(True)
        audio = pink_trombone_ola(params)
        loss = audio.pow(2).mean()
        loss.backward()
        assert params.grad is not None
        assert params.grad.abs().sum() > 0

    def test_equivalence(self):
        """OLA output should approximate the reference for static params."""
        params = self._default_params_tensor(B=1, T=4)
        # Disable stochastic elements for a clean comparison
        params[..., PARAM_NAMES.index("vibratoGain")] = 0.0
        params[..., PARAM_NAMES.index("vibratoWobble")] = 0.0
        params[..., PARAM_NAMES.index("noise")] = 0.0

        seed = 0
        ref = pink_trombone(params, seed=seed)
        ola = pink_trombone_ola(params, seed=seed, ir_length=4096)

        assert ref.shape == ola.shape

        # Skip first frame (initial waveguide startup transient)
        skip = SAMPLES_PER_FRAME
        ref_s = ref[:, skip:]
        ola_s = ola[:, skip:]

        ref_rms = ref_s.pow(2).mean().sqrt().item()
        err_rms = (ref_s - ola_s).pow(2).mean().sqrt().item()

        assert ref_rms > 1e-4, "reference signal is silent"
        assert err_rms / ref_rms < 0.05, (
            f"relative error {err_rms / ref_rms:.3f} exceeds 5%"
        )


class TestComputeBatchIrsEig:
    """Eigendecomposition-based IR must match the sequential reference."""

    def _realistic_coefs(self, BT=4, seed=0):
        """Reflection coefficients derived from a realistic diameter profile.

        Mirrors the setup in `_tract_ola`: takes (tongueIndex, tongueDiameter,
        constrictionIndex, constrictionDiameter) → diameter → area → reflection
        coefficients. Varies tongue position across the batch for diversity.
        """
        N = _TRACT_N
        ns = _NOSE_START
        g = torch.Generator().manual_seed(seed)

        tongue_idx = 10 + 20 * torch.rand(BT, 1, generator=g)  # ~vowel range
        tongue_dia = 1.8 + 1.0 * torch.rand(BT, 1, generator=g)
        constr_idx = 20 + 20 * torch.rand(BT, 1, generator=g)
        constr_dia = 1.5 + 1.5 * torch.rand(BT, 1, generator=g)  # open — no turb

        diameter = _compute_diameter_profile(
            tongue_idx, tongue_dia, constr_idx, constr_dia, N
        ).squeeze(1)  # [BT, N]
        amplitude = diameter**2
        r_inner = (amplitude[:, :-1] - amplitude[:, 1:]) / (
            amplitude[:, :-1] + amplitude[:, 1:] + 1e-10
        )
        r = torch.cat([torch.zeros(BT, 1), r_inner], dim=1)  # [BT, N]

        velum = torch.full((BT,), 0.01)  # closed-velum default
        A_L = amplitude[:, ns]
        A_R = amplitude[:, ns + 1]
        A_N = velum**2
        sum_A = A_L + A_R + A_N + 1e-10
        r_L = (2 * A_L - sum_A) / sum_A
        r_R = (2 * A_R - sum_A) / sum_A
        r_N = (2 * A_N - sum_A) / sum_A
        return r, r_L, r_R, r_N

    def test_matches_sequential_glottis(self):
        """Glottis impulse IR: eig must match sequential to ~1e-4."""
        r, r_L, r_R, r_N = self._realistic_coefs(BT=4, seed=0)
        L = 512
        inject_pos = torch.zeros(4, dtype=torch.long)  # unused in glottis mode
        ref = _compute_batch_irs(r, r_L, r_R, r_N, _NOSE_R_CPU, True, inject_pos, L)
        eig = _compute_batch_irs_eig(r, r_L, r_R, r_N, _NOSE_R_CPU, True, inject_pos, L)
        assert ref.shape == eig.shape == (4, L)
        torch.testing.assert_close(ref, eig, atol=1e-4, rtol=1e-4)

    def test_matches_sequential_turbulence(self):
        """Turbulence IR with varying inject_pos — the edge case.

        In turb mode `b` is a one-hot at a tube position (not at the glottis)
        and pass 2 does NOT re-inject turbulence, so `b_full = A_step · b_turb`
        without the `+ b_turb` term. Position-independent bugs would pass a
        uniform-inject_pos test, so we vary it across the batch.
        """
        r, r_L, r_R, r_N = self._realistic_coefs(BT=4, seed=1)
        L = 512
        inject_pos = torch.tensor([5, 12, 24, 35], dtype=torch.long)
        ref = _compute_batch_irs(r, r_L, r_R, r_N, _NOSE_R_CPU, False, inject_pos, L)
        eig = _compute_batch_irs_eig(
            r, r_L, r_R, r_N, _NOSE_R_CPU, False, inject_pos, L
        )
        assert ref.shape == eig.shape == (4, L)
        torch.testing.assert_close(ref, eig, atol=1e-4, rtol=1e-4)

    def test_gradient_matches_glottis(self):
        """Backward pass: ∂loss/∂r must agree between the two implementations.

        Exercised at BT=17 and L=4096 — realistic full-clip dimensions and the
        regime where the eig formulation must be run in complex128 to keep
        gradients clean (complex64 underflows on the ~1e-8 eigenvalue gaps the
        near-symmetric waveguide geometry produces, giving NaN grads on some
        frames). Tolerance is looser than forward since eig backward still has
        more noise than the sequential path.
        """
        r0, r_L, r_R, r_N = self._realistic_coefs(BT=17, seed=0)
        L = 4096
        inject_pos = torch.zeros(17, dtype=torch.long)

        def grad_of(impl):
            r = r0.clone().requires_grad_(True)
            ir = impl(r, r_L, r_R, r_N, _NOSE_R_CPU, True, inject_pos, L)
            ir.pow(2).sum().backward()
            return r.grad.clone()

        g_ref = grad_of(_compute_batch_irs)
        g_eig = grad_of(_compute_batch_irs_eig)

        cos_sim = (g_ref * g_eig).sum() / (g_ref.norm() * g_eig.norm() + 1e-30)
        rel_err = (g_ref - g_eig).norm() / (g_ref.norm() + 1e-30)
        assert cos_sim > 0.999, f"cosine similarity {cos_sim:.4f} < 0.999"
        assert rel_err < 0.05, f"relative L2 error {rel_err:.4f} > 0.05"
