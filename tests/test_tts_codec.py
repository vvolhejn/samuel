"""SamuelCodec: shapes, stacking, and the latent <-> parameter round trip.

Uses a randomly initialised controller, so no checkpoint download.
"""

from __future__ import annotations

import math

import pytest
import torch

from samuel.model import PinkTromboneController, PinkTromboneControllerConfig
from samuel.pink_trombone import N_PARAMS, PARAM_NAMES
from samuel.tts.codec import PYIN_FMAX, PYIN_FMIN, SamuelCodec


def _codec(stack: int) -> SamuelCodec:
    torch.manual_seed(0)
    controller = PinkTromboneController(
        PinkTromboneControllerConfig(samples_per_frame=512)
    )
    return SamuelCodec(controller, stack=stack, target_rms=0.05, pitch_workers=1)


def _audio(n_samples: int, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    t = torch.arange(n_samples) / 44100
    tone = 0.3 * torch.sin(2 * math.pi * 150 * t)
    return (
        (tone + 0.01 * torch.randn(n_samples, generator=g)).expand(batch, 1, -1).clone()
    )


@pytest.mark.parametrize("stack", [1, 8])
def test_encode_shapes_and_rates(stack: int):
    codec = _codec(stack)
    n = 44100  # 1 s -> ceil(44100 / 512) = 87 control frames
    lat = codec.encode_to_latent(_audio(n))
    frames = math.ceil(87 / stack)
    assert lat.shape == (1, frames, 8 * stack)
    assert codec.latent_dim == 8 * stack
    assert codec.frame_size == 512 * stack
    assert abs(codec.frame_rate * 512 * stack - 44100) < 1e-6
    assert torch.isfinite(lat).all()


def test_latents_round_trip_to_params():
    codec = _codec(1)
    lat = codec.encode_to_latent(_audio(44100))
    params = codec.latents_to_params(lat)
    assert params.shape == (1, 87, N_PARAMS)
    # Trainable params sit on bucket centres inside their ranges, so scaling
    # back is exact; f0 round-trips through log2.
    for name in codec.trainable_names:
        lo, hi, _ = codec.controller.config.param_spec[name]
        col = params[..., PARAM_NAMES.index(name)]
        assert (col >= lo - 1e-5).all() and (col <= hi + 1e-5).all()
    f0 = params[..., PARAM_NAMES.index("frequency")]
    assert (f0 >= PYIN_FMIN).all() and (f0 <= PYIN_FMAX).all()
    for name, value in codec.controller.config.frozen_values.items():
        assert torch.allclose(
            params[..., PARAM_NAMES.index(name)], torch.tensor(float(value))
        )
    again = codec.encode_to_latent.__wrapped__  # noqa: F841 -- documents that encode is no_grad
    rescaled = (params[..., codec.trainable_idx] - codec.lo) / (
        codec.hi - codec.lo
    ) * 2 - 1
    assert torch.allclose(rescaled, lat[..., :7], atol=1e-5)
    assert torch.allclose(torch.log2(f0 / codec.f0_ref), lat[..., 7], atol=1e-5)


def test_stacking_is_a_pure_reshape():
    audio = _audio(44100)
    p1 = _codec(1).latents_to_params(_codec(1).encode_to_latent(audio))
    codec8 = _codec(8)
    p8 = codec8.latents_to_params(codec8.encode_to_latent(audio))
    assert p8.shape[1] == 11 * 8
    assert torch.allclose(p1, p8[:, : p1.shape[1]], atol=1e-5)
    # The padding frames repeat the last real frame.
    assert torch.allclose(
        p8[:, p1.shape[1] :], p1[:, -1:].expand(-1, 88 - 87, -1), atol=1e-5
    )


def test_decode_shape():
    codec = _codec(1)
    lat = torch.zeros(2, 20, 8)
    audio = codec.decode_to_audio(lat)
    assert audio.shape == (2, 1, 20 * 512)
    assert torch.isfinite(audio).all()


def test_padding_rows_do_not_change_valid_rows():
    codec = _codec(1)
    a = _audio(44100)
    padded = torch.cat([a, torch.zeros(1, 1, 20000)], dim=-1)
    lat_a = codec.encode_to_latent(a)
    lat_p = codec.encode_to_latent(padded)
    # Same loudness normalisation (padding excluded), so the controller sees
    # the same signal on the valid frames up to pyin's own smoothing.
    assert torch.allclose(lat_a[:, :80, :7], lat_p[:, :80, :7], atol=1e-4)
