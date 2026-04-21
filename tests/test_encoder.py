import numpy as np
import torch

from samuel.encoder import CausalConv1d, SEANetEncoder, SEANetEncoderConfig


def test_output_shape_matches_hop_length():
    cfg = SEANetEncoderConfig(
        dimension=512,
        n_filters=64,
        n_residual_layers=1,
        ratios=[6, 5, 4],
        last_kernel_size=3,
    )
    enc = SEANetEncoder(cfg)
    hop = int(np.prod(cfg.ratios))
    assert enc.hop_length == hop

    n_frames = 7
    x = torch.randn(2, cfg.channels, hop * n_frames)
    y = enc(x)
    assert y.shape == (2, cfg.dimension, n_frames)


def test_defaults_produce_valid_encoder():
    enc = SEANetEncoder(SEANetEncoderConfig())
    hop = enc.hop_length
    y = enc(torch.randn(1, 1, hop * 3))
    assert y.shape == (1, 128, 3)


def test_causal_conv1d_preserves_length_when_stride_one():
    conv = CausalConv1d(4, 8, kernel_size=5, dilation=2, pad_mode="constant")
    x = torch.randn(1, 4, 16)
    assert conv(x).shape == (1, 8, 16)


def test_causal_conv1d_is_causal():
    # Changing a future sample must not affect past outputs.
    torch.manual_seed(0)
    conv = CausalConv1d(3, 3, kernel_size=5, pad_mode="constant")
    x = torch.randn(1, 3, 32)
    y = conv(x)

    x2 = x.clone()
    x2[..., 20:] += 10.0
    y2 = conv(x2)

    # Outputs at t < 20 depend only on samples at t' <= t, so they must match.
    torch.testing.assert_close(y[..., :20], y2[..., :20])
    # Later outputs should differ.
    assert not torch.allclose(y[..., 20:], y2[..., 20:])


def test_encoder_is_causal():
    cfg = SEANetEncoderConfig(
        dimension=32,
        n_filters=8,
        n_residual_layers=1,
        ratios=[4, 2],
        last_kernel_size=3,
    )
    enc = SEANetEncoder(cfg).eval()
    hop = enc.hop_length  # 8
    n_frames = 12
    torch.manual_seed(0)
    x = torch.randn(1, 1, hop * n_frames)
    y = enc(x)

    # Perturb the second half of the input — latent frames covering only the
    # unperturbed prefix must be identical.
    split = n_frames // 2
    x2 = x.clone()
    x2[..., split * hop :] += 5.0
    y2 = enc(x2)

    torch.testing.assert_close(y[..., :split], y2[..., :split])


def test_config_is_pydantic_serializable():
    cfg = SEANetEncoderConfig(dimension=256, ratios=[4, 4])
    round_tripped = SEANetEncoderConfig.model_validate_json(cfg.model_dump_json())
    assert round_tripped == cfg
