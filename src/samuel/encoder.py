"""SEANet 1D-convnet encoder ported from kyutai-labs/pocket-tts (Mimi).

The streaming state machinery from pocket-tts is dropped: convolutions use
static causal left-padding so the output matches the non-streaming path
(`model_state=None`) in pocket-tts. Causal padding is preserved so this
module can later be adapted to streaming inference.
"""

from typing import Literal

import numpy as np
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch import Tensor, nn
from torch.nn.utils.parametrizations import weight_norm

PadMode = Literal["constant", "replicate"]


class SEANetEncoderConfig(BaseModel):
    channels: int = 1
    dimension: int = 128
    n_filters: int = 32
    n_residual_layers: int = 3
    ratios: list[int] = Field(default_factory=lambda: [8, 5, 4, 2])
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    pad_mode: PadMode = "constant"
    compress: int = 2


class CausalConv1d(nn.Module):
    """Conv1d with causal left-padding.

    Matches pocket-tts `StreamingConv1d` behavior when called with
    `model_state=None`: a buffer of `(kernel_size - 1) * dilation - (stride - 1)`
    samples is prepended to the input. With `pad_mode="constant"` the buffer
    is zeros; with `pad_mode="replicate"` the first input sample is repeated.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: PadMode = "constant",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.stride = stride
        effective_kernel = (kernel_size - 1) * dilation + 1
        self.left_pad = effective_kernel - stride
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        _, _, t = x.shape
        assert t > 0 and t % self.stride == 0, "Steps must be a multiple of stride"
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0), mode=self.pad_mode)
        return self.conv(x)


class SEANetResnetBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_sizes: list[int] = [3, 1],
        dilations: list[int] = [1, 1],
        pad_mode: PadMode = "constant",
        compress: int = 2,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        hidden = dim // compress
        layers: list[nn.Module] = []
        for i, (k, d) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            layers += [
                nn.ELU(alpha=1.0),
                CausalConv1d(
                    in_chs, out_chs, kernel_size=k, dilation=d, pad_mode=pad_mode
                ),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class SEANetEncoder(nn.Module):
    """1D conv encoder from the Mimi VAE.

    Downsamples a waveform `[B, channels, T]` by `prod(ratios)` into latent
    frames `[B, dimension, T // prod(ratios)]`.
    """

    def __init__(self, config: SEANetEncoderConfig):
        super().__init__()
        self.config = config
        # pocket-tts reverses ratios inside the encoder so the first conv
        # downsampling uses the innermost ratio.
        ratios = list(reversed(config.ratios))
        self.hop_length = int(np.prod(ratios))

        n_filters = config.n_filters
        pad_mode = config.pad_mode

        mult = 1
        layers: list[nn.Module] = [
            CausalConv1d(
                config.channels, mult * n_filters, config.kernel_size, pad_mode=pad_mode
            ),
        ]
        for ratio in ratios:
            for j in range(config.n_residual_layers):
                layers.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[config.dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=config.compress,
                    )
                )
            layers += [
                nn.ELU(alpha=1.0),
                CausalConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        layers += [
            nn.ELU(alpha=1.0),
            CausalConv1d(
                mult * n_filters,
                config.dimension,
                config.last_kernel_size,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SEANetDecoder(nn.Module):
    """Symmetric decoder to ``SEANetEncoder``.

    Takes a latent ``[B, dimension, T_enc]`` and upsamples back to a waveform
    ``[B, channels, T_enc * prod(ratios)]`` using transposed convolutions.
    Used for the no-bottleneck sanity check (neural autoencoder instead of
    Pink Trombone as decoder); not streaming/causal.
    """

    def __init__(self, config: SEANetEncoderConfig):
        super().__init__()
        self.config = config
        ratios = list(reversed(config.ratios))  # encoder applies in this order
        # Decoder mirrors the encoder, so iterate in encoder order's reverse
        # (which equals config.ratios as written: outermost ratio first).
        dec_ratios = list(reversed(ratios))
        self.hop_length = int(np.prod(ratios))

        n_filters = config.n_filters
        pad_mode = config.pad_mode

        mult = 2 ** len(ratios)
        layers: list[nn.Module] = [
            CausalConv1d(
                config.dimension,
                mult * n_filters,
                config.last_kernel_size,
                pad_mode=pad_mode,
            ),
        ]
        for ratio in dec_ratios:
            layers += [
                nn.ELU(alpha=1.0),
                weight_norm(
                    nn.ConvTranspose1d(
                        mult * n_filters,
                        mult * n_filters // 2,
                        kernel_size=ratio * 2,
                        stride=ratio,
                        padding=ratio // 2 + ratio % 2,
                        output_padding=ratio % 2,
                    )
                ),
            ]
            mult //= 2
            for j in range(config.n_residual_layers):
                layers.append(
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[config.residual_kernel_size, 1],
                        dilations=[config.dilation_base**j, 1],
                        pad_mode=pad_mode,
                        compress=config.compress,
                    )
                )
        layers += [
            nn.ELU(alpha=1.0),
            CausalConv1d(
                mult * n_filters,
                config.channels,
                config.kernel_size,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.model(z)
