"""The latent codec the FlowLM is trained on top of.

Training only ever needs a handful of operations from the codec: turn audio
into a `[B, T, C]` latent sequence, turn latents back into audio for samples
and evals, and describe itself (rates, latent size, a hash of the encoder for
the precomputed-latents store). `LatentCodec` is that contract; `MimiCodec`
wraps the released Mimi, and `codec.type` in the training config can name any
other implementation by import path (`package.module:ClassName`), built with
`codec.kwargs`.
"""

import hashlib
import importlib
import logging

import torch
from torch import nn

from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules.stateful_module import init_states

logger = logging.getLogger(__name__)


class LatentCodec(nn.Module):
    """Frozen audio <-> latent codec. Subclasses set the four attributes."""

    sample_rate: int
    frame_rate: float
    latent_dim: int
    # Frames the decoder needs before it can produce audio (generations
    # shorter than this are treated as empty).
    min_decode_frames: int = 1
    # Frames of audio re-encoded on the fly at the cut in latent-mode training,
    # to match a cold-start encoder. None asks the precompute to calibrate it
    # (see precompute_latents.measure_stitch_frames); 0 disables stitching,
    # which is exact for a codec whose latents are per-frame or causal.
    stitch_frames: int | None = None

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate / self.frame_rate)

    def encode_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        """`[B, 1, S]` audio at sample_rate -> `[B, T, C]` latents."""
        raise NotImplementedError

    def decode_to_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """`[B, T, C]` latents -> `[B, 1, S]` audio at sample_rate (non-streaming)."""
        raise NotImplementedError

    def encode_hash(self) -> str:
        """Identifies the encoder: latents precomputed under a different hash are stale."""
        raise NotImplementedError

    def export_state(self) -> dict[str, torch.Tensor]:
        """Tensors to bundle into the pocket-tts model.safetensors export."""
        return {}

    def compile(self):  # ty: ignore[invalid-method-override]
        """Optional torch.compile of the encode path."""

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.encode_to_latent(audio)


def _hash_tensors(h: "hashlib._Hash", named: list[tuple[str, torch.Tensor]]):
    for k, v in sorted(named):
        h.update(f"{k}:{tuple(v.shape)}".encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())


class MimiCodec(LatentCodec):
    """The released Mimi neural codec (24 kHz, 12.5 Hz, 32-dim latents)."""

    def __init__(self, mimi: MimiModel):
        super().__init__()
        self.mimi = mimi
        self.sample_rate = mimi.sample_rate
        self.frame_rate = mimi.frame_rate
        self.latent_dim = mimi.quantizer.dimension
        self.min_decode_frames = 8
        self.mimi.eval()
        for p in self.mimi.parameters():
            p.requires_grad_(False)

    def encode_to_latent(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mimi.encode_to_latent(audio)

    def decode_to_audio(self, latents: torch.Tensor) -> torch.Tensor:
        ratio = round(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        state = init_states(
            self.mimi, latents.shape[0], (latents.shape[1] + self.min_decode_frames) * ratio
        )
        return self.mimi.decode_from_latent(latents, state)

    def encode_hash(self) -> str:
        """Hash of the weights on the encode path (encoder, encoder transformer,
        downsample): identifies which Mimi produced a latents store."""
        h = hashlib.sha256()
        for name in ("encoder", "encoder_transformer", "downsample"):
            module = getattr(self.mimi, name, None)
            if module is None:
                continue
            _hash_tensors(h, [(f"{name}.{k}", v) for k, v in module.state_dict().items()])
        return h.hexdigest()

    def export_state(self) -> dict[str, torch.Tensor]:
        return {f"mimi.{k}": v.detach().cpu() for k, v in self.mimi.state_dict().items()}

    def compile(self):
        # Dynamic shapes: batches pad to their own longest row.
        self.mimi.encoder.compile(dynamic=True)
        self.mimi.encoder_transformer.compile(dynamic=True)


def import_codec_class(path: str) -> type[LatentCodec]:
    """`package.module:ClassName` (or `package.module.ClassName`) -> class."""
    if ":" in path:
        module_name, cls_name = path.split(":", 1)
    else:
        module_name, _, cls_name = path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), cls_name)
    if not (isinstance(cls, type) and issubclass(cls, LatentCodec)):
        raise TypeError(f"{path} is not a LatentCodec subclass")
    return cls
