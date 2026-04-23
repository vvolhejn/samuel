"""Try several encoder fixes to break init-time feature collapse.

Baseline: cosine sim ~0.998 across batch.
Goal: cosine sim ~0.0 so inputs are preserved through encoder.
"""

from __future__ import annotations

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from samuel.config import TrainConfig
from samuel.data import build_dataloader
from samuel.encoder import SEANetEncoder, SEANetEncoderConfig


def collapse_stats(z: torch.Tensor, label: str) -> None:
    B = z.shape[0]
    z_flat = z.reshape(B, -1)
    z_n = z_flat / z_flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sims = z_n @ z_n.T
    od = sims[~torch.eye(B, dtype=torch.bool, device=z.device)]
    print(
        f"  {label:<40s}  overall std={z.std():.4f}  "
        f"std_batch={z.std(dim=0).mean():.4f}  "
        f"cos_sim={od.mean():.4f}"
    )


def apply_weight_norm(mod: nn.Module) -> None:
    for m in mod.modules():
        if isinstance(m, nn.Conv1d):
            nn.utils.parametrizations.weight_norm(m)


class SimpleEncoder(nn.Module):
    """Plain Conv1d + GroupNorm + ReLU stack, stride-matched to SEANet."""

    def __init__(self, dimension: int = 128):
        super().__init__()
        self.hop_length = 320
        c = [1, 32, 64, 128, dimension]
        s = [2, 4, 5, 8]
        layers = []
        for i in range(4):
            layers += [
                nn.Conv1d(
                    c[i], c[i + 1], kernel_size=s[i] * 2, stride=s[i], padding=s[i] // 2
                ),
                nn.GroupNorm(8 if c[i + 1] >= 8 else 1, c[i + 1]),
                nn.ReLU(),
            ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        batch_size=min(cfg.batch_size, 16),
        num_workers=0,
        sample_rate=cfg.data.sample_rate,
        chunk_seconds=cfg.data.chunk_seconds,
        rank=0,
        world_size=1,
        epoch=0,
        seed=0,
    )
    batch = next(iter(loader)).to(device)
    wav = batch.unsqueeze(1)
    hop_needed = 320
    pad = (hop_needed - wav.shape[-1] % hop_needed) % hop_needed
    if pad:
        wav = F.pad(wav, (0, pad))

    collapse_stats(wav.squeeze(1), "INPUT wav")

    # 1. Baseline SEANet (as-is)
    enc = SEANetEncoder(SEANetEncoderConfig(**cfg.model.encoder.model_dump())).to(
        device
    )
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "SEANet (default)")

    # 2. SEANet + weight_norm
    torch.manual_seed(0)
    enc = SEANetEncoder(SEANetEncoderConfig(**cfg.model.encoder.model_dump())).to(
        device
    )
    apply_weight_norm(enc)
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "SEANet + weight_norm")

    # 3. Simple 4-layer encoder
    torch.manual_seed(0)
    enc = SimpleEncoder(dimension=cfg.model.encoder.dimension).to(device)
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "SimpleEncoder (Conv+GN+ReLU)")

    # 4. Same SimpleEncoder but with BatchNorm (normalizes across batch)
    class SimpleEncoderBN(nn.Module):
        def __init__(self, dimension: int = 128):
            super().__init__()
            self.hop_length = 320
            c = [1, 32, 64, 128, dimension]
            s = [2, 4, 5, 8]
            layers = []
            for i in range(4):
                layers += [
                    nn.Conv1d(
                        c[i],
                        c[i + 1],
                        kernel_size=s[i] * 2,
                        stride=s[i],
                        padding=s[i] // 2,
                    ),
                    nn.BatchNorm1d(c[i + 1]),
                    nn.ReLU(),
                ]
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    torch.manual_seed(0)
    enc = (
        SimpleEncoderBN(dimension=cfg.model.encoder.dimension).to(device).eval()
    )  # eval=use running stats
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "SimpleEncoder + BatchNorm (eval)")

    torch.manual_seed(0)
    enc = SimpleEncoderBN(dimension=cfg.model.encoder.dimension).to(
        device
    )  # train mode
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "SimpleEncoder + BatchNorm (train)")

    # 4c. SEANet + BatchNorm injected after every Conv1d
    def inject_bn_after_convs(root: nn.Module) -> nn.Module:
        for name, child in list(root.named_children()):
            if isinstance(child, nn.Sequential):
                new_layers = []
                for sub in child:
                    new_layers.append(sub)
                    if isinstance(sub, CausalConv1d):
                        new_layers.append(nn.BatchNorm1d(sub.conv.out_channels))
                setattr(root, name, nn.Sequential(*new_layers))
            else:
                inject_bn_after_convs(child)
        return root

    from samuel.encoder import CausalConv1d

    torch.manual_seed(0)
    enc = SEANetEncoder(SEANetEncoderConfig(**cfg.model.encoder.model_dump())).to(
        device
    )
    enc = inject_bn_after_convs(enc).to(device)
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "SEANet + BatchNorm (train)")

    # 5. A very shallow 2-layer encoder
    class ShallowEncoder(nn.Module):
        def __init__(self, dimension: int = 128):
            super().__init__()
            self.hop_length = 320
            self.model = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=40, stride=20, padding=10),  # x20
                nn.ReLU(),
                nn.Conv1d(64, dimension, kernel_size=32, stride=16, padding=8),  # x320
            )

        def forward(self, x):
            return self.model(x)

    torch.manual_seed(0)
    enc = ShallowEncoder(dimension=cfg.model.encoder.dimension).to(device)
    with torch.no_grad():
        z = enc(wav)
    collapse_stats(z, "ShallowEncoder (2-layer)")


if __name__ == "__main__":
    main()
