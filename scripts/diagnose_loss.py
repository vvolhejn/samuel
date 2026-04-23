"""What does the multi-scale STFT loss look like for different prediction strategies?

- pred = target (perfect):         expect ~0
- pred = zeros (silence):
- pred = noise (matched std):
- pred = mean of targets:
- pred = time-shifted target:      phase-invariant, should be low
- pred = single vowel synth:
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from samuel.config import TrainConfig
from samuel.data import build_dataloader
from samuel.losses import MultiScaleLogMagSTFTLoss


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
    batch = next(iter(loader)).to(device)  # [B, S]
    loss_fn = MultiScaleLogMagSTFTLoss().to(device)

    target_std = batch.std().item()
    target_rms = batch.pow(2).mean().sqrt().item()
    print(
        f"target stats: std={target_std:.4f} rms={target_rms:.4f} peak={batch.abs().max():.4f}"
    )

    def L(p):
        return loss_fn(p, batch).item()

    # 1. Perfect
    print(f"\nloss(pred = target):            {L(batch):.5f}")

    # 2. Silence
    print(f"loss(pred = zeros):             {L(torch.zeros_like(batch)):.5f}")

    # 3. Small silence (not exact)
    print(f"loss(pred = 1e-6):              {L(torch.full_like(batch, 1e-6)):.5f}")

    # 4. White noise, matched std
    noise = torch.randn_like(batch) * target_std
    print(f"loss(pred = white noise, std matched): {L(noise):.5f}")

    # 5. White noise, 0.1x std
    print(f"loss(pred = white noise, 0.1x std):    {L(noise * 0.1):.5f}")

    # 6. Copy of first clip -> all batch items
    first = batch[0:1].expand_as(batch)
    print(f"loss(pred = first-clip broadcast): {L(first):.5f}")

    # 7. Mean over batch, broadcast (what model might collapse to)
    mean_audio = batch.mean(dim=0, keepdim=True).expand_as(batch)
    print(f"loss(pred = mean audio broadcast): {L(mean_audio):.5f}")

    # 8. Time-shifted target
    shifted = torch.roll(batch, shifts=1000, dims=-1)
    print(f"loss(pred = target shifted by 1000 samples): {L(shifted):.5f}")

    # 9. Sine waves at various frequencies
    t = (
        torch.arange(batch.shape[-1], device=device, dtype=torch.float32)
        / cfg.data.sample_rate
    )
    for freq in [100, 200, 440, 800, 1500]:
        sine = (
            torch.sin(2 * 3.14159 * freq * t).unsqueeze(0).expand_as(batch) * target_std
        )
        print(f"loss(pred = sine @ {freq} Hz, std matched): {L(sine):.5f}")


if __name__ == "__main__":
    main()
