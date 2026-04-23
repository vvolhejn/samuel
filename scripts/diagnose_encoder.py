"""Check encoder-output variance: does SEANet differentiate inputs at init?

If z.std(dim=0).mean() ≪ z.std(dim=1).mean() or ≪ z.std(dim=2).mean(),
the encoder is collapsed — all batch items map to nearly the same features.
"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from samuel.config import TrainConfig
from samuel.data import build_dataloader
from samuel.model import PinkTromboneController


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = PinkTromboneController(cfg.model).to(device)

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

    # pad to hop multiple
    hop = model.encoder.hop_length
    pad = (hop - wav.shape[-1] % hop) % hop
    if pad:
        wav = torch.nn.functional.pad(wav, (0, pad))

    with torch.no_grad():
        z = model.encoder(wav)  # [B, 128, T]
    print(f"z shape: {tuple(z.shape)}  (B, C, T_enc)")
    print(
        f"input wav stats: mean={batch.mean():.4f} std={batch.std():.4f}  peak={batch.abs().max():.4f}"
    )
    print(f"  std over batch (per time; averaged):  {batch.std(dim=0).mean():.4f}")
    print(f"  std over time (per batch; averaged):   {batch.std(dim=1).mean():.4f}")
    # Pairwise cosine of input waveforms
    wav_flat = batch.view(batch.shape[0], -1)
    wav_n = wav_flat / wav_flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sims_wav = wav_n @ wav_n.T
    od = sims_wav[~torch.eye(batch.shape[0], dtype=torch.bool, device=device)]
    print(f"  input pairwise cosine sim: mean={od.mean():.4f}  min={od.min():.4f}")
    print()
    print(f"z overall:  mean={z.mean():.4f}  std={z.std():.4f}")
    print(f"  std over batch  (per C,T; averaged): {z.std(dim=0).mean():.4f}")
    print(f"  std over channel(per B,T; averaged): {z.std(dim=1).mean():.4f}")
    print(f"  std over time   (per B,C; averaged): {z.std(dim=2).mean():.4f}")

    # Similarity between pairs
    B = z.shape[0]
    z_flat = z.view(B, -1)
    z_norm = z_flat / z_flat.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sims = z_norm @ z_norm.T
    off_diag = sims[~torch.eye(B, dtype=torch.bool, device=device)]
    print("\nPairwise cosine similarity across batch:")
    print(f"  mean: {off_diag.mean():.4f}  (1.0 = collapsed, 0.0 = orthogonal)")
    print(f"  min: {off_diag.min():.4f}")


if __name__ == "__main__":
    main()
