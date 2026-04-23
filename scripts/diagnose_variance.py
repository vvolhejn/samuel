"""Check whether the model produces varying params across batch items.

If param std across batch (per-frame, per-param) ≈ 0, the encoder isn't
differentiating between clips and has collapsed to a constant output.
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

    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ),
    ):
        params = model(batch.unsqueeze(1))
    params = params.float()
    print(f"params shape: {tuple(params.shape)}  (B, T, 13)")

    from samuel.pink_trombone import PARAM_NAMES

    trainable = model.trainable_names_
    param_names = PARAM_NAMES

    print(
        f"\n{'param':<22s}  {'mean':>8s}  {'std_batch':>10s}  {'std_time':>10s}  {'range':>16s}"
    )
    for i, name in enumerate(param_names):
        p = params[..., i]  # [B, T]
        if name not in trainable and name in cfg.model.frozen_values:
            tag = "[frozen]"
        else:
            tag = ""
        mean = p.mean().item()
        std_batch = p.std(dim=0).mean().item()  # per-frame batch variance
        std_time = p.std(dim=1).mean().item()  # per-item temporal variance
        lo, hi = p.min().item(), p.max().item()
        print(
            f"  {name:<20s}{tag:<4s}  {mean:8.3f}  {std_batch:10.4f}  {std_time:10.4f}  [{lo:6.2f}, {hi:6.2f}]"
        )


if __name__ == "__main__":
    main()
