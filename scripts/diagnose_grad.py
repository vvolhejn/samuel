"""Minimal diagnostic: does the training step produce nonzero gradients?"""

from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from samuel.config import TrainConfig
from samuel.data import build_dataloader
from samuel.losses import MultiScaleLogMagSTFTLoss
from samuel.model import PinkTromboneController
from samuel.pink_trombone import pink_trombone_ola


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = PinkTromboneController(cfg.model).to(device)
    loss_fn = MultiScaleLogMagSTFTLoss().to(device)

    loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        batch_size=min(cfg.batch_size, 8),
        num_workers=0,
        sample_rate=cfg.data.sample_rate,
        chunk_seconds=cfg.data.chunk_seconds,
        rank=0,
        world_size=1,
        epoch=0,
        seed=0,
    )
    batch = next(iter(loader)).to(device)
    wav_in = batch.unsqueeze(1)
    target = batch

    for use_autocast in [True, False]:
        model.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda"),
            ):
                params = model(wav_in)
        else:
            params = model(wav_in)
        params = params.float()

        print(f"\n=== autocast={use_autocast} ===")
        print(
            f"params dtype={params.dtype}, stats: min={params.min():.4f}, max={params.max():.4f}, mean={params.mean():.4f}, std={params.std():.4f}"
        )

        pred = pink_trombone_ola(
            params,
            ir_length=cfg.synth.ir_length,
            ir_impl=cfg.synth.ir_impl,
            control_rate=cfg.model.frame_rate,
        )
        S = min(pred.shape[-1], target.shape[-1])
        loss = loss_fn(pred[..., :S], target[..., :S])
        print(
            f"pred stats: min={pred.min():.4f}, max={pred.max():.4f}, mean={pred.mean():.4f}, std={pred.std():.4f}"
        )
        print(f"target stats: min={target.min():.4f}, max={target.max():.4f}")
        print(f"loss={loss.item():.6f}")

        loss.backward()

        total_norm = 0.0
        zero_params = 0
        nz_params = 0
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            gn = p.grad.norm().item()
            total_norm += gn**2
            if gn == 0.0:
                zero_params += 1
                print(f"  ZERO GRAD: {name}  shape={tuple(p.shape)}")
            else:
                nz_params += 1
                if "head" in name or "_blocks.0" in name:
                    print(f"  {name}  shape={tuple(p.shape)}  grad_norm={gn:.6e}")
        print(
            f"total grad norm={total_norm**0.5:.6e}, nonzero={nz_params}, zero={zero_params}"
        )


if __name__ == "__main__":
    main()
