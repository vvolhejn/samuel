"""Time each stage of a training step to locate the bottleneck.

Run with: uv run python scripts/benchmark_step.py
"""

from __future__ import annotations

import time

import hydra
import torch
from omegaconf import DictConfig

from samuel.config import TrainConfig
from samuel.data import build_dataloader
from samuel.losses import MultiScaleLogMagSTFTLoss
from samuel.model import PinkTromboneController
from samuel.pink_trombone import pink_trombone_ola


class Timer:
    def __init__(self, device: torch.device):
        self.device = device
        self.records: dict[str, list[float]] = {}
        self._t0: float = 0.0
        self._label: str = ""

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def __enter__(self) -> "Timer":
        raise RuntimeError("use start/stop")

    def start(self, label: str) -> None:
        self._sync()
        self._label = label
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._sync()
        dt = time.perf_counter() - self._t0
        self.records.setdefault(self._label, []).append(dt)

    def summary(self, warmup: int) -> None:
        print(f"\n=== Per-stage timing (averaged over steps {warmup + 1}..n) ===")
        total = 0.0
        for label, xs in self.records.items():
            xs = xs[warmup:]
            if not xs:
                continue
            mean = sum(xs) / len(xs)
            total += mean
            print(
                f"  {label:<20s} {mean * 1000:8.1f} ms   (min {min(xs) * 1000:6.1f}, max {max(xs) * 1000:6.1f})"
            )
        print(f"  {'TOTAL':<20s} {total * 1000:8.1f} ms  -> {total:.2f} s/step")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = TrainConfig.from_hydra(hydra_cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(0)

    model = PinkTromboneController(cfg.model).to(device)
    loss_fn = MultiScaleLogMagSTFTLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr)

    loader = build_dataloader(
        manifest_path=cfg.data.manifest_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.data.num_workers,
        sample_rate=cfg.data.sample_rate,
        chunk_seconds=cfg.data.chunk_seconds,
        rank=0,
        world_size=1,
        epoch=0,
        seed=0,
    )
    data_iter = iter(loader)
    frame_rate = cfg.model.frame_rate

    import os

    n_steps = int(os.environ.get("BENCH_TRIALS", 6))
    warmup = int(os.environ.get("BENCH_WARMUP", 2))
    timer = Timer(device)
    print(
        f"benchmark: batch_size={cfg.batch_size}, chunk_seconds={cfg.data.chunk_seconds}, "
        f"sample_rate={cfg.data.sample_rate}, num_workers={cfg.data.num_workers}, "
        f"ir_impl={cfg.synth.ir_impl}, ir_length={cfg.synth.ir_length}"
    )
    print(f"running {n_steps} steps ({warmup} warmup)")

    for step in range(n_steps):
        timer.start("data")
        batch = next(data_iter)
        timer.stop()

        timer.start("to_device")
        wav = batch.to(device, non_blocking=True)
        wav_in = wav.unsqueeze(1)
        target = wav
        timer.stop()

        optimizer.zero_grad(set_to_none=True)

        timer.start("encoder_fwd")
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(device.type == "cuda"),
        ):
            params = model(wav_in)
        params = params.float()
        timer.stop()

        timer.start("synth_fwd")
        pred = pink_trombone_ola(
            params,
            ir_length=cfg.synth.ir_length,
            ir_impl=cfg.synth.ir_impl,
            control_rate=frame_rate,
        )
        timer.stop()

        timer.start("loss")
        S = min(pred.shape[-1], target.shape[-1])
        loss = loss_fn(pred[..., :S], target[..., :S])
        timer.stop()

        timer.start("backward")
        loss.backward()
        timer.stop()

        timer.start("optim_step")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        timer.stop()

        print(f"step {step}: loss={loss.item():.4f}")

    timer.summary(warmup=warmup)

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\nPeak GPU memory allocated: {peak:.2f} GiB")


if __name__ == "__main__":
    main()
