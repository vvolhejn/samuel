"""Loading a trained run for inference: (model, codec, step) from a run dir, an
exported run, or a Hugging Face repo holding one.

Kept apart from training.eval.librispeech so that inference (samuel.server)
does not import the evaluation's dependencies (jiwer, sphn).
"""

from pathlib import Path
from typing import Any

import huggingface_hub
import safetensors.torch
import torch

from training.args import load_args
from training.checkpointing import EMA, latest_checkpoint, load_checkpoint
from training.codec import LatentCodec
from training.modules.builders import build_models
from training.modules.model import TrainableTTS


def load_run(
    run_dir: str | Path,
    device: torch.device,
    use_ema: bool = False,
    checkpoint: str | Path | None = None,
    codec_kwargs: dict[str, Any] | None = None,
) -> tuple[TrainableTTS, LatentCodec, int]:
    """(model, codec, step) from a run dir, weights on `device`, in eval mode.

    `run_dir` may also be an exported run (`args.yaml` + `model.safetensors`, as
    published), either a local directory or a Hugging Face repo,
    `hf:<user>/<repo>[@revision]` (e.g. `hf:vvolhejn/samuel-tts`); the returned
    step is then -1. `codec_kwargs` replace entries of the run's codec kwargs,
    e.g. to hand the codec a model that is already loaded.
    """
    if str(run_dir).startswith("hf:"):
        repo_id, _, revision = str(run_dir).removeprefix("hf:").removeprefix("//").partition("@")
        run_dir = huggingface_hub.snapshot_download(
            repo_id,
            revision=revision or None,
            allow_patterns=["*.yaml", "*.safetensors", "tokenizer.model"],
        )
    run_dir = Path(run_dir)
    exported = checkpoint is None and latest_checkpoint(run_dir) is None
    if exported and (run_dir / "model.safetensors").exists():
        return _load_exported_run(run_dir, device, codec_kwargs)
    args = load_args(run_dir / "args.yaml")
    # Inference never needs the distillation teacher: it only shapes the training
    # loss, and the student's weights come from the checkpoint loaded below.
    # Building it would require the teacher checkpoint to still sit at the
    # relative path recorded at training time.
    args.distill_cfg_coef = 0.0
    args.codec.kwargs.update(codec_kwargs or {})
    model, codec, _ = build_models(args)
    ckpt = Path(checkpoint) if checkpoint else latest_checkpoint(run_dir)
    assert ckpt is not None and ckpt.exists(), f"no checkpoint in {run_dir}"
    ema = EMA(model, 1.0) if use_ema else None
    step = load_checkpoint(ckpt, model, ema=ema)
    if ema is not None:
        model.load_state_dict(ema.shadow, strict=False)
    model.to(device).eval()
    codec.to(device)
    return model, codec, step


def _load_exported_run(
    run_dir: Path, device: torch.device, codec_kwargs: dict[str, Any] | None
) -> tuple[TrainableTTS, LatentCodec, int]:
    """An exported run: `args.yaml`, `model.safetensors` and, if present,
    `tokenizer.model` (used instead of the one the model config names)."""
    args = load_args(run_dir / "args.yaml")
    args.distill_cfg_coef = 0.0
    args.codec.kwargs.update(codec_kwargs or {})
    # args.yaml records model_config relative to the repo root it was trained from.
    model_config = Path(args.model_config)
    if not model_config.is_absolute() and not model_config.exists():
        model_config = Path(__file__).resolve().parents[3] / model_config
    args.model_config = str(model_config)
    if (run_dir / "tokenizer.model").exists():
        args.model_overrides = {
            **args.model_overrides,
            "flow_lm.lookup_table.tokenizer_path": str(run_dir / "tokenizer.model"),
        }
    model, codec, _ = build_models(args)
    state = safetensors.torch.load_file(str(run_dir / "model.safetensors"))
    flow_state = {
        k.removeprefix("flow_lm."): v for k, v in state.items() if k.startswith("flow_lm.")
    }
    model.flow_lm.load_state_dict(flow_state, strict=True)
    model.to(device).eval()
    codec.to(device)
    return model, codec, -1
