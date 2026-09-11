"""Parameter counts of FlowLM variants, for sizing the shrunken TTS runs.

Usage:
    uv run python scripts/tts/count_params.py configs/tts/b_stack8.yaml \\
        --override flow_lm.transformer.d_model=512 flow_lm.transformer.num_layers=4 ...
"""

from __future__ import annotations

import argparse

from pocket_tts.models.flow_lm import FlowLMModel
from training.args import load_args
from training.modules.builders import load_model_config


def count(model: object) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6  # ty: ignore[unresolved-attribute]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--override", nargs="*", default=[], help="dotted.path=value pairs")
    args = ap.parse_args()
    targs = load_args(args.config)
    overrides = dict(targs.model_overrides)
    for item in args.override:
        key, value = item.split("=", 1)
        overrides[key] = int(value) if value.isdigit() else value
    config = load_model_config(targs.model_config, overrides)
    flow_lm = FlowLMModel.from_pydantic_config(
        config.flow_lm, latent_dim=args.latent_dim, insert_bos_before_voice=True
    )
    d_model = config.flow_lm.transformer.d_model
    total = count(flow_lm) + d_model * args.latent_dim / 1e6  # + speaker_proj
    print(
        f"total {total:.2f}M | transformer {count(flow_lm.transformer):.2f}M | "
        f"flow head {count(flow_lm.flow_net):.2f}M | text embed {count(flow_lm.conditioner):.2f}M"
    )


if __name__ == "__main__":
    main()
