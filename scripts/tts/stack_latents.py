"""Derive a stacked SamuelCodec latents store from an unstacked one.

Stacking k control frames into one latent is a reshape, so the store for a
``stack: k`` run follows from a ``stack: 1`` run's store without re-running
pyin and the controller. The two runs use separate data directories holding
the same audio manifest; this writes the destination's per-utterance files,
``<stem>_latents.jsonl`` and ``.meta.json``, which is what
training.train_utils.ensure_train_latents looks for.

Usage:
    uv run python scripts/tts/stack_latents.py configs/tts/a_unstacked.yaml configs/tts/b_stack8.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import safetensors.torch
import torch
from einops import rearrange
from tqdm import tqdm
from training.args import load_args
from training.modules.builders import build_codec


def latents_paths(train_jsonl: str) -> tuple[Path, Path, Path]:
    """(audio manifest, latents manifest, meta) for a config's train_jsonl."""
    manifest = Path(train_jsonl)
    if manifest.stem.endswith("_latents"):
        manifest = manifest.with_name(
            manifest.stem.removesuffix("_latents") + manifest.suffix
        )
    latents = manifest.with_name(manifest.stem + "_latents.jsonl")
    return manifest, latents, latents.with_suffix(".meta.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "src_config", help="training config of the stack=1 run (already precomputed)"
    )
    ap.add_argument("dst_config", help="training config of the stack=k run")
    args = ap.parse_args()
    src, dst = load_args(args.src_config), load_args(args.dst_config)
    src_codec, dst_codec = build_codec(src), build_codec(dst)
    k = dst_codec.stack
    assert src_codec.stack == 1, (
        f"source must be unstacked, got stack={src_codec.stack}"
    )
    assert src_codec.n_trainable == dst_codec.n_trainable

    src_manifest, src_latents, src_meta_path = latents_paths(src.data.train_jsonl)
    dst_manifest, dst_latents, dst_meta_path = latents_paths(dst.data.train_jsonl)
    assert src_manifest != dst_manifest, (
        "source and destination must be separate data dirs"
    )
    if not dst_manifest.exists():
        dst_manifest.parent.mkdir(parents=True, exist_ok=True)
        dst_manifest.write_bytes(src_manifest.read_bytes())
    assert dst_manifest.read_bytes() == src_manifest.read_bytes(), (
        "audio manifests differ"
    )
    src_meta = json.loads(src_meta_path.read_text())
    assert src_meta["codec_hash"] == src_codec.encode_hash(), "source latents are stale"

    dst_hash = dst_codec.encode_hash()
    tag = dst_hash[:8]
    out_lines = []
    for line in tqdm(src_latents.read_text().splitlines(), desc=f"stack x{k}"):
        d = json.loads(line)
        src_file = src_manifest.parent / d["latents_file"]
        dst_rel = f"latents/{tag}/{src_file.name}"
        dst_file = dst_manifest.parent / dst_rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        if not dst_file.exists():
            lat = safetensors.torch.load_file(str(src_file))["latents"]
            pad = (-lat.shape[0]) % k
            if pad:
                lat = torch.cat([lat, lat[-1:].expand(pad, -1)], dim=0)
            lat = rearrange(lat, "(t k) c -> t (k c)", k=k).contiguous()
            tmp = dst_file.with_suffix(".tmp")
            safetensors.torch.save_file({"latents": lat}, str(tmp))
            tmp.rename(dst_file)
        d["latents_file"] = dst_rel
        out_lines.append(json.dumps(d))
    dst_latents.write_text("\n".join(out_lines) + "\n")
    meta = dict(src_meta, frame_rate=dst_codec.frame_rate, codec_hash=dst_hash)
    meta["codec"] = f"{dst.codec.type} {dst.model_config}"
    dst_meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {dst_latents} ({len(out_lines)} entries, codec hash {tag})")


if __name__ == "__main__":
    main()
