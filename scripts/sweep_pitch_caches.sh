#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=14   # leave room for the concurrent training run + dataloader workers

echo "==> precomputing spf=1024"
uv run python scripts/precompute_pitch_parallel.py \
  --manifest manifests/librilight_1000h.jsonl \
  --out manifests/pitch_cache/librilight_1000h_spf1024.npz \
  --samples-per-frame 1024 \
  --workers "$WORKERS"

echo "==> precomputing spf=512"
uv run python scripts/precompute_pitch_parallel.py \
  --manifest manifests/librilight_1000h.jsonl \
  --out manifests/pitch_cache/librilight_1000h_spf512.npz \
  --samples-per-frame 512 \
  --workers "$WORKERS"

echo "==> done"
