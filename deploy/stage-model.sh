#!/usr/bin/env bash
# Prepare a run's weights for upload to the public Hugging Face model repo that
# deploy/Dockerfile downloads at build time.
#
# Usage: deploy/stage-model.sh <run-dir>
#
# last.pt is ~42 MB, most of it optimizer state the server never reads; this
# strips it to ~14 MB. Writes deploy/model/, which is gitignored.
#
# Reads only local files, so it needs no credentials and can run on the cluster.
# Uploading is a separate step, from a machine that has a Hub write token — see
# README.md beside this.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=${1:?usage: deploy/stage-model.sh <run-dir>}
OUT=deploy/model

rm -rf "$OUT"
mkdir -p "$OUT/checkpoints"
cp "$RUN/config.json" "$OUT/"
# The Hub renders this as the model card; kept in git so re-staging cannot
# silently drop it.
cp deploy/model-card.md "$OUT/README.md"
uv run python -c '
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
torch.save({"model": ckpt["model"], "step": ckpt["step"]}, sys.argv[2])
' "$RUN/checkpoints/last.pt" "$OUT/checkpoints/last.pt"

echo "Staged into $OUT:"
du -h "$OUT/checkpoints/last.pt" "$OUT/config.json"
