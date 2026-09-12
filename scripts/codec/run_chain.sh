#!/usr/bin/env bash
# Sequential controller (codec) shrink runs: each argument is "<name>|<hydra overrides>".
# Shared recipe: from scratch, 30k steps, seed 1, half-strength trajectory penalties ramped in.
set -u
cd "$(dirname "$0")/../.."
log=runs/codec_chain.log
common="optim.max_steps=30000 run.seed=1 loss.smooth=0.15 loss.accel=0.15 loss.rest=0.01 loss.reg_ramp_steps=10000 run.runs_root=runs/codec"
mkdir -p runs/codec
for spec in "$@"; do
  name="${spec%%|*}"; overrides="${spec#*|}"
  echo "[$(date)] start $name: $overrides" >> $log
  uv run python -m samuel.train run.name=$name $common $overrides > runs/codec/$name.stdout 2>&1
  echo "[$(date)] end $name (exit $?): $(grep -h 'eval/wer' runs/codec/$name.stdout | tail -1 | cut -c1-200)" >> $log
done
echo "[$(date)] chain done" >> $log
