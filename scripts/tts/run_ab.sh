#!/usr/bin/env bash
# Proof-of-concept pair: precompute the unstacked store once, derive the
# stacked store from it, train B (cheap) then A. Run from the worktree root.
set -u
cd "$(dirname "$0")/../.."
log=runs/tts/run_ab.log
echo "[$(date)] precompute A" >> $log
uv run python -m training.scripts.precompute_latents configs/tts/a_unstacked.yaml --batch-size 32 >> runs/tts/precompute_a.log 2>&1
echo "[$(date)] stack -> B (exit $?)" >> $log
uv run python scripts/tts/stack_latents.py configs/tts/a_unstacked.yaml configs/tts/b_stack8.yaml >> runs/tts/stack_b.log 2>&1
echo "[$(date)] train B (exit $?)" >> $log
uv run python -m training.train configs/tts/b_stack8.yaml >> runs/tts/b_stack8.stdout 2>&1
echo "[$(date)] train A (exit $?)" >> $log
uv run python -m training.train configs/tts/a_unstacked.yaml >> runs/tts/a_unstacked.stdout 2>&1
echo "[$(date)] done (exit $?)" >> $log
