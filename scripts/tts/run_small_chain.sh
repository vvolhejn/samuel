#!/usr/bin/env bash
# Sequential small-model runs, each evaluated on completion. Run from the worktree root.
set -u
cd "$(dirname "$0")/../.."
log=runs/tts/run_small.log
LS=/lustre/scwpod02/client/kyutai/datasets/LibriSpeech/test-clean
for run in "$@"; do
  echo "[$(date)] train $run" >> $log
  uv run python -m training.train configs/tts/$run.yaml >> runs/tts/$run.stdout 2>&1
  echo "[$(date)] eval $run (train exit $?)" >> $log
  uv run python -m training.eval.librispeech runs/tts/$run --librispeech-root $LS --skip-sim --use-ema --cfg 2.0 --batch-size 32 > runs/tts/$run/eval_full.log 2>&1
  echo "[$(date)] $run: $(grep -h FINAL runs/tts/$run/eval_full.log | tail -1 | cut -c1-300)" >> $log
done
echo "[$(date)] chain done" >> $log
