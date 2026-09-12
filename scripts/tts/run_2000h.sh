#!/usr/bin/env bash
# Last TTS run: 7M model on the 2000h store. Waits for the 2000h precompute and for the
# codec chain to be idle (the GPU cannot hold both), derives the stacked store, trains, evals.
set -u
cd "$(dirname "$0")/../.."
log=runs/tts/run_2000h.log
LS=/lustre/scwpod02/client/kyutai/datasets/LibriSpeech/test-clean
echo "[$(date)] waiting for precompute + codec chain" >> $log
until [ -f data/tts/hifitts2_2000h/train_aligned_latents.meta.json ]; do sleep 300; done
echo "[$(date)] precompute done" >> $log
until tail -1 runs/codec_chain.log | grep -q "chain done"; do sleep 300; done
echo "[$(date)] codec chain idle; stacking" >> $log
uv run python scripts/tts/stack_latents.py configs/tts/base_2000h.yaml configs/tts/base_2000h_s8.yaml > runs/tts/stack_2000h.log 2>&1
echo "[$(date)] stack exit $?; train s_d256_2000h" >> $log
.venv/bin/python -m training.train configs/tts/s_d256_2000h.yaml >> runs/tts/s_d256_2000h.stdout 2>&1
echo "[$(date)] train exit $?; eval" >> $log
.venv/bin/python -m training.eval.librispeech runs/tts/s_d256_2000h --librispeech-root $LS --skip-sim --use-ema --cfg 2.0 --batch-size 32 > runs/tts/s_d256_2000h/eval_full.log 2>&1
echo "[$(date)] s_d256_2000h: $(grep -h FINAL runs/tts/s_d256_2000h/eval_full.log | tail -1 | cut -c1-300)" >> $log
echo "[$(date)] done" >> $log
