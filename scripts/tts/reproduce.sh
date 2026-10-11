#!/usr/bin/env bash
# Train and evaluate the published TTS model (hf:vvolhejn/samuel-tts): the 7M FlowLM on
# the 2000 h HiFiTTS-2 set, Samuel latents stacked 8 frames per step. Run from the repo root.
#
# Expects the aligned manifests (train_aligned.jsonl, eval_aligned.jsonl) from
# pocket-tts/training/scripts/prepare_data.py in both data/tts/hifitts2_2000h/ and
# data/tts/hifitts2_2000h_s8/ (see pocket-tts/training/README.md).
#
# Usage: scripts/tts/reproduce.sh <LibriSpeech test-clean dir>
set -eu
cd "$(dirname "$0")/../.."
librispeech=$1
run=runs/tts/s_d256_2000h

# Encode the training audio once with Samuel (pyin + controller), unstacked...
uv run python -m training.scripts.precompute_latents configs/tts/base_2000h.yaml --batch-size 32
# ...and derive the stack-8 store from it, which is a reshape.
uv run python scripts/tts/stack_latents.py configs/tts/base_2000h.yaml configs/tts/base_2000h_s8.yaml
uv run python -m training.train configs/tts/s_d256_2000h.yaml
uv run python -m training.eval.librispeech $run --librispeech-root "$librispeech" \
    --skip-sim --use-ema --cfg 2.0 --batch-size 32
