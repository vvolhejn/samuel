#!/usr/bin/env bash
# Start a codec chain once the 2000h TTS run has finished (they do not fit on the GPU together).
set -u
cd "$(dirname "$0")/../.."
until tail -1 runs/tts/run_2000h.log 2>/dev/null | grep -q "\] done"; do sleep 300; done
echo "[$(date)] TTS run done; starting codec chain 2" >> runs/codec_chain.log
exec bash scripts/codec/run_chain.sh "$@"
