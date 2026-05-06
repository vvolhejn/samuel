#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

run() {
  local name=$1; shift
  echo "==> starting $name"
  if uv run python -m samuel.train run.name="$name" \
    model.samples_per_frame=512 \
    batch_size=32 \
    data.pitch_cache_path=manifests/pitch_cache/librilight_1000h_spf512.npz \
    "$@" \
    > "logs/${name}.log" 2>&1; then
    echo "==> finished $name"
  else
    echo "==> FAILED $name (continuing)"
  fi
}

run e1-mfcc-spf512      loss.mfcc=1.0 loss.mel=0.0 loss.stft=0.0 loss.ms_mfcc=0.0 loss.ms_mel=0.0
run e1-mel-spf512       loss.mfcc=0.0 loss.mel=1.0 loss.stft=0.0 loss.ms_mfcc=0.0 loss.ms_mel=0.0
run e1-stft-spf512      loss.mfcc=0.0 loss.mel=0.0 loss.stft=1.0 loss.ms_mfcc=0.0 loss.ms_mel=0.0 data.num_workers=0
run e1-msmfcc-spf512    loss.mfcc=0.0 loss.mel=0.0 loss.stft=0.0 loss.ms_mfcc=1.0 loss.ms_mel=0.0 data.num_workers=0
run e1-msmel-spf512     loss.mfcc=0.0 loss.mel=0.0 loss.stft=0.0 loss.ms_mfcc=0.0 loss.ms_mel=1.0 data.num_workers=0
