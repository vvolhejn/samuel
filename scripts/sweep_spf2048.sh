#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run() {
  local name=$1; shift
  echo "==> starting $name"
  uv run python -m samuel.train run.name="$name" "$@" \
    > "logs/${name}.log" 2>&1
  echo "==> finished $name"
}

run e1-mel-spf2048      loss.mfcc=0.0 loss.mel=1.0 loss.stft=0.0 loss.ms_mfcc=0.0 loss.ms_mel=0.0
run e1-stft-spf2048     loss.mfcc=0.0 loss.mel=0.0 loss.stft=1.0 loss.ms_mfcc=0.0 loss.ms_mel=0.0 data.num_workers=0
run e1-msmfcc-spf2048   loss.mfcc=0.0 loss.mel=0.0 loss.stft=0.0 loss.ms_mfcc=1.0 loss.ms_mel=0.0 data.num_workers=0
run e1-msmel-spf2048    loss.mfcc=0.0 loss.mel=0.0 loss.stft=0.0 loss.ms_mfcc=0.0 loss.ms_mel=1.0 data.num_workers=0
