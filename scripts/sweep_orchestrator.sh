#!/usr/bin/env bash
# Orchestrate the full multi-spf sweep:
#   1. wait for the spf=2048 training sweep to finish
#   2. wait for the spf=1024 pitch cache, run the spf=1024 sweep
#   3. wait for the spf=512 pitch cache, run the spf=512 sweep
set -euo pipefail
cd "$(dirname "$0")/.."

CACHE_1024=manifests/pitch_cache/librilight_1000h_spf1024.npz
CACHE_512=manifests/pitch_cache/librilight_1000h_spf512.npz

wait_for_no_train() {
  while pgrep -fa "samuel.train" >/dev/null 2>&1; do
    sleep 60
  done
}

wait_for_cache() {
  local f=$1
  while [ ! -s "$f" ]; do
    sleep 120
  done
  # ensure the file isn't being written: size stable for 30s
  local prev=0 cur
  while true; do
    cur=$(stat -c %s "$f")
    if [ "$cur" -eq "$prev" ] && [ "$cur" -gt 0 ]; then
      break
    fi
    prev=$cur
    sleep 30
  done
}

echo "==> waiting for spf=2048 sweep to finish"
wait_for_no_train
echo "==> spf=2048 sweep done"

echo "==> waiting for spf=1024 cache: $CACHE_1024"
wait_for_cache "$CACHE_1024"
echo "==> spf=1024 cache ready"

bash scripts/sweep_spf1024.sh

echo "==> waiting for spf=512 cache: $CACHE_512"
wait_for_cache "$CACHE_512"
echo "==> spf=512 cache ready"

bash scripts/sweep_spf512.sh

echo "==> all sweeps finished"
