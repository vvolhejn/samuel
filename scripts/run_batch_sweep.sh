#!/usr/bin/env bash
# Sweep batch size for sequential, L=256
set -u

cd "$(dirname "$0")/.."

run() {
    local b=$1
    echo "============================================================"
    echo "batch=$b ir_length=256 ir_impl=sequential"
    echo "============================================================"
    BENCH_TRIALS=5 BENCH_WARMUP=1 \
        uv run python scripts/benchmark_step.py \
        batch_size=$b synth.ir_length=256 synth.ir_impl=sequential 2>&1 \
        | grep -E "benchmark:|TOTAL|Peak GPU|OutOfMemory|CUDA error" || true
    echo
}

for b in 16 32 64 128 192 256; do
    run $b
done
