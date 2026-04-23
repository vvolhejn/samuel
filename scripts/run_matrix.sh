#!/usr/bin/env bash
# Run benchmark_step.py over a matrix of (batch_size, ir_length, ir_impl)
set -u

cd "$(dirname "$0")/.."

run() {
    local b=$1 L=$2 impl=$3 trials=$4 warmup=$5
    echo "============================================================"
    echo "batch=$b ir_length=$L ir_impl=$impl trials=$trials warmup=$warmup"
    echo "============================================================"
    BENCH_TRIALS=$trials BENCH_WARMUP=$warmup \
        uv run python scripts/benchmark_step.py \
        batch_size=$b synth.ir_length=$L synth.ir_impl=$impl 2>&1 \
        | grep -E "benchmark:|TOTAL|Peak GPU|synth_fwd|backward|step [0-9]" || true
    echo
}

# Fast configs (sequential and small shapes): more trials
run 8  256  sequential 5 1
run 8  2048 sequential 5 1
run 64 256  sequential 5 1
run 64 2048 sequential 5 1

# Eig: smaller shapes
run 8  256  eig 4 1
run 8  2048 eig 3 1
run 64 256  eig 3 1

# Eig+large — likely very slow; run few trials
run 64 2048 eig 2 1
