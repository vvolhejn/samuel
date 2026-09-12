#!/usr/bin/env bash
# Sequential controller (codec) shrink runs: each argument is "<name>|<hydra overrides>".
# Shared recipe: from scratch, 30k steps, half-strength trajectory penalties ramped in.
# From-scratch runs sometimes fall into the constant-trajectory collapse basin, which
# is fully visible at the step-1000 eval (param_variation ~1e-4): such a run is killed
# and retried with the next seed, up to MAX_ATTEMPTS.
set -u
cd "$(dirname "$0")/../.."
log=runs/codec_chain.log
common="optim.max_steps=30000 loss.smooth=0.15 loss.accel=0.15 loss.rest=0.01 loss.reg_start_steps=5000 loss.reg_ramp_steps=5000 run.runs_root=runs/codec"
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
PVAR_MIN=${PVAR_MIN:-0.002}
mkdir -p runs/codec

pvar_at_1000() {  # prints the step-1000 eval pvar of a stdout file, or nothing
  tr '\r' '\n' < "$1" | grep -o "\[eval\] step=1000 .*pvar=[0-9.e-]*" | tail -1 | sed 's/.*pvar=//'
}

kill_tree() {  # the training process and its dataloader workers, then wait for the GPU memory
  pkill -9 -P "$1" 2>/dev/null; kill -9 "$1" 2>/dev/null
  for _ in $(seq 1 60); do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$1" || break
    sleep 5
  done
}

for spec in "$@"; do
  name="${spec%%|*}"; overrides="${spec#*|}"
  for attempt in $(seq 1 $MAX_ATTEMPTS); do
    seed=$attempt
    out=runs/codec/$name.stdout
    [ $attempt -gt 1 ] && out=runs/codec/$name.seed$seed.stdout
    echo "[$(date)] start $name seed=$seed: $overrides" >> $log
    # The venv python directly (not `uv run`), so $pid is the training process itself.
    .venv/bin/python -m samuel.train run.name=$name run.seed=$seed $common $overrides > $out 2>&1 &
    pid=$!
    collapsed=0
    while kill -0 $pid 2>/dev/null; do
      sleep 60
      pv=$(pvar_at_1000 $out)
      if [ -n "$pv" ]; then
        if awk -v p="$pv" -v m="$PVAR_MIN" 'BEGIN{exit !(p < m)}'; then
          echo "[$(date)] $name seed=$seed collapsed at step 1000 (pvar=$pv), killing" >> $log
          kill_tree $pid; collapsed=1
        fi
        break
      fi
    done
    if [ $collapsed -eq 1 ]; then continue; fi
    wait $pid; rc=$?
    if [ $rc -ne 0 ] && [ -z "$(pvar_at_1000 $out)" ]; then
      echo "[$(date)] $name seed=$seed died before step 1000 (exit $rc), retrying" >> $log
      kill_tree $pid; continue
    fi
    echo "[$(date)] end $name seed=$seed (exit $rc): $(tr '\r' '\n' < $out | grep '\[eval\]' | tail -1 | cut -c1-200)" >> $log
    break
  done
done
echo "[$(date)] chain done" >> $log
