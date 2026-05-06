#!/usr/bin/env bash
# Run spf=1024 then spf=512 sweeps. Caches already exist, so no waiting.
set -uo pipefail
cd "$(dirname "$0")/.."

bash scripts/sweep_spf1024.sh
bash scripts/sweep_spf512.sh

echo "==> all sweeps finished"
