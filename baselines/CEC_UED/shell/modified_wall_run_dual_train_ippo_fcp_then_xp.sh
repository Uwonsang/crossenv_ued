#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MAPS="${MAPS:-wall_a}"
SEEDS="${SEEDS:-1 2 3 4 5}"
export MAPS SEEDS

echo "Running Modified-Wall Dual Destination training + XP"
echo "  maps: $MAPS"
echo "  seeds: $SEEDS"

bash baselines/CEC_UED/shell/modified_wall_dual_baseline_train.sh \
  "$GPU_ID"

bash baselines/CEC_UED/shell/modified_wall_run_dual_xp_all.sh \
  "$GPU_ID"

echo
echo "Modified-Wall Dual Destination training and XP finished."
