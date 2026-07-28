#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
export SEEDS

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running Dual Destination training + XP"
echo "  repo: $REPO_ROOT"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  seeds: $SEEDS"
echo "  CEC training: skipped; XP will use existing CEC checkpoints"

echo
echo "===== 1/2 Train IPPO and FCP baselines ====="
bash baselines/CEC_UED/shell/dual_baseline_train.sh \
  "$GPU_ID"

echo
echo "===== 2/2 Run XP evaluation ====="
bash baselines/CEC_UED/shell/run_dual_xp_all.sh \
  "$GPU_ID"

echo
echo "Dual Destination IPPO/FCP training and XP finished."
