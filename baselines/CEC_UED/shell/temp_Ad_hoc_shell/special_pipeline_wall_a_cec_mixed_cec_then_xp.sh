#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"

echo "Running full wall_a + mixed CEC experiment"

SEEDS=0 MAPS=wall_a TRAIN_IPPO=true TRAIN_CEC=true \
  bash baselines/CEC_UED/shell/modified_wall_dual_baseline_train.sh "$GPU_ID"

SEEDS="0 1 2 3 4 5" MAPS=mixed TRAIN_IPPO=false TRAIN_CEC=true \
  bash baselines/CEC_UED/shell/modified_wall_dual_baseline_train.sh "$GPU_ID"

MODELS="IPPO CEC CEC_MIXED" bash baselines/CEC_UED/shell/wall_a_multi_model_xp_all.sh "$GPU_ID"

echo
echo "Full wall_a + mixed CEC experiment finished."
