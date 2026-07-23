#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MAPS="${MAPS:-wall_a}"
# To sweep all wall maps, run with:
# MAPS="wall_a wall_b wall_c" bash baselines/CEC_UED/shell/modified_wall_run_dual_xp_all.sh
MODELS="${MODELS:-IPPO CEC}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running Modified-Wall Dual Destination XP"
echo "  maps: $MAPS"
echo "  models: $MODELS"

for map_name in $MAPS; do
  for model in $MODELS; do
    echo
    echo "===== Fixed-task XP: ${model} on ${map_name} ====="
    python3 baselines/CEC_UED/modified_wall_dual_xp_test.py \
      model_name="$model" \
      map_name="$map_name" \
      ENV_KWARGS.random_reset=false \
      ENV_KWARGS.check_held_out=false
  done
done

echo
echo "All Modified-Wall Dual Destination XP runs finished."
