#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MAPS="${MAPS:-wall_a}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Training missing Modified-Wall CEC seed 0"
echo "  maps: $MAPS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for map_name in $MAPS; do
  echo
  echo "----- CEC ${map_name} seed 0 -----"
  python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
    SEED=0 \
    map_name="$map_name" \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true \
    WANDB_MODE=online
done

echo
echo "Missing Modified-Wall CEC seed 0 training finished."
