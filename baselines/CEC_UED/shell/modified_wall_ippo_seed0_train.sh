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

echo "Training Modified-Wall IPPO seed 0 with the current wall-observation code"
echo "  maps: $MAPS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for map_name in $MAPS; do
  echo
  echo "----- IPPO ${map_name} seed 0 -----"
  python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
    SEED=0 \
    model_name=IPPO_baseline \
    map_name="$map_name" \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false \
    WANDB_MODE=online
done

echo
echo "Modified-Wall IPPO seed 0 training finished."
