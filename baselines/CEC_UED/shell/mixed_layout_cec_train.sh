#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Training CEC on mixed empty + wall layouts"
echo "  map_name: mixed"
echo "  seeds: $SEEDS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for seed in $SEEDS; do
  echo
  echo "----- CEC mixed seed ${seed} -----"
  python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
    SEED="$seed" \
    map_name=mixed \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true \
    WANDB_MODE=online
done

echo
echo "Mixed-layout CEC training finished."
