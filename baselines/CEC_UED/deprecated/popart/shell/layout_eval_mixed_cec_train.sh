#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2}"
MAP_NAME="${MAP_NAME:-mixed}"
TRAIN_CEC="${TRAIN_CEC:-true}"
TRAIN_POPART="${TRAIN_POPART:-true}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running modified-wall mixed CEC training with fixed layout eval"
echo "  map: $MAP_NAME"
echo "  layouts: config layout_names"
echo "  seeds: $SEEDS"
echo "  train_cec: $TRAIN_CEC"
echo "  train_popart: $TRAIN_POPART"
echo "  wandb: $WANDB_MODE"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

if [[ "$TRAIN_CEC" == "true" ]]; then
  echo
  echo "===== Train CEC with config-driven layout eval ====="
  for seed in $SEEDS; do
    echo "----- CEC layout-eval ${MAP_NAME} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_layout_eval.py \
      SEED="$seed" \
      map_name="$MAP_NAME" \
      WANDB_MODE="$WANDB_MODE"
  done
fi

if [[ "$TRAIN_POPART" == "true" ]]; then
  echo
  echo "===== Train PopArt CEC with config-driven layout eval ====="
  for seed in $SEEDS; do
    echo "----- CEC_POPART layout-eval ${MAP_NAME} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_ippo_general_gradient_pop_layout_eval.py \
      SEED="$seed" \
      map_name="$MAP_NAME" \
      WANDB_MODE="$WANDB_MODE"
  done
fi

echo
echo "Modified-wall mixed CEC layout-eval training finished."
