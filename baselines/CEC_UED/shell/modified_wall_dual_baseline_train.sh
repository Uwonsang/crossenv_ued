#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
MAPS="${MAPS:-wall_a}"
TRAIN_IPPO="${TRAIN_IPPO:-true}"
TRAIN_CEC="${TRAIN_CEC:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
# To sweep all wall maps, run with:
# MAPS="wall_a wall_b wall_c" bash baselines/CEC_UED/shell/modified_wall_dual_baseline_train.sh

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running Modified-Wall Dual Destination baseline training"
echo "  maps: $MAPS"
echo "  seeds: $SEEDS"
echo "  train_ippo: $TRAIN_IPPO"
echo "  train_cec: $TRAIN_CEC"
echo "  wandb_mode: $WANDB_MODE"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for map_name in $MAPS; do
  if [[ "$TRAIN_IPPO" == "true" ]]; then
    echo
    echo "===== Train IPPO population on ${map_name} ====="
    for seed in $SEEDS; do
      echo "----- IPPO ${map_name} seed ${seed} -----"
      python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
        SEED="$seed" \
        model_name=IPPO_baseline \
        map_name="$map_name" \
        ENV_KWARGS.random_reset=false \
        ENV_KWARGS.check_held_out=false \
        WANDB_MODE="$WANDB_MODE"
    done
  fi

  if [[ "$TRAIN_CEC" == "true" ]]; then
    echo
    echo "===== Train CEC on procedural held-out tasks for ${map_name} ====="
    for seed in $SEEDS; do
      echo "----- CEC ${map_name} seed ${seed} -----"
      python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
        SEED="$seed" \
        map_name="$map_name" \
        ENV_KWARGS.random_reset=true \
        ENV_KWARGS.check_held_out=true \
        WANDB_MODE="$WANDB_MODE"
    done
  fi

  # FCP is intentionally skipped for the current modified-wall check.
  #
  # echo
  # echo "===== Train FCP on ${map_name} against that IPPO population ====="
  # for seed in $SEEDS; do
  #   echo "----- FCP ${map_name} seed ${seed} -----"
  #   python3 baselines/CEC_UED/modified_wall_fcp_general_dual_destination.py \
  #     SEED="$seed" \
  #     map_name="$map_name"
  # done
done

echo
echo "Modified-Wall Dual Destination baseline training finished."
