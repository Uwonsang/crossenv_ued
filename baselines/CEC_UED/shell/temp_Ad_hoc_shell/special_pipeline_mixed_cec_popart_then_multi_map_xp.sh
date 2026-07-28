#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_MAP="${TRAIN_MAP:-mixed}"
XP_MAPS="${XP_MAPS:-wall_a wall_b empty}"
XP_MODELS="${XP_MODELS:-CEC_MIXED CEC_POPART_MIXED}"
TRAIN_CEC="${TRAIN_CEC:-true}"
TRAIN_POPART="${TRAIN_POPART:-true}"
RUN_XP="${RUN_XP:-true}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running mixed CEC/PopArt retrain + multi-map XP"
echo "  train map: $TRAIN_MAP"
echo "  xp maps: $XP_MAPS"
echo "  xp models: $XP_MODELS"
echo "  seeds: $SEEDS"
echo "  train_cec: $TRAIN_CEC"
echo "  train_popart: $TRAIN_POPART"
echo "  run_xp: $RUN_XP"
echo "  wandb: $WANDB_MODE"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

if [[ "$TRAIN_CEC" == "true" ]]; then
  echo
  echo "===== Train mixed-layout CEC ====="
  for seed in $SEEDS; do
    echo "----- CEC ${TRAIN_MAP} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
      SEED="$seed" \
      map_name="$TRAIN_MAP" \
      ENV_KWARGS.random_reset=true \
      ENV_KWARGS.check_held_out=true \
      WANDB_MODE="$WANDB_MODE"
  done
fi

if [[ "$TRAIN_POPART" == "true" ]]; then
  echo
  echo "===== Train mixed-layout PopArt CEC ====="
  for seed in $SEEDS; do
    echo "----- CEC_POPART ${TRAIN_MAP} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_ippo_general_gradient_pop.py \
      SEED="$seed" \
      map_name="$TRAIN_MAP" \
      ENV_KWARGS.random_reset=true \
      ENV_KWARGS.check_held_out=true \
      WANDB_MODE="$WANDB_MODE"
  done
fi

if [[ "$RUN_XP" == "true" ]]; then
  echo
  echo "===== XP on requested eval maps ====="
  for eval_map in $XP_MAPS; do
    echo
    echo "----- XP on ${eval_map} -----"
    MODELS="$XP_MODELS" EVAL_MAP="$eval_map" SEEDS="$SEEDS" WANDB_MODE="$WANDB_MODE" \
      bash baselines/CEC_UED/shell/wall_a_multi_model_xp_all.sh "$GPU_ID"
  done
fi

echo
echo "Mixed CEC/PopArt retrain + multi-map XP finished."
