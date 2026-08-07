#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1}"
NUM_ENVS_LIST="${NUM_ENVS_LIST:-256 128 64}"
TRAIN_MAP="${TRAIN_MAP:-mixed}"
LAYOUT_LIST="${LAYOUT_LIST:-[empty,wall_a,wall_b,wall_c]}"
XP_MAPS="${XP_MAPS:-empty wall_a wall_b wall_c}"
XP_MODELS="${XP_MODELS:-CEC_MIXED CEC_POPART_MIXED}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
DEBUG_GIF_MAX_PAIRS="${DEBUG_GIF_MAX_PAIRS:-9999}"
DEBUG_GIF_ONLY_CROSS_PLAY="${DEBUG_GIF_ONLY_CROSS_PLAY:-false}"
TRAIN_CEC="${TRAIN_CEC:-true}"
TRAIN_POPART="${TRAIN_POPART:-true}"
RUN_XP="${RUN_XP:-true}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running layout-eval CEC/PopArt NUM_ENVS sweep + XP"
echo "  num envs: $NUM_ENVS_LIST"
echo "  seeds: $SEEDS"
echo "  train map: $TRAIN_MAP"
echo "  train layouts: $LAYOUT_LIST"
echo "  xp maps: $XP_MAPS"
echo "  xp models: $XP_MODELS"
echo "  eval interval: $EVAL_INTERVAL"
echo "  debug gif max pairs: $DEBUG_GIF_MAX_PAIRS"
echo "  debug gif only cross play: $DEBUG_GIF_ONLY_CROSS_PLAY"
echo "  train_cec: $TRAIN_CEC"
echo "  train_popart: $TRAIN_POPART"
echo "  run_xp: $RUN_XP"
echo "  wandb: $WANDB_MODE"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for num_envs in $NUM_ENVS_LIST; do
  run_suffix="numenv${num_envs}"

  echo
  echo "=============================================="
  echo "NUM_ENVS=${num_envs}"
  echo "=============================================="

  if [[ "$TRAIN_CEC" == "true" ]]; then
    echo
    echo "===== Train CEC layout-eval NUM_ENVS=${num_envs} ====="
    for seed in $SEEDS; do
      echo "----- CEC ${TRAIN_MAP} seed ${seed} num_envs ${num_envs} -----"
      python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_layout_eval.py \
        SEED="$seed" \
        NUM_ENVS="$num_envs" \
        map_name="$TRAIN_MAP" \
        CKPT_TAG="$run_suffix" \
        layout_names="$LAYOUT_LIST" \
        EVAL_KWARGS.eval_interval="$EVAL_INTERVAL" \
        WANDB_MODE="$WANDB_MODE"
    done
  fi

  if [[ "$TRAIN_POPART" == "true" ]]; then
    echo
    echo "===== Train PopArt CEC layout-eval NUM_ENVS=${num_envs} ====="
    for seed in $SEEDS; do
      echo "----- CEC_POPART ${TRAIN_MAP} seed ${seed} num_envs ${num_envs} -----"
      python3 baselines/CEC_UED/modified_wall_ippo_general_gradient_pop_layout_eval.py \
        SEED="$seed" \
        NUM_ENVS="$num_envs" \
        map_name="$TRAIN_MAP" \
        CKPT_TAG="$run_suffix" \
        layout_names="$LAYOUT_LIST" \
        EVAL_KWARGS.eval_interval="$EVAL_INTERVAL" \
        WANDB_MODE="$WANDB_MODE"
    done
  fi

  if [[ "$RUN_XP" == "true" ]]; then
    echo
    echo "===== XP for NUM_ENVS=${num_envs} checkpoints ====="
    for eval_map in $XP_MAPS; do
      echo
      echo "----- XP on ${eval_map}, NUM_ENVS=${num_envs} -----"
      MODELS="$XP_MODELS" \
      EVAL_MAP="$eval_map" \
      SEEDS="$SEEDS" \
      RUN_SUFFIX="$run_suffix" \
      CKPT_TAG="$run_suffix" \
      DEBUG_GIF_MAX_PAIRS="$DEBUG_GIF_MAX_PAIRS" \
      DEBUG_GIF_ONLY_CROSS_PLAY="$DEBUG_GIF_ONLY_CROSS_PLAY" \
      WANDB_MODE="$WANDB_MODE" \
        bash baselines/CEC_UED/shell/wall_a_multi_model_xp_all.sh "$GPU_ID"
    done
  fi
done

echo
echo "Layout-eval CEC/PopArt NUM_ENVS sweep + XP finished."
