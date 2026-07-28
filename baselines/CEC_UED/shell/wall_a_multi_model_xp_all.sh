#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MODELS="${MODELS:-IPPO CEC CEC_MIXED CEC_POPART_MIXED}"
EVAL_MAP="${EVAL_MAP:-wall_a}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
WANDB_MODE="${WANDB_MODE:-online}"
SEED_LIST="[${SEEDS// /,}]"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running wall_a XP for modified-wall model population"
echo "  eval map: $EVAL_MAP"
echo "  models: $MODELS"
echo "  seeds: $SEEDS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  wandb: $WANDB_MODE"

for model in $MODELS; do
  for partner in $MODELS; do
    echo
    echo "===== Fixed-task XP: ${model} x ${partner} on ${EVAL_MAP} ====="
    python3 baselines/CEC_UED/modified_wall_dual_xp_test.py \
      model_name="$model" \
      partner_model_name="$partner" \
      map_name="$EVAL_MAP" \
      ENV_KWARGS.random_reset=false \
      ENV_KWARGS.check_held_out=false \
      DEBUG_GIFS.enabled=true \
      SEEDS="$SEED_LIST" \
      PARTNER_SEEDS="$SEED_LIST" \
      WANDB_MODE="$WANDB_MODE"
  done
done

echo
echo "wall_a multi-model XP finished."
