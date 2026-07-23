#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MODELS="${MODELS:-IPPO CEC CEC_MIXED}"
EVAL_MAP="${EVAL_MAP:-wall_a}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running wall_a XP for IPPO, CEC, and mixed-layout CEC"
echo "  eval map: $EVAL_MAP"
echo "  models: $MODELS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

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
      WANDB_MODE=online
  done
done

echo
echo "wall_a three-way XP finished."
