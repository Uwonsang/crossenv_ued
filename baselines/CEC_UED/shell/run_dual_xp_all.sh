#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MODELS="${MODELS:-IPPO FCP CEC}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running Dual Destination XP"
echo "  repo: $REPO_ROOT"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  models: $MODELS"

for model in $MODELS; do
  echo
  echo "===== Fixed-task XP: ${model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false
done

for model in $MODELS; do
  echo
  echo "===== Procedural held-out XP: ${model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true
done

DIAGNOSTIC_PAIRS=(
  "FCP IPPO_POP"
  "IPPO_POP FCP"
)

for pair in "${DIAGNOSTIC_PAIRS[@]}"; do
  read -r model partner_model <<< "$pair"
  echo
  echo "===== Fixed-task diagnostic XP: ${model} x ${partner_model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    partner_model_name="$partner_model" \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false
done

for pair in "${DIAGNOSTIC_PAIRS[@]}"; do
  read -r model partner_model <<< "$pair"
  echo
  echo "===== Procedural held-out diagnostic XP: ${model} x ${partner_model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    partner_model_name="$partner_model" \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true
done

echo
echo "All Dual Destination XP runs finished."
