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

echo "Running Dual Destination baseline training"
echo "  repo: $REPO_ROOT"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  seeds: $SEEDS"
echo "  CEC training: skipped"

echo
# echo "===== 1/2 Train fixed-task IPPO population ====="
# for seed in $SEEDS; do
#   echo
#   echo "----- IPPO seed ${seed} -----"
#   python3 baselines/CEC_UED/ippo_general_dual_destination.py \
#     SEED="$seed" \
#     model_name=IPPO_baseline \
#     ENV_KWARGS.random_reset=false \
#     ENV_KWARGS.check_held_out=false
# done

# Uncomment this block when fresh CEC checkpoints are needed.
#
# echo
# echo "===== Train CEC on procedural held-out tasks ====="
# for seed in $SEEDS; do
#   echo
#   echo "----- CEC seed ${seed} -----"
#   python3 baselines/CEC_UED/ippo_general_dual_destination.py \
#     SEED="$seed" \
#     model_name=CEC \
#     ENV_KWARGS.random_reset=true \
#     ENV_KWARGS.check_held_out=true
# done

echo
echo "===== 2/2 Train FCP against IPPO population ====="
for seed in $SEEDS; do
  echo
  echo "----- FCP seed ${seed} -----"
  python3 baselines/CEC_UED/fcp_general_dual_destination.py \
    SEED="$seed"
done

echo
echo "Dual Destination baseline training finished."
