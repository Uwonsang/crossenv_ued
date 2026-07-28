#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
WANDB_MODE="${WANDB_MODE:-online}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running mixed-layout PopArt CEC + wall_a XP"
echo "  popart seeds: $SEEDS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  wandb: $WANDB_MODE"

# Already completed:
# echo
# echo "===== Supplement wall_a IPPO/CEC seed0 ====="
# SEEDS=0 MAPS=wall_a TRAIN_IPPO=true TRAIN_CEC=true WANDB_MODE="$WANDB_MODE" \
#   bash baselines/CEC_UED/shell/modified_wall_dual_baseline_train.sh "$GPU_ID"

echo
echo "===== Train mixed-layout PopArt CEC ====="
for seed in $SEEDS; do
  echo "----- CEC_POPART mixed seed ${seed} -----"
  python3 baselines/CEC_UED/modified_wall_ippo_general_gradient_pop.py \
    SEED="$seed" \
    map_name=mixed \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true \
    WANDB_MODE="$WANDB_MODE"
done

echo
echo "===== XP: IPPO, wall_a CEC, mixed CEC, mixed PopArt CEC on wall_a ====="
MODELS="IPPO CEC CEC_MIXED CEC_POPART_MIXED" WANDB_MODE="$WANDB_MODE" \
  bash baselines/CEC_UED/shell/wall_a_multi_model_xp_all.sh "$GPU_ID"

echo
echo "Mixed-layout PopArt CEC pipeline finished."
