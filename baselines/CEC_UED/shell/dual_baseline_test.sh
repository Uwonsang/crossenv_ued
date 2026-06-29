#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MEM_FRACTION="${2:-0.80}"
IPPO_TIMESTEPS="${3:-2e8}"
FCP_TIMESTEPS="${4:-2e8}"
WANDB_MODE="${5:-online}"
MASK_FCP_LOSS="${6:-true}"
CEC_TIMESTEPS="${CEC_TIMESTEPS:-3e8}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"
export HYDRA_FULL_ERROR=1

WANDB_NAMESPACE="${WANDB_NAMESPACE:-${USER:-user}_dual_baseline}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config_${WANDB_NAMESPACE}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache_${WANDB_NAMESPACE}}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/tmp/wandb_data_${WANDB_NAMESPACE}}"
mkdir -p "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

WANDB_OVERRIDES=()
if [[ -n "${WANDB_ENTITY_OVERRIDE:-}" ]]; then
  WANDB_OVERRIDES+=(ENTITY="$WANDB_ENTITY_OVERRIDE")
fi
if [[ -n "${WANDB_PROJECT_OVERRIDE:-}" ]]; then
  WANDB_OVERRIDES+=(PROJECT="$WANDB_PROJECT_OVERRIDE")
fi

FCP_MODEL_NAME="FCP"
if [[ "$MASK_FCP_LOSS" == "true" ]]; then
  FCP_MODEL_NAME="FCP_masked"
fi

echo "Running Dual Destination baseline training"
echo "  repo: $REPO_ROOT"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  xla mem fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "  seeds: $SEEDS"
echo "  ippo timesteps: $IPPO_TIMESTEPS"
echo "  fcp timesteps: $FCP_TIMESTEPS"
echo "  cec timesteps if uncommented: $CEC_TIMESTEPS"
echo "  masked fcp loss: $MASK_FCP_LOSS"
echo "  wandb mode: $WANDB_MODE"
echo "  CEC training: skipped"
echo "  wandb config dir: $WANDB_CONFIG_DIR"
echo "  wandb cache dir: $WANDB_CACHE_DIR"
echo "  wandb data dir: $WANDB_DATA_DIR"

echo
# echo "===== 1/2 Train fixed-task IPPO population ====="
# for seed in $SEEDS; do
#   echo
#   echo "----- IPPO seed ${seed} -----"
#   python3 baselines/CEC_UED/ippo_general_dual_destination.py \
#     SEED="$seed" \
#     model_name=IPPO_baseline \
#     ENV_KWARGS.random_reset=false \
#     ENV_KWARGS.check_held_out=false \
#     TOTAL_TIMESTEPS="$IPPO_TIMESTEPS" \
#     MAX_TRAIN_STEPS="$IPPO_TIMESTEPS" \
#     WANDB_MODE="$WANDB_MODE" \
#     "${WANDB_OVERRIDES[@]}"
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
#     ENV_KWARGS.check_held_out=true \
#     TOTAL_TIMESTEPS="$CEC_TIMESTEPS" \
#     MAX_TRAIN_STEPS="$CEC_TIMESTEPS" \
#     WANDB_MODE="$WANDB_MODE" \
#     "${WANDB_OVERRIDES[@]}"
# done

echo
echo "===== 2/2 Train FCP against IPPO population ====="
for seed in $SEEDS; do
  echo
  echo "----- ${FCP_MODEL_NAME} seed ${seed} -----"
  python3 baselines/CEC_UED/fcp_general_dual_destination.py \
    SEED="$seed" \
    model_name="$FCP_MODEL_NAME" \
    FCP_KWARGS.mask_frozen_agent_loss="$MASK_FCP_LOSS" \
    FCP_PARTNER_ROOT=ckpts/ippo/ToyCoop/ikFalse/reset_all \
    TOTAL_TIMESTEPS="$FCP_TIMESTEPS" \
    MAX_TRAIN_STEPS="$FCP_TIMESTEPS" \
    WANDB_MODE="$WANDB_MODE" \
    "${WANDB_OVERRIDES[@]}"
done

echo
echo "Dual Destination baseline training finished."
