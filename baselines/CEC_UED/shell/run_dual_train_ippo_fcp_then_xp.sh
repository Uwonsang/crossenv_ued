#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MEM_FRACTION="${2:-0.80}"
IPPO_TIMESTEPS="${3:-2e8}"
FCP_TIMESTEPS="${4:-2e8}"
NUM_TRAJS="${5:-100}"
WANDB_MODE="${6:-online}"
DEBUG_GIFS="${7:-false}"
DEBUG_MAX_PAIRS="${8:-4}"
MASK_FCP_LOSS="${MASK_FCP_LOSS:-true}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
export SEEDS

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"
export HYDRA_FULL_ERROR=1

WANDB_NAMESPACE="${WANDB_NAMESPACE:-${USER:-user}_dual_train_xp}"
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

echo "Running Dual Destination training + XP"
echo "  repo: $REPO_ROOT"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  xla mem fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "  seeds: $SEEDS"
echo "  ippo timesteps: $IPPO_TIMESTEPS"
echo "  fcp timesteps: $FCP_TIMESTEPS"
echo "  xp num trajs per pair: $NUM_TRAJS"
echo "  wandb mode: $WANDB_MODE"
echo "  debug gifs: $DEBUG_GIFS"
echo "  masked fcp loss: $MASK_FCP_LOSS"
echo "  CEC training: skipped; XP will use existing CEC checkpoints"

echo
echo "===== 1/2 Train IPPO and FCP baselines ====="
bash baselines/CEC_UED/shell/dual_baseline_test.sh \
  "$GPU_ID" \
  "$MEM_FRACTION" \
  "$IPPO_TIMESTEPS" \
  "$FCP_TIMESTEPS" \
  "$WANDB_MODE" \
  "$MASK_FCP_LOSS"

echo
echo "===== 2/2 Run XP evaluation ====="
bash baselines/CEC_UED/shell/run_dual_xp_all.sh \
  "$GPU_ID" \
  "$MEM_FRACTION" \
  "$NUM_TRAJS" \
  "$WANDB_MODE" \
  "$DEBUG_GIFS" \
  "$DEBUG_MAX_PAIRS"

echo
echo "Dual Destination IPPO/FCP training and XP finished."
