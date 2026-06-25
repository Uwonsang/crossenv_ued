#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
MEM_FRACTION="${2:-0.20}"
NUM_TRAJS="${3:-100}"
WANDB_MODE="${4:-online}"
DEBUG_GIFS="${5:-false}"
DEBUG_MAX_PAIRS="${6:-4}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"
export HYDRA_FULL_ERROR=1

WANDB_NAMESPACE="${WANDB_NAMESPACE:-${USER:-user}_dual_xp}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config_${WANDB_NAMESPACE}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache_${WANDB_NAMESPACE}}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/tmp/wandb_data_${WANDB_NAMESPACE}}"
mkdir -p "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running Dual Destination XP"
echo "  repo: $REPO_ROOT"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"
echo "  xla mem fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "  num trajs per pair: $NUM_TRAJS"
echo "  wandb mode: $WANDB_MODE"
echo "  wandb config dir: $WANDB_CONFIG_DIR"
echo "  wandb cache dir: $WANDB_CACHE_DIR"
echo "  wandb data dir: $WANDB_DATA_DIR"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "  wandb auth: WANDB_API_KEY env"
else
  echo "  wandb auth: existing wandb login/netrc, if available"
fi
WANDB_OVERRIDES=()
if [[ -n "${WANDB_ENTITY_OVERRIDE:-}" ]]; then
  WANDB_OVERRIDES+=(ENTITY="$WANDB_ENTITY_OVERRIDE")
  echo "  wandb entity override: $WANDB_ENTITY_OVERRIDE"
fi
if [[ -n "${WANDB_PROJECT_OVERRIDE:-}" ]]; then
  WANDB_OVERRIDES+=(PROJECT="$WANDB_PROJECT_OVERRIDE")
  echo "  wandb project override: $WANDB_PROJECT_OVERRIDE"
fi
echo "  debug gifs: $DEBUG_GIFS"
echo "  debug gif pairs per run: $DEBUG_MAX_PAIRS"

MODELS=(IPPO FCP CEC)

for model in "${MODELS[@]}"; do
  echo
  echo "===== Fixed-task XP: ${model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    TEST_KWARGS.num_trajs="$NUM_TRAJS" \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false \
    DEBUG_GIFS.enabled="$DEBUG_GIFS" \
    DEBUG_GIFS.max_pairs="$DEBUG_MAX_PAIRS" \
    WANDB_MODE="$WANDB_MODE" \
    "${WANDB_OVERRIDES[@]}"
done

for model in "${MODELS[@]}"; do
  echo
  echo "===== Procedural held-out XP: ${model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    TEST_KWARGS.num_trajs="$NUM_TRAJS" \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true \
    DEBUG_GIFS.enabled="$DEBUG_GIFS" \
    DEBUG_GIFS.max_pairs="$DEBUG_MAX_PAIRS" \
    WANDB_MODE="$WANDB_MODE" \
    "${WANDB_OVERRIDES[@]}"
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
    TEST_KWARGS.num_trajs="$NUM_TRAJS" \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false \
    DEBUG_GIFS.enabled="$DEBUG_GIFS" \
    DEBUG_GIFS.max_pairs="$DEBUG_MAX_PAIRS" \
    WANDB_MODE="$WANDB_MODE" \
    "${WANDB_OVERRIDES[@]}"
done

for pair in "${DIAGNOSTIC_PAIRS[@]}"; do
  read -r model partner_model <<< "$pair"
  echo
  echo "===== Procedural held-out diagnostic XP: ${model} x ${partner_model} ====="
  python3 baselines/CEC_UED/dual_xp_test.py \
    model_name="$model" \
    partner_model_name="$partner_model" \
    TEST_KWARGS.num_trajs="$NUM_TRAJS" \
    ENV_KWARGS.random_reset=true \
    ENV_KWARGS.check_held_out=true \
    DEBUG_GIFS.enabled="$DEBUG_GIFS" \
    DEBUG_GIFS.max_pairs="$DEBUG_MAX_PAIRS" \
    WANDB_MODE="$WANDB_MODE" \
    "${WANDB_OVERRIDES[@]}"
done

echo
echo "All Dual Destination XP runs finished."
