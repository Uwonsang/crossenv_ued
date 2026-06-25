#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

WANDB_NAMESPACE="${WANDB_NAMESPACE:-${USER:-user}_dual_baseline}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-/tmp/wandb_config_${WANDB_NAMESPACE}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/tmp/wandb_cache_${WANDB_NAMESPACE}}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-/tmp/wandb_data_${WANDB_NAMESPACE}}"
mkdir -p "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR"

echo "W&B dirs:"
echo "  config: $WANDB_CONFIG_DIR"
echo "  cache:  $WANDB_CACHE_DIR"
echo "  data:   $WANDB_DATA_DIR"
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "  auth:   WANDB_API_KEY env"
else
  echo "  auth:   existing wandb login/netrc, if available"
fi

WANDB_OVERRIDES=()
if [[ -n "${WANDB_ENTITY_OVERRIDE:-}" ]]; then
  WANDB_OVERRIDES+=(ENTITY="$WANDB_ENTITY_OVERRIDE")
  echo "  entity override:  $WANDB_ENTITY_OVERRIDE"
fi
if [[ -n "${WANDB_PROJECT_OVERRIDE:-}" ]]; then
  WANDB_OVERRIDES+=(PROJECT="$WANDB_PROJECT_OVERRIDE")
  echo "  project override: $WANDB_PROJECT_OVERRIDE"
fi

# 1. Figure 3의 IPPO baseline / FCP partner population
# for s in 0 1 2 3 4 5; do
#   python3 baselines/CEC_UED/ippo_general_dual_destination.py \
#     SEED=$s \
#     model_name=IPPO_baseline \
#     ENV_KWARGS.random_reset=false \
#     ENV_KWARGS.check_held_out=false \
#     TOTAL_TIMESTEPS=1.536e8 \
#     MAX_TRAIN_STEPS=1.536e8 \
#     "${WANDB_OVERRIDES[@]}"
# done

# # 2. Figure 3의 CEC
# for s in 0 1 2 3 4 5; do
#   python3 baselines/CEC_UED/ippo_general_dual_destination.py \
#     SEED=$s \
#     model_name=CEC \
#     ENV_KWARGS.random_reset=true \
#     ENV_KWARGS.check_held_out=true \
#     "${WANDB_OVERRIDES[@]}"
# done

# 3. Figure 3의 FCP
for s in 2 3 4 5; do
  python3 baselines/CEC_UED/fcp_general_dual_destination.py \
    SEED=$s \
    model_name=FCP \
    FCP_PARTNER_ROOT=ckpts/ippo/ToyCoop/ikFalse/reset_all \
    "${WANDB_OVERRIDES[@]}"
done
