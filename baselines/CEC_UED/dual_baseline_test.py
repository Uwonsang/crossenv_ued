#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# 1. Figure 3의 IPPO baseline / FCP partner population
for s in 0 1 2 3 4 5; do
  python3 baselines/CEC_UED/ippo_general_dual_destination.py \
    SEED=$s \
    model_name=IPPO_baseline \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false \
    TOTAL_TIMESTEPS=1.536e8 \
    MAX_TRAIN_STEPS=1.536e8
done

# # 2. Figure 3의 CEC
# for s in 0 1 2 3 4 5; do
#   python3 baselines/CEC_UED/ippo_general_dual_destination.py \
#     SEED=$s \
#     model_name=CEC \
#     ENV_KWARGS.random_reset=true \
#     ENV_KWARGS.check_held_out=true
# done

# 3. Figure 3의 FCP
for s in 0 1 2 3 4 5; do
  python3 baselines/CEC_UED/fcp_general_dual_destination.py \
    SEED=$s \
    model_name=FCP \
    FCP_PARTNER_ROOT=ckpts/ippo/ToyCoop/ikFalse/reset_all
done
