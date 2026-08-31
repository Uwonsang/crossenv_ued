#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

seeds=(0 1 2 3 4 5)

python3 baselines/CEC_UED/modified_wall_idaac_general_gradient_with_xp.py SEED=98 NUM_ENVS=64 CKPT_TAG=with_xp_numenv64 XP_KWARGS.enabled=false WANDB_GROUP="'XP PARTNER IDAAC CEC 64'" WANDB_MODE=online

for seed in "${seeds[@]}"; do
  python3 baselines/CEC_UED/modified_wall_idaac_general_gradient_with_xp.py SEED="$seed" NUM_ENVS=64 CKPT_TAG=with_xp_numenv64 XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_GROUP="'IDAAC CEC 64'" WANDB_MODE=online
done
