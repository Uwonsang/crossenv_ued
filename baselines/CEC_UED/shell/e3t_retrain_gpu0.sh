#!/usr/bin/env bash
# E3T Dual 재학습 (conv 64/32 수정 후) - gpu0 담당: empty seed 0,1,2
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

map=empty
seeds=(0 1)

for seed in "${seeds[@]}"; do
  python3 baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py SEED="$seed" NUM_ENVS=256 map_name="$map" CKPT_TAG=with_xp_numenv256 XP_KWARGS.enabled=false WANDB_GROUP="'E3T 256 retrain ${map^^}'" WANDB_MODE=online
done

