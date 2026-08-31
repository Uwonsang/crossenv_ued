#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

seeds=(0 1 2 3 4 5)

for map in empty wall_a; do
  group="E3T 256 ${map^^}"
  partner_group="XP PARTNER E3T 256 ${map^^}"
  python3 baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py SEED=98 NUM_ENVS=256 map_name="$map" CKPT_TAG=with_xp_numenv256 XP_KWARGS.enabled=false WANDB_GROUP="'$partner_group'" WANDB_MODE=online
  for seed in "${seeds[@]}"; do
    python3 baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py SEED="$seed" NUM_ENVS=256 map_name="$map" CKPT_TAG=with_xp_numenv256 XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_GROUP="'$group'" WANDB_MODE=online
  done
done
