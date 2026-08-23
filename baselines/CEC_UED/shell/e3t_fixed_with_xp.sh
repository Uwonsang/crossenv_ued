#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

seeds=(3 4 5)

for map in empty wall_a; do
  for seed in "${seeds[@]}"; do
    python3 baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py SEED="$seed" NUM_ENVS=64 map_name="$map" XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_MODE=online
  done
done
