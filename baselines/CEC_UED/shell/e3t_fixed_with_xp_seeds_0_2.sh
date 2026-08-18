#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

for map in empty wall_a; do
  python3 baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py SEED=98 NUM_ENVS=64 map_name="$map" XP_KWARGS.enabled=false WANDB_MODE=online
  for seed in 0 1 2; do
    python3 baselines/CEC_UED/modified_wall_e3t_dual_destination_with_xp.py SEED="$seed" NUM_ENVS=64 map_name="$map" XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_MODE=online
  done
done
