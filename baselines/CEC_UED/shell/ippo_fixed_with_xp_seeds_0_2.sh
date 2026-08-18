#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

for map in empty wall_a; do
  python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py SEED=98 NUM_ENVS=64 model_name=IPPO_baseline map_name="$map" ENV_KWARGS.random_reset=false ENV_KWARGS.check_held_out=false layout_names='[empty,wall_a]' EVAL_KWARGS.eval_interval=25 XP_KWARGS.enabled=false WANDB_MODE=online
  for seed in 0 1 2; do
    python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py SEED="$seed" NUM_ENVS=64 model_name=IPPO_baseline map_name="$map" ENV_KWARGS.random_reset=false ENV_KWARGS.check_held_out=false layout_names='[empty,wall_a]' EVAL_KWARGS.eval_interval=25 XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_MODE=online
  done
done
