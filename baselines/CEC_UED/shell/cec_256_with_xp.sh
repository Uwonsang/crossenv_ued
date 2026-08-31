#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

seeds=(0 1 2 3 4 5)

python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py --config-name ippo_overcooked_CEC_dual_destination_with_xp SEED=98 NUM_ENVS=256 map_name=mixed CKPT_TAG=with_xp_numenv256 layout_names='[empty,wall_a]' TOY_HELDOUT_NUM=100 EVAL_KWARGS.eval_interval=50 XP_KWARGS.enabled=false WANDB_GROUP="'XP PARTNER ONLY CEC 256'" WANDB_MODE=online

for seed in "${seeds[@]}"; do
  python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py --config-name ippo_overcooked_CEC_dual_destination_with_xp SEED="$seed" NUM_ENVS=256 map_name=mixed CKPT_TAG=with_xp_numenv256 layout_names='[empty,wall_a]' TOY_HELDOUT_NUM=100 EVAL_KWARGS.eval_interval=50 XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_GROUP="'ONLY CEC 256'" WANDB_MODE=online
done
