#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py --config-name ippo_overcooked_CEC_dual_destination_with_xp SEED=3 NUM_ENVS=64 map_name=mixed CKPT_TAG=with_xp_numenv64 layout_names='[empty,wall_a]' TOY_HELDOUT_NUM=100 EVAL_KWARGS.eval_interval=50 XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_MODE=online

bash baselines/CEC_UED/shell/procedural_xp.sh cec
