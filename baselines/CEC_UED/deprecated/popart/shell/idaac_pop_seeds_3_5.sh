#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for seed in 3 4 5; do
  python3 baselines/CEC_UED/modified_wall_idaac_general_gradient_pop.py SEED="$seed" NUM_ENVS=256 EVAL_KWARGS.eval_interval=25 XP_KWARGS.enabled=true XP_KWARGS.partner_seed=98 WANDB_MODE=online
done
