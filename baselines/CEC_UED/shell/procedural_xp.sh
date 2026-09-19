#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

groups=(ippo_empty ippo_wall_a e3t_empty e3t_wall_a cec idaac_cec)

for num_envs in 64 256; do
  for group in "${groups[@]}"; do
    python3 baselines/CEC_UED/modified_wall_procedural_xp_eval.py \
      MODEL_GROUP="$group" \
      CKPT_NUM_ENVS="$num_envs" \
      WANDB_MODE=disabled
  done
done
