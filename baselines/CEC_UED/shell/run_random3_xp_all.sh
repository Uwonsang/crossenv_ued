#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

for model_group in \
  ippo_empty \
  ippo_wall_a \
  e3t_empty \
  e3t_wall_a \
  cec_random3 \
  dcec_random3
do
  python3 baselines/CEC_UED/random3_procedural_xp_eval.py \
    MODEL_GROUP="${model_group}" \
    CKPT_NUM_ENVS=256 \
    WANDB_MODE=disabled
done
