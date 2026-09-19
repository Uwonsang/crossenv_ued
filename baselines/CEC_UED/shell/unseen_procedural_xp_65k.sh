#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

groups=(ippo_empty ippo_wall_a e3t_empty e3t_wall_a cec idaac_cec)

for group in "${groups[@]}"; do
  python3 baselines/CEC_UED/modified_wall_unseen_procedural_xp_eval.py \
    MODEL_GROUP="$group" \
    CKPT_NUM_ENVS=256
done

python3 baselines/CEC_UED/plot_unseen_procedural_xp.py
