#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

for group in "$@"; do
  python3 baselines/CEC_UED/modified_wall_procedural_xp_eval.py MODEL_GROUP="$group" WANDB_MODE=online
done

