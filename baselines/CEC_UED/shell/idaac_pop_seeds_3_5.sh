#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for seed in 3 4 5; do
  python3 baselines/CEC_UED/idaac_general_gradient_pop.py SEED="$seed" WANDB_MODE=online
done
