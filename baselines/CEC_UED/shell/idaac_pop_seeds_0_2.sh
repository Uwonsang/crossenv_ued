#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../../.."

for seed in 0 1 2; do
  python3 baselines/CEC_UED/idaac_general_gradient_pop.py SEED="$seed" WANDB_MODE=online
done
