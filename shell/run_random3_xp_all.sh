#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

groups=(
  ippo_empty
  ippo_wall_a
  e3t_empty
  e3t_wall_a
  cec_random3
  dcec_random3
)

for group in "${groups[@]}"; do
  python3 baselines/CEC_UED/random3_procedural_xp_eval.py \
    MODEL_GROUP="$group" \
    WANDB_MODE=disabled
done
