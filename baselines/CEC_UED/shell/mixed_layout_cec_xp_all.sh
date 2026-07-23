#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
EVAL_MAPS="${EVAL_MAPS:-empty wall_a}"
# To evaluate all wall maps, run with:
# EVAL_MAPS="empty wall_a wall_b wall_c" bash baselines/CEC_UED/shell/mixed_layout_cec_xp_all.sh

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running mixed-layout CEC XP on fixed evaluation maps"
echo "  trained model root: mixed"
echo "  eval maps: $EVAL_MAPS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for eval_map in $EVAL_MAPS; do
  echo
  echo "===== XP: mixed CEC evaluated on ${eval_map} ====="
  python3 baselines/CEC_UED/modified_wall_dual_xp_test.py \
    model_name=CEC \
    map_name="$eval_map" \
    MODEL_ROOT=ckpts/ippo/ToyCoop/modified_wall/mixed \
    ENV_KWARGS.random_reset=false \
    ENV_KWARGS.check_held_out=false
done

echo
echo "Mixed-layout CEC XP finished."
