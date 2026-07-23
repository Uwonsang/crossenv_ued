#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"

echo "Running full wall_a + mixed CEC experiment"

bash baselines/CEC_UED/shell/modified_wall_ippo_seed0_train.sh "$GPU_ID"
bash baselines/CEC_UED/shell/modified_wall_cec_seed0_train.sh "$GPU_ID"
bash baselines/CEC_UED/shell/mixed_layout_cec_train.sh "$GPU_ID"
bash baselines/CEC_UED/shell/wall_a_three_way_xp_all.sh "$GPU_ID"

echo
echo "Full wall_a + mixed CEC experiment finished."
