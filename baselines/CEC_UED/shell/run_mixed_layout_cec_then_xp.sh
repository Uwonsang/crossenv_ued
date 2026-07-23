#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"

bash baselines/CEC_UED/shell/mixed_layout_cec_train.sh "$GPU_ID"
bash baselines/CEC_UED/shell/mixed_layout_cec_xp_all.sh "$GPU_ID"
