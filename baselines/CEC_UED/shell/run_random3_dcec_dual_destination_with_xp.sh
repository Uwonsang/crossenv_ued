#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )) || [[ -z "$1" || -z "$2" || "$1" == *,* ]]; then
  echo "Usage: $0 GPU 'SEED [SEED ...]'" >&2
  echo "Example: $0 0 '1 2 3 4'" >&2
  exit 2
fi

read -r -a SEEDS <<< "$2"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for seed in "${SEEDS[@]}"; do
  echo "[GPU $1] Starting SEED=$seed"
  CUDA_VISIBLE_DEVICES="$1" \
    python3 baselines/CEC_UED/random3_dcec_dual_destination_with_xp.py SEED="$seed"
done
