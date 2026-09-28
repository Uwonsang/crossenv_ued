#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

SEEDS="${SEEDS:-0 1 2 3 4 5}"
MODELS="${MODELS:-CEC DCEC}"

for model in $MODELS; do
  case "$model" in
    CEC)
      for seed in $SEEDS; do
        python3 baselines/CEC_UED/random3_cec_dual_destination_with_xp.py \
          --config-name cec_toycoop_nopink_random3_65k \
          SEED="$seed"
      done
      ;;
    DCEC)
      for seed in $SEEDS; do
        python3 baselines/CEC_UED/random3_dcec_dual_destination_with_xp.py \
          --config-name dcec_toycoop_nopink_random3_65k \
          SEED="$seed"
      done
      ;;
    *)
      echo "Unknown model: $model (expected CEC or DCEC)" >&2
      exit 2
      ;;
  esac
done
