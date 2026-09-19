#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

SEEDS="${SEEDS:-0 1 2 3 4 5}"
MODELS="${MODELS:-CEC DCEC}"

CEC_PARTNER="ckpts/ippo/ToyCoopNoPink/modified_wall/mixed_empty_wall_a_with_xp_numenv256/ikTrue/reset_all/cec_layout_eval/lr-20260831-062443/seed98_ckpt0_improved_updates3906.pkl"
DCEC_PARTNER="ckpts/idaac/ToyCoopNoPink/modified_wall/mixed_empty_wall_a_with_xp_numenv256/ikTrue/reset_all/lr-20260819-144828/seed98_ckpt0_improved_updates3906.pkl"

for model in $MODELS; do
  case "$model" in
    CEC)
      [[ -f "$CEC_PARTNER" ]] || { echo "Missing CEC seed98: $CEC_PARTNER" >&2; exit 1; }
      for seed in $SEEDS; do
        python3 baselines/CEC_UED/random3_cec_dual_destination_with_xp.py \
          --config-name cec_toycoop_nopink_random3_65k \
          SEED="$seed"
      done
      ;;
    DCEC)
      [[ -f "$DCEC_PARTNER" ]] || { echo "Missing DCEC seed98: $DCEC_PARTNER" >&2; exit 1; }
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
