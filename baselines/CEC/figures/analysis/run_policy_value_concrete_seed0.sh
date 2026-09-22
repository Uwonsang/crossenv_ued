#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_ROOT="${1:-/app/nas/models/ICRL}"

cd "${REPO_ROOT}"

families=(
  asymm_advantages
  coord_ring
  counter_circuit
  forced_coord
  cramped_room
)

for family in "${families[@]}"; do
  echo "[concrete seed0] family=${family}"
  python baselines/CEC/figures/analysis/policy_value_concrete_example.py \
    --model-root "${MODEL_ROOT}" \
    --models cec_64 dcec_64 \
    --seeds 0 \
    --family "${family}" \
    --map-candidates 500 \
    --num-pairs 10
done
