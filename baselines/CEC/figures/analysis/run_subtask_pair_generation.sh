#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
MODEL_ROOT="${1:-/app/nas/models/ICRL}"
MAP_CANDIDATES="${MAP_CANDIDATES:-3000}"
PAIRS_PER_SUBTASK="${PAIRS_PER_SUBTASK:-50}"

cd "${REPO_ROOT}"

families=(
  asymm_advantages
  coord_ring
  counter_circuit
  forced_coord
  cramped_room
)

for family in "${families[@]}"; do
  echo "[subtask consistency] family=${family}"
  python baselines/CEC/figures/analysis/subtask_pair_consistency.py \
    --model-root "${MODEL_ROOT}" \
    --family "${family}" \
    --map-candidates "${MAP_CANDIDATES}" \
    --pairs-per-subtask "${PAIRS_PER_SUBTASK}" \
    --prepare-only
done
