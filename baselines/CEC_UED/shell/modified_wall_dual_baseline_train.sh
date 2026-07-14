#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/baselines/CEC_UED:$REPO_ROOT:${PYTHONPATH:-}"

GPU_ID="${1:-}"
SEEDS="${SEEDS:-0 1 2 3 4 5}"
MAPS="${MAPS:-wall_a}"

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running Modified-Wall Dual Destination baseline training"
echo "  maps: $MAPS"
echo "  seeds: $SEEDS"
echo "  gpu: ${CUDA_VISIBLE_DEVICES:-default}"

for map_name in $MAPS; do
  echo
  echo "===== Train IPPO population on ${map_name} ====="
  for seed in $SEEDS; do
    echo "----- IPPO ${map_name} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
      SEED="$seed" \
      model_name=IPPO_baseline \
      map_name="$map_name" \
      ENV_KWARGS.random_reset=false \
      ENV_KWARGS.check_held_out=false
  done

  # Uncomment this block when fresh modified-wall CEC checkpoints are needed.
  #
  echo
  echo "===== Train CEC on procedural held-out tasks for ${map_name} ====="
  for seed in $SEEDS; do
    echo "----- CEC ${map_name} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination.py \
      SEED="$seed" \
      map_name="$map_name" \
      ENV_KWARGS.random_reset=true \
      ENV_KWARGS.check_held_out=true
  done

  echo
  echo "===== Train FCP on ${map_name} against that IPPO population ====="
  for seed in $SEEDS; do
    echo "----- FCP ${map_name} seed ${seed} -----"
    python3 baselines/CEC_UED/modified_wall_fcp_general_dual_destination.py \
      SEED="$seed" \
      map_name="$map_name"
  done
done

echo
echo "Modified-Wall Dual Destination baseline training finished."
