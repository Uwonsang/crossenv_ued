#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <layout>" >&2
    echo "Example: $0 counter_circuit_9" >&2
    exit 2
fi

LAYOUT="$1"
case "${LAYOUT}" in
    cramped_room_9|asymm_advantages_9|coord_ring_9|counter_circuit_9|forced_coord_9)
        ;;
    *)
        echo "Unsupported layout: ${LAYOUT}" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPOSITORY_ROOT}"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/app/nas/models/ICRL/CEC_IDDAC/256}"
PROJECT="${PROJECT:-crossenv_ICLR}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_ID="${RUN_ID:-lr-$(date +%Y%m%d-%H%M%S)}"

for SEED in 0 1 2 3 4 5; do
    echo "Starting IDAAC fine-tuning: layout=${LAYOUT}, seed=${SEED}, run_id=${RUN_ID}"

    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
        baselines/CEC_UED/idaac_general_gradient.py \
        SEED="${SEED}" \
        NUM_ENVS=256 \
        NUM_STEPS=256 \
        TOTAL_TIMESTEPS=1e8 \
        ENV_KWARGS.layout="${LAYOUT}" \
        ENV_KWARGS.random_reset=False \
        ENV_KWARGS.check_held_out=False \
        ENV_KWARGS.shuffle_inv_and_pot=False \
        TRAIN_KWARGS.finetune=True \
        TRAIN_KWARGS.ckpt_id=0 \
        ++TRAIN_KWARGS.finetune_checkpoint_root="${CHECKPOINT_ROOT}" \
        RESUME_XPID="${RUN_ID}" \
        PROJECT="${PROJECT}" \
        WANDB_MODE="${WANDB_MODE}"
done
