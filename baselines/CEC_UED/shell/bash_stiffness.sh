#!/usr/bin/env bash

set -euo pipefail

if (( $# < 3 || $# > 4 )); then
    echo "Usage: bash $0 <gpu> <ippo|idaac|both> \"seed ...\" [\"num_envs ...\"]" >&2
    echo "Example: bash $0 0 both \"0 1\" \"32 64 128 256\"" >&2
    exit 1
fi

GPU_ID=$1
ALGORITHM=$2
LAYOUT="cramped_room_9"
read -r -a SEEDS <<< "$3"

if (( $# >= 4 )); then
    read -r -a NUM_ENVS_VALUES <<< "$4"
else
    NUM_ENVS_VALUES=(32 64 128 256)
fi

case "$ALGORITHM" in
    ippo)
        TRAIN_SCRIPTS=("baselines/CEC_UED/ippo_general_gradient.py")
        ;;
    idaac)
        TRAIN_SCRIPTS=("baselines/CEC_UED/idaac_general_gradient.py")
        ;;
    both)
        TRAIN_SCRIPTS=(
            "baselines/CEC_UED/ippo_general_gradient.py"
            "baselines/CEC_UED/idaac_general_gradient.py"
        )
        ;;
    *)
        echo "Algorithm must be one of: ippo, idaac, both." >&2
        exit 1
        ;;
esac

if (( ${#SEEDS[@]} == 0 || ${#NUM_ENVS_VALUES[@]} == 0 )); then
    echo "At least one seed and one NUM_ENVS value are required." >&2
    exit 1
fi

TOTAL_STEPS=300000000
# Keep reward-shaping and learning-rate annealing aligned with the original
# 3B-step training run while collecting only its first 300M steps.
SCHEDULE_STEPS=3000000000
PROJECT_NAME="cec_stiffness_100m"

echo "GPU: ${GPU_ID}"
echo "Algorithm: ${ALGORITHM}"
echo "Layout: ${LAYOUT}"
echo "Seeds: ${SEEDS[*]}"
echo "NUM_ENVS: ${NUM_ENVS_VALUES[*]}"
echo "Total environment steps: ${TOTAL_STEPS}"
echo "Reward/LR schedule horizon: ${SCHEDULE_STEPS} environment steps"
echo "Stiffness measurements: approximately 100 (interval computed automatically)"

for train_script in "${TRAIN_SCRIPTS[@]}"; do
    for num_envs in "${NUM_ENVS_VALUES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "Running ${train_script}: NUM_ENVS=${num_envs}, SEED=${seed}"
            CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${train_script}" \
                ENV_KWARGS.layout="${LAYOUT}" \
                NUM_ENVS="${num_envs}" \
                SEED="${seed}" \
                TOTAL_TIMESTEPS="${TOTAL_STEPS}" \
                MAX_TRAIN_STEPS="${SCHEDULE_STEPS}" \
                STIFFNESS.ENABLED=True \
                STIFFNESS.CHUNK_SIZE=16 \
                STIFFNESS.INTERVAL_ENV_STEPS=0 \
                PROJECT="${PROJECT_NAME}" \
                WANDB_MODE=online
        done
    done
done
