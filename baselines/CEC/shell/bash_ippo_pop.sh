#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
SEEDS=(0 1 2 3 4 5)


for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population.py \
      ENV_KWARGS.layout="cramped_room_9" \
      SEED="$seed" \
      WANDB_MODE=online
done


# for seed in "${SEEDS[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population.py \
#       ENV_KWARGS.layout="coord_ring_9" \
#       SEED="$seed" \
#       WANDB_MODE=online
# done


# for seed in "${SEEDS[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population.py \
#       ENV_KWARGS.layout="asymm_advantages_9" \
#       SEED="$seed" \
#       WANDB_MODE=online
# done


# for seed in "${SEEDS[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population.py \
#       ENV_KWARGS.layout="forced_coord_9" \
#       SEED="$seed" \
#       WANDB_MODE=online
# done


# for seed in "${SEEDS[@]}"; do
#     CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population.py \
#       ENV_KWARGS.layout="counter_circuit_9" \
#       SEED="$seed" \
#       WANDB_MODE=online
# done