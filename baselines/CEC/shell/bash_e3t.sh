#! /bin/bash
#layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

# layout=$1
# gpu=$2
# SEEDS=(0 1 2 3 4 5)
# for seed in "${SEEDS[@]}"; do
#     CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC/e3t.py \
#       ENV_KWARGS.layout=${layout} \
#       SEED=$seed \
#       WANDB_MODE=online
# done

SEEDS=(0 1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py \
      ENV_KWARGS.layout=asymm_advantages_9 \
      SEED=$seed \
      WANDB_MODE=online
done

SEEDS=(0 1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py \
      ENV_KWARGS.layout=coord_ring_9 \
      SEED=$seed \
      WANDB_MODE=online
done

SEEDS=(0 1 2 3 4 5)
for seed in "${SEEDS[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python baselines/CEC/e3t.py \
      ENV_KWARGS.layout=forced_coord_9 \
      SEED=$seed \
      WANDB_MODE=online
done