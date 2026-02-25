#! /bin/bash

layouts=(cramped_room_9 coord_ring_9 asymm_advantages_9 forced_coord_9 counter_circuit_9)

for layout in "${layouts[@]}"; do
  CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population.py \
    ENV_KWARGS.layout="$layout" \
    SEED=0 \
    WANDB_MODE=online
done