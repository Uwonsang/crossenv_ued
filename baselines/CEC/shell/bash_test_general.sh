#!/usr/bin/env bash

gpu="${1:-0}"

# IPPO
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=IPPO

# E3T
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=E3T

# FCP
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=FCP

# CEC (64 environments)
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=CEC_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=CEC_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=CEC_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=CEC_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=CEC_64

# CEC (previous)
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=CEC_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=CEC_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=CEC_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=CEC_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=CEC_PREV

# CEC PopArt (64 environments)
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=CEC_POP_ART_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=CEC_POP_ART_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=CEC_POP_ART_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=CEC_POP_ART_64
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=CEC_POP_ART_64

# CEC PopArt (previous)
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=CEC_POP_ART_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=CEC_POP_ART_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=CEC_POP_ART_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=CEC_POP_ART_PREV
CUDA_VISIBLE_DEVICES="${gpu}" python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=CEC_POP_ART_PREV
