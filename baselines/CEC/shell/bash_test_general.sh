#! /bin/bash
gpu=$0

# IPPO
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=IPPO 
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=IPPO
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=IPPO

# E3T
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=E3T
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=E3T

# FCP
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=FCP
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=FCP

# CEC
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=CEC
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=CEC
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=CEC
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=CEC
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=CEC

# CEC pop_art
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=asymm_advantages_9 NUM_MODELS=3 model_name=CEC_POP_ART
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=coord_ring_9 NUM_MODELS=3 model_name=CEC_POP_ART
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=counter_circuit_9 NUM_MODELS=3 model_name=CEC_POP_ART
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=cramped_room_9 NUM_MODELS=3 model_name=CEC_POP_ART
CUDA_VISIBLES_DEVICES=${gpu} python baselines/CEC/test_general.py ENV_KWARGS.layout=forced_coord_9 NUM_MODELS=3 model_name=CEC_POP_ART
