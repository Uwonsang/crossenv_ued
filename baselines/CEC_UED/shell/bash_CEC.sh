#! /bin/bash
gpu=$1

CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=0
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=1
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=2
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=3
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=4
CUDA_VISIBLE_DEVICES=${gpu} python baselines/CEC_UED/ippo_general.py SEED=5
