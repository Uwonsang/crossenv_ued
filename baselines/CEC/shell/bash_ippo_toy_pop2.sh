#! /bin/bash
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=16 NUM_MINIBATCHES=8 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=4 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=16 NUM_MINIBATCHES=4 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=64 NUM_MINIBATCHES=2 WANDB_MODE=online
CUDA_VISIBLE_DEVICES=1 python baselines/CEC/ippo_general_population_v2.py --config-name=ippo_final_toy.yaml SEED=0 ENV_KWARGS.random_reset=True UPDATE_EPOCHS=32 NUM_MINIBATCHES=2 WANDB_MODE=online

