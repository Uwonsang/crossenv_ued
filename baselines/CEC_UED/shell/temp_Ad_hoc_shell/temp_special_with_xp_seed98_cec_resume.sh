#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

PARTNER_SEED=98
LAYOUTS="[empty,wall_a]"
XP_MAPS="empty wall_a"

for map in $XP_MAPS; do
  python3 baselines/CEC_UED/modified_wall_dual_xp_test.py \
    model_name=CEC_MIXED partner_model_name=CEC_MIXED map_name="$map" \
    ENV_KWARGS.random_reset=false ENV_KWARGS.check_held_out=false \
    SEEDS='[0]' PARTNER_SEEDS="[$PARTNER_SEED]" \
    CKPT_TAG=with_xp_numenv64 layout_names="$LAYOUTS" \
    DEBUG_GIFS.enabled=true DEBUG_GIFS.max_pairs=1 \
    DEBUG_GIFS.only_cross_play=true \
    run_suffix="with_xp_numenv64_seed0_x_seed${PARTNER_SEED}" \
    WANDB_MODE=online
done

for seed in 1 2; do
  python3 baselines/CEC_UED/modified_wall_ippo_general_dual_destination_with_xp.py \
    --config-name ippo_overcooked_CEC_dual_destination_with_xp \
    SEED="$seed" NUM_ENVS=64 map_name=mixed \
    CKPT_TAG=with_xp_numenv64 layout_names="$LAYOUTS" \
    TOY_HELDOUT_NUM=100 EVAL_KWARGS.eval_interval=50 \
    XP_KWARGS.enabled=true XP_KWARGS.partner_seed="$PARTNER_SEED" \
    WANDB_MODE=online
  for map in $XP_MAPS; do
    python3 baselines/CEC_UED/modified_wall_dual_xp_test.py \
      model_name=CEC_MIXED partner_model_name=CEC_MIXED map_name="$map" \
      ENV_KWARGS.random_reset=false ENV_KWARGS.check_held_out=false \
      SEEDS="[$seed]" PARTNER_SEEDS="[$PARTNER_SEED]" \
      CKPT_TAG=with_xp_numenv64 layout_names="$LAYOUTS" \
      DEBUG_GIFS.enabled=true DEBUG_GIFS.max_pairs=1 \
      DEBUG_GIFS.only_cross_play=true \
      run_suffix="with_xp_numenv64_seed${seed}_x_seed${PARTNER_SEED}" \
      WANDB_MODE=online
  done
done
