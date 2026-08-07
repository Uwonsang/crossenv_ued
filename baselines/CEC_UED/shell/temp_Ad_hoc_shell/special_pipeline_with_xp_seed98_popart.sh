#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
export PYTHONPATH="$PWD/baselines/CEC_UED:$PWD:${PYTHONPATH:-}"

PARTNER_SEED=98
TRAIN_SEEDS="0 1 2"
LAYOUTS="[empty,wall_a]"
XP_MAPS="empty wall_a"

python3 baselines/CEC_UED/modified_wall_ippo_general_gradient_pop_with_xp.py \
  --config-name ippo_overcooked_CEC_dual_destination_popart_with_xp \
  SEED="$PARTNER_SEED" NUM_ENVS=256 map_name=mixed \
  CKPT_TAG=with_xp_numenv256 layout_names="$LAYOUTS" \
  TOY_HELDOUT_NUM=100 EVAL_KWARGS.eval_interval=50 \
  XP_KWARGS.enabled=false WANDB_MODE=online

for seed in $TRAIN_SEEDS; do
  python3 baselines/CEC_UED/modified_wall_ippo_general_gradient_pop_with_xp.py \
    --config-name ippo_overcooked_CEC_dual_destination_popart_with_xp \
    SEED="$seed" NUM_ENVS=256 map_name=mixed \
    CKPT_TAG=with_xp_numenv256 layout_names="$LAYOUTS" \
    TOY_HELDOUT_NUM=100 EVAL_KWARGS.eval_interval=50 \
    XP_KWARGS.enabled=true XP_KWARGS.partner_seed="$PARTNER_SEED" \
    WANDB_MODE=online

  for map in $XP_MAPS; do
    python3 baselines/CEC_UED/modified_wall_dual_xp_test.py \
      model_name=CEC_POPART_MIXED partner_model_name=CEC_POPART_MIXED map_name="$map" \
      ENV_KWARGS.random_reset=false ENV_KWARGS.check_held_out=false \
      SEEDS="[$seed]" PARTNER_SEEDS="[$PARTNER_SEED]" \
      CKPT_TAG=with_xp_numenv256 \
      DEBUG_GIFS.enabled=true DEBUG_GIFS.max_pairs=1 \
      DEBUG_GIFS.only_cross_play=true \
      run_suffix="with_xp_numenv256_seed${seed}_x_seed${PARTNER_SEED}" \
      WANDB_MODE=online
  done
done
