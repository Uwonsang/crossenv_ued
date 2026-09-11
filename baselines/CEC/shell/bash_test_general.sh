#!/usr/bin/env bash

# Usage: bash_test_general.sh [GPU] [MODEL|MODEL,...|all] [MODEL_NUM_ENVS] [OUTPUT_DIR]
# Example: bash_test_general.sh 0 CEC_IDAAC 128 /mnt/nas/wonsang/xp_results

gpu="${1:-0}"
model_arg="${2:-all}"
model_num_envs="${3:-256}"
output_dir="${4:-}"

all_models=(
  CEC
  CEC_Finetune
  CEC_IDAAC
  CEC_IDAAC_Finetune
  E3T
  IPPO
)

if [[ "${model_arg}" == "all" ]]; then
  models=("${all_models[@]}")
else
  IFS=',' read -r -a models <<< "${model_arg}"
  for model in "${models[@]}"; do
    case " ${all_models[*]} " in
      *" ${model} "*) ;;
      *)
        echo "Unknown model: ${model}" >&2
        echo "Available: all, ${all_models[*]}" >&2
        exit 2
        ;;
    esac
  done
fi

case "${model_num_envs}" in
  32|64|128|256) ;;
  *)
    echo "MODEL_NUM_ENVS must be one of: 32, 64, 128, 256" >&2
    exit 2
    ;;
esac

layouts=(
  asymm_advantages_9
  coord_ring_9
  counter_circuit_9
  cramped_room_9
  forced_coord_9
)

for layout in "${layouts[@]}"; do
  for model in "${models[@]}"; do
    model_output_dir=""
    if [[ -n "${output_dir}" ]]; then
      model_output_dir="${output_dir%/}/${model}"
      if [[ "${model}" == "CEC" || "${model}" == "CEC_IDAAC" ]]; then
        model_output_dir="${model_output_dir}/envs${model_num_envs}"
      fi
    fi
    echo "XP evaluation: layout=${layout}, model=${model}, seeds=0-5"
    if [[ -n "${model_output_dir}" ]]; then
      echo "Output directory: ${model_output_dir}"
    fi
    command=(python baselines/CEC/test_general.py
      "model_name=${model}"
      "ENV_KWARGS.layout=${layout}"
      "MODEL_NUM_ENVS=${model_num_envs}"
      NUM_MODELS=6
      XP_ONLY=False
    )
    if [[ -n "${model_output_dir}" ]]; then
      command+=("OUTPUT_DIR=${model_output_dir}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}"
  done
done
