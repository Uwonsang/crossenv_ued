#!/usr/bin/env bash
set -uo pipefail

################ USER CONFIG ################
MAX_PER_GPU=1
BASE_CMD='CUDA_VISIBLE_DEVICES={gpu}'

COMMANDS=(
    
  "python fast_td3/train.py --env_name h1hand-walk-v0  --seed 1 --agent fasttd3_fastswiglulnlast_critic --tag baseline"
  "python fast_td3/train.py --env_name h1hand-walk-v0  --seed 2 --agent fasttd3_fastswiglulnlast_critic --tag baseline"
  "python fast_td3/train.py --env_name h1hand-walk-v0  --seed 3 --agent fasttd3_fastswiglulnlast_critic --tag baseline"

)
#############################################
# BASE_CMD='docker run --rm --gpus="device={gpu}" --ipc=host \
#   -v "$(pwd)":/app -w /app fasttd3-hb' for docker 

# --- GPU 목록 파싱 ---
if [ "$#" -lt 1 ]; then
  echo "[$(date +'%F %T')] Usage: $0 <GPU0> [GPU1 GPU2 ...]" >&2; exit 1
fi
GPUS=("$@"); for g in "${GPUS[@]}"; do [[ $g =~ ^[0-9]+$ ]] || { echo "GPU index must be numeric: $g" >&2; exit 1; }; done

# --- 내부 상태 ---
declare -A PID2GPU
declare -A GPU_LOAD
PIDS=()

TOTAL=${#COMMANDS[@]}
INDEX=0
FINISHED=0
FAILED=0

trap_ctrlc() { echo "[$(date +'%F %T')] Ctrl+C → killing all jobs"; for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; exit 1; }
trap trap_ctrlc INT

while [ $((FINISHED + FAILED)) -lt "$TOTAL" ]; do
  # 새 작업 실행
  for gpu in "${GPUS[@]}"; do
    load=${GPU_LOAD[$gpu]:-0}
    if [ $load -lt $MAX_PER_GPU ] && [ $INDEX -lt $TOTAL ]; then
      cmd="${COMMANDS[$INDEX]}"
      [ -z "$cmd" ] && { INDEX=$((INDEX+1)); continue; }

      cname="multi-sdrl-${gpu}-${INDEX}"
      full_cmd="${BASE_CMD//\{gpu\}/$gpu}"
      full_cmd="${full_cmd//\{cname\}/$cname}"
      full_cmd="$full_cmd $cmd"

      echo "[$(date +'%F %T')] [GPU $gpu] START: $cmd"
      bash -c "$full_cmd" &
      pid=$!

      PIDS+=("$pid"); PID2GPU[$pid]=$gpu; GPU_LOAD[$gpu]=$((load+1)); INDEX=$((INDEX+1))
    fi
  done

  # 종료된 작업 정리
  new_PIDS=()
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_PIDS+=("$pid")
    else
      gpu=${PID2GPU[$pid]}
      GPU_LOAD[$gpu]=$((GPU_LOAD[$gpu]-1))
      unset PID2GPU[$pid]
      wait "$pid" 2>/dev/null
      exit_code=$?
      if [ "$exit_code" -eq 0 ]; then
        echo "[$(date +'%F %T')] [GPU $gpu] FINISH (pid=$pid)"
        ((FINISHED++))
      else
        echo "[$(date +'%F %T')] [GPU $gpu] FAILED (pid=$pid, exit=$exit_code)"
        ((FAILED++))
      fi
    fi
  done
  PIDS=("${new_PIDS[@]}")

  echo "[$(date +'%F %T')] Running:${#PIDS[@]} / Finished:$FINISHED / Failed:$FAILED / Remaining:$((TOTAL-FINISHED-FAILED-${#PIDS[@]}))"
  sleep 1
done



# clean local wandb files
wandb sync --clean-force --clean-old-hours 3
echo "cleaned wandb dir"

echo "[$(date +'%F %T')] All jobs done. Finished:$FINISHED Failed:$FAILED"