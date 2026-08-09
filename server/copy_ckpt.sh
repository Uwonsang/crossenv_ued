#!/usr/bin/env bash
# Copy trained-checkpoint lr-* folders from /app/ckpts/... into the shared
# NAS layout under /app/nas/models/ICRL/<MODEL>/<N>/<lr-folder>, where <N> is
# picked from the "updates" count baked into the final checkpoint filename.
set -euo pipefail

NAS_ROOT="/app/nas/models/ICRL"
CKPTS_ROOT="/app/ckpts"
FORCE=0

declare -A UPDATES_TO_N=(
  [45776]=256
  [91552]=128
  [183105]=64
  [366210]=32
)

usage() {
  echo "Usage: $0 [-f|--force] [lr-folder-path ...]" >&2
  echo "  With no path given, scans every lr-* folder under $CKPTS_ROOT." >&2
  echo "  Example: $0 /app/ckpts/idaac/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260809-090611" >&2
  exit 1
}

model_for_path() {
  local src="$1"
  case "$src" in
    */ckpts/idaac_pop/*) echo "CEC_POP_IDDAC" ;;
    */ckpts/idaac/*)     echo "CEC_IDDAC" ;;
    */ckpts/ippo_pop/*)  echo "CEC_POP" ;;
    */ckpts/ippo/*)      echo "CEC" ;;
    *) echo "" ;;
  esac
}

ARGS=()
for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE=1 ;;
    -h|--help) usage ;;
    *) ARGS+=("$arg") ;;
  esac
done

if [ "${#ARGS[@]}" -eq 0 ]; then
  echo "[SCAN] no path given, discovering lr-* folders under $CKPTS_ROOT" >&2
  mapfile -t ARGS < <(find "$CKPTS_ROOT" -type d -name 'lr-*' | sort)
  if [ "${#ARGS[@]}" -eq 0 ]; then
    echo "[SCAN] no lr-* folders found under $CKPTS_ROOT" >&2
    exit 0
  fi
fi

for src in "${ARGS[@]}"; do
  src="${src%/}"

  if [ ! -d "$src" ]; then
    echo "[SKIP] not a directory: $src" >&2
    continue
  fi

  model="$(model_for_path "$src")"
  if [ -z "$model" ]; then
    echo "[SKIP] can't infer model from path (expected .../ckpts/{idaac,idaac_pop,ippo,ippo_pop}/...): $src" >&2
    continue
  fi

  mapfile -t ckpt_files < <(find "$src" -maxdepth 1 -type f -name '*ckpt*updates*')
  if [ "${#ckpt_files[@]}" -eq 0 ]; then
    echo "[SKIP] no *ckpt*updates* checkpoint found in: $src" >&2
    continue
  fi
  if [ "${#ckpt_files[@]}" -gt 1 ]; then
    echo "[SKIP] multiple checkpoint files found, please resolve manually:" >&2
    printf '    %s\n' "${ckpt_files[@]}" >&2
    continue
  fi

  fname="$(basename "${ckpt_files[0]}")"
  if [[ ! "$fname" =~ updates([0-9]+) ]]; then
    echo "[SKIP] couldn't parse updates count from filename: $fname" >&2
    continue
  fi
  updates="${BASH_REMATCH[1]}"
  n="${UPDATES_TO_N[$updates]:-}"
  if [ -z "$n" ]; then
    echo "[SKIP] unrecognized updates count ($updates), no destination mapping: $fname" >&2
    continue
  fi

  lr_name="$(basename "$src")"
  dest_dir="$NAS_ROOT/$model/$n"
  dest="$dest_dir/$lr_name"

  if [ -e "$dest" ] && [ "$FORCE" -ne 1 ]; then
    echo "[SKIP] destination already exists (use -f to overwrite): $dest" >&2
    continue
  fi

  mkdir -p "$dest_dir"
  echo "[COPY] $src"
  echo "    -> $dest  (updates=$updates -> $n)"
  cp -a "$src" "$dest_dir/"
done
