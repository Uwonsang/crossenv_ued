#!/usr/bin/env bash
# Copy trained-checkpoint lr-* folders from /app/ckpts/... into the shared
# NAS layout under /app/nas/stiffness_ckpt/models/ICLR/<MODEL>/<N>/seed<K> for CEC and
# CEC_IDAAC (where <N> is picked from the "updates" count and seed<K> from the
# "seed" prefix of the final checkpoint filename). Only ikTrue runs are copied.
# Existing destinations are reported and never overwritten.
set -euo pipefail

NAS_ROOT="/app/nas/stiffness_ckpt/models/ICLR"
CKPTS_ROOT="/app/ckpts"

declare -A UPDATES_TO_N=(
  [2929]=256
  [5859]=128
  [11718]=64
  [23437]=32
)

usage() {
  echo "Usage: $0 [lr-folder-path ...]" >&2
  echo "  With no path given, scans every lr-* folder under $CKPTS_ROOT." >&2
  echo "  Example: $0 /app/ckpts/idaac/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260809-090611" >&2
  exit 1
}

model_for_path() {
  local src="$1"
  case "$src" in
    */ckpts/idaac/*) echo "CEC_IDAAC" ;;
    */ckpts/ippo/*)  echo "CEC" ;;
    *) echo "" ;;
  esac
}

ARGS=()
for arg in "$@"; do
  case "$arg" in
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

  if [[ ! "$src" =~ /ikTrue(/|$) ]]; then
    echo "[SKIP] not an ikTrue run: $src" >&2
    continue
  fi

  model="$(model_for_path "$src")"
  if [ -z "$model" ]; then
    echo "[SKIP] can't infer model from path (expected .../ckpts/{idaac,ippo}/...): $src" >&2
    continue
  fi

  pattern='*ckpt*updates*.pkl'
  mapfile -t ckpt_files < <(find "$src" -maxdepth 1 -type f -name "$pattern")
  if [ "${#ckpt_files[@]}" -eq 0 ]; then
    echo "[SKIP] no checkpoint matching '$pattern' found in: $src" >&2
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
  dest_dir="$NAS_ROOT/$model/$n"

  if [[ ! "$fname" =~ ^(seed[0-9]+)_ ]]; then
    echo "[SKIP] couldn't parse seed from filename: $fname" >&2
    continue
  fi
  seed_name="${BASH_REMATCH[1]}"
  dest="$dest_dir/$seed_name"

  if [ -e "$dest" ]; then
    echo "[EXISTS] destination already exists, not overwriting: $dest  (from $src)" >&2
    continue
  fi

  mkdir -p "$dest_dir"
  echo "[COPY] $src"
  echo "    -> $dest  (updates=$updates -> $n)"
  cp -a --no-preserve=ownership "$src" "$dest"
done
