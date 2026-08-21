#!/usr/bin/env bash
# Copy trained-checkpoint lr-* folders from /app/ckpts/... into the shared
# NAS layout under /app/nas/models/ICRL/<MODEL>/<N>/<lr-folder> for CEC and
# CEC_IDDAC (where <N> is picked from the "updates" count baked into the
# final checkpoint filename), or /app/nas/models/ICRL/<MODEL>/<lr-folder>
# for IPPO/E3T/FCP (no <N>). ikFalse runs additionally get a <layout>
# subfolder before <lr-folder>.
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

# ikFalse runs train one fixed layout at a time, so their checkpoints need a
# layout subfolder to avoid colliding under the same model/N destination.
LAYOUTS=(asymm_advantages_9 coord_ring_9 counter_circuit_9 cramped_room_9 forced_coord_9)

usage() {
  echo "Usage: $0 [-f|--force] [lr-folder-path ...]" >&2
  echo "  With no path given, scans every lr-* folder under $CKPTS_ROOT." >&2
  echo "  Example: $0 /app/ckpts/idaac/overcooked/cramped_room_9/ikTrue/reset_all/lr-20260809-090611" >&2
  exit 1
}

model_for_path() {
  local src="$1" ik="$2"
  case "$src" in
    */ckpts/idaac/*)
      if [ "$ik" = "True" ]; then echo "CEC_IDDAC"; fi ;;
    */ckpts/ippo/*)
      if [ "$ik" = "False" ]; then echo "IPPO"; else echo "CEC"; fi ;;
    */ckpts/e3t/*)   echo "E3T" ;;
    */ckpts/fcp/*)   echo "FCP" ;;
    *) echo "" ;;
  esac
}

# Only the population-trained models are split by the "updates" -> N size;
# the single-layout baselines (IPPO/E3T/FCP) don't have a population size.
model_needs_n() {
  local model="$1"
  case "$model" in
    CEC|CEC_IDDAC) return 0 ;;
    *) return 1 ;;
  esac
}

# Some model types only place their final checkpoint under a distinctive
# filename tag; everything else keeps matching the generic pattern.
ckpt_pattern_for_model() {
  local model="$1"
  case "$model" in
    E3T) echo '*ckpt*e3t*updates*' ;;
    FCP) echo '*ckpt*fcp_improved*updates*' ;;
    *)   echo '*ckpt*updates*' ;;
  esac
}

# Extract the ikTrue/ikFalse segment from a checkpoint path, if present.
ik_for_path() {
  local src="$1"
  if [[ "$src" =~ /ik(True|False)(/|$) ]]; then
    echo "${BASH_REMATCH[1]}"
  fi
}

# Extract the layout name from a checkpoint path, if it matches a known layout.
layout_for_path() {
  local src="$1"
  local layout_regex="/($(IFS='|'; echo "${LAYOUTS[*]}"))(/|$)"
  if [[ "$src" =~ $layout_regex ]]; then
    echo "${BASH_REMATCH[1]}"
  fi
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

  ik="$(ik_for_path "$src")"
  model="$(model_for_path "$src" "$ik")"
  if [ -z "$model" ]; then
    echo "[SKIP] can't infer model from path (expected .../ckpts/{idaac,ippo,e3t,fcp}/...): $src" >&2
    continue
  fi

  pattern="$(ckpt_pattern_for_model "$model")"
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

  n=""
  dest_dir="$NAS_ROOT/$model"
  if model_needs_n "$model"; then
    n="${UPDATES_TO_N[$updates]:-}"
    if [ -z "$n" ]; then
      echo "[SKIP] unrecognized updates count ($updates), no destination mapping: $fname" >&2
      continue
    fi
    dest_dir="$dest_dir/$n"
  fi

  lr_name="$(basename "$src")"

  if [ "$ik" = "False" ]; then
    layout="$(layout_for_path "$src")"
    if [ -z "$layout" ]; then
      echo "[SKIP] ikFalse path but couldn't determine layout (expected one of: ${LAYOUTS[*]}): $src" >&2
      continue
    fi
    dest_dir="$dest_dir/$layout"
  fi

  dest="$dest_dir/$lr_name"

  if [ -e "$dest" ] && [ "$FORCE" -ne 1 ]; then
    echo "[SKIP] destination already exists (use -f to overwrite): $dest" >&2
    continue
  fi

  mkdir -p "$dest_dir"
  echo "[COPY] $src"
  if [ -n "$n" ]; then
    echo "    -> $dest  (updates=$updates -> $n)"
  else
    echo "    -> $dest  (updates=$updates)"
  fi
  cp -a --no-preserve=ownership "$src" "$dest_dir/"
done
