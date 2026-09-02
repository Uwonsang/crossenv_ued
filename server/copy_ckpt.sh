#!/usr/bin/env bash
# Copy trained-checkpoint lr-* folders from /app/ckpts/... into the shared
# NAS layout under /app/nas/models/ICRL/<MODEL>/<N>/<seed> for CEC and
# CEC_IDDAC (where <N> is picked from the "updates" count baked into the
# final checkpoint filename), or /app/nas/models/ICRL/<MODEL>/<folder>
# for IPPO/E3T/FCP (no <N>). ikFalse runs additionally get a <layout>
# subfolder before <folder>. CEC, CEC_IDDAC, E3T, and FCP use the seed prefix
# in their checkpoint filename as the destination folder name. IPPO ikFalse
# runs use fcp_pool/seed*_final.pkl and likewise copy under the corresponding seed.
# CEC_IDDAC_Finetune and CEC_Finetune runs bundle multiple seeds' checkpoints
# in one lr-* folder, so each seed{N}_ckpt*.pkl (plus its seed{N}_mid_ckpts
# folder, if present) is split out into its own
# /app/nas/models/ICRL/<MODEL>/<layout>/<seed>.
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
    */ckpts/iddac_finetune/*) echo "CEC_IDDAC_Finetune" ;;
    */ckpts/ippo_finetune/*)  echo "CEC_Finetune" ;;
    */ckpts/ippo/*)
      if [ "$ik" = "False" ]; then echo "IPPO"; else echo "CEC"; fi ;;
    */ckpts/e3t/*)   echo "E3T" ;;
    */ckpts/fcp/*)   echo "FCP" ;;
    *) echo "" ;;
  esac
}

# Finetune runs bundle every seed's checkpoint (and mid_ckpts) together in one
# lr-* folder instead of one seed per folder, so they need the per-seed split
# handled separately from the single-checkpoint-per-folder models below.
model_is_finetune() {
  local model="$1"
  case "$model" in
    CEC_IDDAC_Finetune|CEC_Finetune) return 0 ;;
    *) return 1 ;;
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

  # This run was already relocated separately, so omit it from scan output.
  if [[ "$src" == */ckpts/ippo/overcooked/counter_circuit_9/ikFalse/reset_all/lr-20260825-132120 ]]; then
    continue
  fi

  if [ ! -d "$src" ]; then
    echo "[SKIP] not a directory: $src" >&2
    continue
  fi

  ik="$(ik_for_path "$src")"
  model="$(model_for_path "$src" "$ik")"
  if [ -z "$model" ]; then
    echo "[SKIP] can't infer model from path (expected .../ckpts/{idaac,iddac_finetune,ippo,ippo_finetune,e3t,fcp}/...): $src" >&2
    continue
  fi

  if model_is_finetune "$model"; then
    layout="$(layout_for_path "$src")"
    if [ -z "$layout" ]; then
      echo "[SKIP] couldn't determine layout (expected one of: ${LAYOUTS[*]}): $src" >&2
      continue
    fi
    dest_dir="$NAS_ROOT/$model/$layout"

    pattern="$(ckpt_pattern_for_model "$model")"
    mapfile -t ckpt_files < <(find "$src" -maxdepth 1 -type f -name "$pattern" | sort)
    if [ "${#ckpt_files[@]}" -eq 0 ]; then
      echo "[SKIP] no checkpoint matching '$pattern' found in: $src" >&2
      continue
    fi

    for ckpt_file in "${ckpt_files[@]}"; do
      fname="$(basename "$ckpt_file")"
      if [[ ! "$fname" =~ ^(seed[0-9]+)_ckpt ]]; then
        echo "[SKIP] couldn't parse seed from checkpoint filename: $fname" >&2
        continue
      fi
      seed_name="${BASH_REMATCH[1]}"
      dest="$dest_dir/$seed_name"

      if [ -e "$dest" ] && [ "$FORCE" -ne 1 ]; then
        echo "[SKIP] destination already exists (use -f to overwrite): $dest" >&2
        continue
      fi

      mkdir -p "$dest"
      echo "[COPY] $src ($fname)"
      echo "    -> $dest"
      cp -a --no-preserve=ownership "$ckpt_file" "$dest/"
      if [ -d "$src/${seed_name}_mid_ckpts" ]; then
        cp -a --no-preserve=ownership "$src/${seed_name}_mid_ckpts" "$dest/"
      fi
    done
    continue
  fi

  updates=""
  seed_name=""
  mapfile -t fcp_final_files < <(find "$src/fcp_pool" -maxdepth 1 -type f \
    -name 'seed*_final.pkl' 2>/dev/null | sort)
  if [[ "$src" == */ckpts/ippo/*/ikFalse/* ]] &&
     [ "${#fcp_final_files[@]}" -gt 0 ]; then
    if [ "${#fcp_final_files[@]}" -gt 1 ]; then
      echo "[SKIP] multiple final seed checkpoints found, can't choose destination name:" >&2
      printf '    %s\n' "${fcp_final_files[@]}" >&2
      continue
    fi
    # This IPPO run stores its final checkpoint in fcp_pool instead of using
    # the usual top-level *ckpt*updates* filename.
    seed_file="$(basename "${fcp_final_files[0]}")"
    seed_name="${seed_file%_final.pkl}"
    checkpoint_desc="fcp_pool/$seed_file"
  else
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
    checkpoint_desc="$fname"

    case "$model" in
      CEC|CEC_IDDAC|E3T|FCP)
        if [[ ! "$fname" =~ ^(seed[0-9]+)_ckpt ]]; then
          echo "[SKIP] couldn't parse seed from checkpoint filename: $fname" >&2
          continue
        fi
        seed_name="${BASH_REMATCH[1]}"
        ;;
    esac
  fi

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

  dest_name="$(basename "$src")"
  if [ -n "$seed_name" ]; then
    dest_name="$seed_name"
  fi

  if [ "$ik" = "False" ]; then
    layout="$(layout_for_path "$src")"
    if [ -z "$layout" ]; then
      echo "[SKIP] ikFalse path but couldn't determine layout (expected one of: ${LAYOUTS[*]}): $src" >&2
      continue
    fi
    dest_dir="$dest_dir/$layout"
  fi

  dest="$dest_dir/$dest_name"

  if [ -e "$dest" ] && [ "$FORCE" -ne 1 ]; then
    echo "[SKIP] destination already exists (use -f to overwrite): $dest" >&2
    continue
  fi

  mkdir -p "$dest_dir"
  echo "[COPY] $src"
  if [ -n "$n" ]; then
    echo "    -> $dest  (updates=$updates -> $n)"
  elif [ -n "$updates" ]; then
    echo "    -> $dest  (updates=$updates)"
  else
    echo "    -> $dest  (checkpoint=$checkpoint_desc)"
  fi
  if [ -n "$seed_name" ]; then
    mkdir -p "$dest"
    cp -a --no-preserve=ownership "$src/." "$dest/"
  else
    cp -a --no-preserve=ownership "$src" "$dest_dir/"
  fi
done
