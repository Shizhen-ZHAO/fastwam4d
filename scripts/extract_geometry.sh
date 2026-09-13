#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/geometry_paths.sh"
command=(python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" extract
  --num-shards "${NUM_SHARDS:-1}" --shard-id "${SHARD_ID:-0}" --device "${DEVICE:-cuda:0}" "$@")
printf '%q ' "${command[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] || exec "${command[@]}"
