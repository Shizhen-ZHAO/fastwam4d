#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/geometry_paths.sh"
: "${CKPT:?Set trained .pt; geometry mode needs adjacent .pt.geometry.json}"
exec python scripts/geometry_ablation.py eval-plus --paths "$GEOMETRY_PATHS" \
  "geometry_enabled=${GEOMETRY_ENABLED:-true}" "$@"
