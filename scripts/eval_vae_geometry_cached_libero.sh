#!/usr/bin/env bash
# Native LIBERO rollout, RGB history only. There is no feature-cache argument.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$repo_root/scripts/eval_track4world_online_libero.sh" \
    EVALUATION.geometry_adapter=outputs/libero_vae_geometry_cached_60steps/geometry_adapter.pt \
    EVALUATION.visualize_future_video=false \
    EVALUATION.output_dir=outputs/libero_vae_geometry_cached_rollout "$@"
