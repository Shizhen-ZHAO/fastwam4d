#!/usr/bin/env bash
# Defaults to a read-only plan. Use --mode extract to start the full job.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$repo_root/../.." && pwd)
cd "$repo_root"
exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONPATH="$repo_root/src:$workspace_root/.deps/fastwam_track4world:$workspace_root/Track4World${PYTHONPATH:+:$PYTHONPATH}" \
    OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    "${FASTWAM_PYTHON:-$workspace_root/.conda/envs/lf_fastwam/bin/python}" \
    scripts/extract_libero_geometry_full.py "$@"
