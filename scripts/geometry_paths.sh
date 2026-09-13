#!/usr/bin/env bash
# On a cluster change this YAML, not the Python implementation.
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GEOMETRY_PATHS="${GEOMETRY_PATHS:-${REPO_ROOT}/configs/paths/libero_track4world_local.yaml}"
if [[ -n "${FASTWAM_ENV:-}" ]]; then
  export PATH="${FASTWAM_ENV}/bin:${PATH}"
  export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFSYNTH_SKIP_DOWNLOAD=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$REPO_ROOT"
