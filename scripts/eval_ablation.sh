#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/geometry_paths.sh"
: "${CKPT:?Set full trained .pt; geometry mode also needs adjacent .pt.geometry.json}"
command=(python scripts/geometry_ablation.py eval --variant "${FASTWAM_VARIANT:-uncond}" --paths "$GEOMETRY_PATHS")
case "${EVAL_MODE:-single}" in single) ;; manager) command+=(--manager);; *) exit 2;; esac
if [[ -z "${EVAL_OUTPUT_DIR:-}" ]]; then EVAL_OUTPUT_DIR="\${paths.output_root}/eval/$(date +%Y-%m-%d_%H-%M-%S)"; fi
command+=("geometry_enabled=${GEOMETRY_ENABLED:-true}" "ckpt=$CKPT"
  "EVALUATION.num_trials=${NUM_TRIALS:-50}" "MULTIRUN.num_gpus=${NUM_EVAL_GPUS:-1}"
  "EVALUATION.output_dir=${EVAL_OUTPUT_DIR}" "$@")
printf '%q ' "${command[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] || exec "${command[@]}"
