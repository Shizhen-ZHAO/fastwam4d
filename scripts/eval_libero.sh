#!/usr/bin/env bash
set -euo pipefail
# 用法：CKPT=/path/to/weights.pt bash scripts/eval_libero.sh uncond|joint|idm [Hydra参数...]
VARIANT="${1:?Choose uncond, joint or idm}"
shift
case "${VARIANT}" in uncond|joint|idm) ;; *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;; esac
source "$(dirname "${BASH_SOURCE[0]}")/libero_cluster_paths.sh"
: "${CKPT:?Set CKPT to trained weights/step_XXXXXX.pt, not the Wan initialization directory}"
if [[ -n "${CUDA_HOME}" ]]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
if [[ -n "${FASTWAM_ENV}" ]]; then
  export PATH="${FASTWAM_ENV}/bin:${PATH}"
  export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}" DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
case "${EVAL_MODE:-single}" in
  single) ENTRY=experiments/libero/eval_libero_single.py ;;
  manager) ENTRY=experiments/libero/run_libero_manager.py ;;
  *) echo 'EVAL_MODE must be single or manager' >&2; exit 2 ;;
esac
cd "${REPO_ROOT}"
command=(python "${ENTRY}" "task=libero_${VARIANT}_2cam224_1e-4"
  "ckpt=${CKPT}" "model.model_id=${MODEL_ID}"
  "model.tokenizer_model_id=${TOKENIZER_MODEL_ID}" "model.redirect_common_files=${REDIRECT_COMMON_FILES}"
  "model.action_dit_config.action_rope_mode=1d"
  "EVALUATION.dataset_stats_path=${DATASET_STATS}" "EVALUATION.compile_action_infer=false"
  "EVALUATION.num_trials=${NUM_TRIALS:-50}" "MULTIRUN.num_gpus=${NUM_EVAL_GPUS:-1}"
  "EVALUATION.output_dir=${EVAL_OUTPUT_DIR:-${REPO_ROOT}/evaluate_results/libero/${VARIANT}/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
  "$@")
printf '%q ' "${command[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
[[ -f "${CKPT}" ]] || { echo "Missing CKPT: ${CKPT}" >&2; exit 2; }
[[ "${DATASET_STATS}" == null || -f "${DATASET_STATS}" ]] || { echo "Missing DATASET_STATS: ${DATASET_STATS}" >&2; exit 2; }
exec "${command[@]}"
