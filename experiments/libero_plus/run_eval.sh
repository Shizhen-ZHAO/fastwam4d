#!/usr/bin/env bash
set -euo pipefail
# 用法：CKPT=/path/step.pt bash experiments/libero_plus/run_eval.sh [--config file.yaml] [Hydra 参数]
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export ROOT_DIR
CONFIG_FILE="${CONFIG_FILE:-}"
if [[ "${1:-}" == --config ]]; then
  CONFIG_FILE="${2:?--config requires a YAML file}"
  shift 2
fi
BOOTSTRAP_PYTHON="${PYTHON_BIN:-python}"
CONFIG_ARGS=("$ROOT_DIR/experiments/libero_plus/eval_config.yaml")
if [[ -n "$CONFIG_FILE" ]]; then CONFIG_ARGS+=("$CONFIG_FILE"); fi
CONFIG_EXPORTS="$("$BOOTSTRAP_PYTHON" "$ROOT_DIR/experiments/libero_plus/load_eval_config.py" "${CONFIG_ARGS[@]}")"
eval "$CONFIG_EXPORTS"
export REPO_ROOT="$ROOT_DIR"
source "$ROOT_DIR/scripts/libero_cluster_paths.sh"
if [[ -n "$FASTWAM_ENV" ]]; then
  export PATH="$FASTWAM_ENV/bin:$PATH"
  export LD_LIBRARY_PATH="$FASTWAM_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -n "$CUDA_HOME" ]]; then export PATH="$CUDA_HOME/bin:$PATH"; fi
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
case "$EVAL_MODE" in
  uncond|joint|idm) ;;
  *) echo 'EVAL_MODE must be uncond, joint or idm' >&2; exit 2 ;;
esac
[[ "$FASTWAM_ACTION_ROPE_MODE" == 1d ]] || { echo 'Action RoPE must be 1d' >&2; exit 2; }
TASK_CONFIG="${TASK_CONFIG:-libero_${EVAL_MODE}_2cam224_1e-4}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-$MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export LIBERO_PLUS_BDDL_DIR="${LIBERO_PLUS_BDDL_DIR:-$LIBERO_PLUS_ROOT/libero/libero/bddl_files}"
export LIBERO_PLUS_INIT_STATES_DIR="${LIBERO_PLUS_INIT_STATES_DIR:-$LIBERO_PLUS_ROOT/libero/libero/init_files}"
export LIBERO_PLUS_BENCHMARK_ROOT="${LIBERO_PLUS_BENCHMARK_ROOT:-$LIBERO_PLUS_ROOT/libero/libero}"
export LIBERO_ASSETS_DIR="$LIBERO_PLUS_ASSETS_DIR"
export PYTHONPATH="$LIBERO_PLUS_ROOT:$ROOT_DIR/experiments/libero:$ROOT_DIR/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL:-osmesa}}"
if [[ "$MUJOCO_GL" == egl ]]; then
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
else
  unset MUJOCO_EGL_DEVICE_ID
fi
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-$ROOT_DIR/evaluate_results/libero_plus}"
RUN_TAG="${RUN_TAG:-${EVAL_MODE}_1d_incrseed-${INFER_INCREMENT_SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_BASE_DIR/$TASK_CONFIG/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)_$$}"
TASK_FILE="${TASK_FILE:-$ROOT_DIR/experiments/libero_plus/full_10030_lpt.txt}"
if [[ "$TASK_FILE" == auto ]]; then TASK_FILE=null; fi
DATASET_STATS_PATH="${DATASET_STATS_PATH:-$DATASET_STATS}"
cd "$ROOT_DIR"
command=("$PYTHON_BIN" experiments/libero_plus/run_libero_plus_manager.py
  "task=$TASK_CONFIG" "ckpt=${CKPT:-null}"
  "model.model_id=$MODEL_ID" "model.tokenizer_model_id=$TOKENIZER_MODEL_ID"
  "model.redirect_common_files=$REDIRECT_COMMON_FILES"
  "model.action_dit_config.action_rope_mode=1d"
  "EVALUATION.dataset_stats_path=$DATASET_STATS_PATH" "EVALUATION.output_dir=$OUTPUT_DIR"
  "EVALUATION.num_trials=$NUM_TRIALS" "EVALUATION.save_video=$SAVE_VIDEO"
  "EVALUATION.visualize_future_video=$VISUALIZE_FUTURE_VIDEO"
  "EVALUATION.skip_unused_render=$SKIP_UNUSED_RENDER"
  "EVALUATION.infer_increment_seed=$INFER_INCREMENT_SEED"
  "MULTIRUN.num_gpus=$NUM_GPUS" "MULTIRUN.max_tasks_per_gpu=$MAX_TASKS_PER_GPU"
  "MULTIRUN.task_file=$TASK_FILE" "$@")
printf '%q ' "${command[@]}"
printf '\n'
# 配置预览不创建输出目录，也不导入 LIBERO/torch 或读取权重。
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exec "${command[@]}" --cfg job --resolve
fi
exec "${command[@]}"
