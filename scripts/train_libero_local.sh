#!/usr/bin/env bash
set -euo pipefail

# Edit these paths when moving to another machine.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FASTWAM_ENV="${FASTWAM_ENV:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_official}"
CUDA_HOME="${CUDA_HOME:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_cuda128}"
DATA_ROOT="${DATA_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2}"
TEXT_CACHE="${TEXT_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"
MODEL_BASE="${MODEL_BASE:-/mnt/homes/zhaoshizhen/lf/repos/FastWAM/checkpoints}"
ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${MODEL_BASE}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
DATASET_STATS="${DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/libero_official/${RUN_ID}}"
RESUME_STATE="${RESUME_STATE:-null}"

# Two A100s: small per-GPU batch; other model settings follow official defaults.
# NUM_GPUS=16 means one node with 16 visible GPUs.
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-null}"
NUM_WORKERS="${NUM_WORKERS:-2}"

if [[ "${NNODES:-1}" != "1" ]]; then
  echo 'This wrapper uses the official single-node launcher; multi-node training needs a separate launch configuration.' >&2
  exit 2
fi

export CUDA_HOME
export PATH="${FASTWAM_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export RUN_ID

cd "${REPO_ROOT}"
printf 'Data root: %s\nText cache: %s\nWan model base: %s\nActionDiT checkpoint: %s\nDataset stats: %s\nResume state: %s\nOutput: %s\n' \
  "${DATA_ROOT}" "${TEXT_CACHE}" "${MODEL_BASE}" "${ACTION_DIT_CHECKPOINT}" "${DATASET_STATS}" "${RESUME_STATE}" "${OUTPUT_DIR}"

# This is the unmodified official ZeRO-1 entrypoint. All overrides below are
# already supported by official FastWAM; no geometry or custom checkpoint code.
bash scripts/train_zero1.sh "${NUM_GPUS}" \
  task=libero_uncond_2cam224_1e-4 \
  "data.train.dataset_dirs=[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]" \
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}" \
  "+data.train.pretrained_norm_stats=${DATASET_STATS}" \
  "model.action_dit_pretrained_path=${ACTION_DIT_CHECKPOINT}" \
  "output_dir=${OUTPUT_DIR}" \
  "resume=${RESUME_STATE}" \
  "batch_size=${BATCH_SIZE}" \
  "num_workers=${NUM_WORKERS}" \
  "max_steps=${MAX_STEPS}" \
  "$@"
