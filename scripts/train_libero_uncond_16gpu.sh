#!/usr/bin/env bash
set -euo pipefail

# Edit paths here when moving to the cluster, or set environment variables.
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export FASTWAM_ENV="${FASTWAM_ENV:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_official}"
export CUDA_HOME="${CUDA_HOME:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_cuda128}"
export DATA_ROOT="${DATA_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2}"
export TEXT_CACHE="${TEXT_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"
export MODEL_BASE="${MODEL_BASE:-/mnt/homes/zhaoshizhen/lf/repos/FastWAM/checkpoints}"
export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${MODEL_BASE}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DATASET_STATS="${DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/libero_uncond_16gpu/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
export RESUME_STATE="${RESUME_STATE:-null}"

# Single node: defaults to 16 GPUs. Two nodes: set NNODES=2 GPUS_PER_NODE=8,
# NODE_RANK=0/1, MASTER_ADDR=<rank-0 IP>, same RUN_ID and shared OUTPUT_DIR.
export NNODES="${NNODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# Conservative memory defaults; effective batch = 16 * BATCH_SIZE * GRAD_ACCUM.
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export MAX_STEPS="${MAX_STEPS:-null}"

bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_uncond_2cam224_1e-4 "$@"
