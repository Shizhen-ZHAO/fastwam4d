#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PATHS TO EDIT ON A NEW MACHINE
# Keep the four LeRobot roots in exactly this order for cache/data alignment.
# Edit this block, or export the same variables before invoking the launcher.
# =============================================================================
FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-/mnt/homes/zhaoshizhen/lf/repos_new/FastWAM}"

FASTWAM_LIBERO_SPATIAL_ROOT="${FASTWAM_LIBERO_SPATIAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot}"
FASTWAM_LIBERO_OBJECT_ROOT="${FASTWAM_LIBERO_OBJECT_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot}"
FASTWAM_LIBERO_GOAL_ROOT="${FASTWAM_LIBERO_GOAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot}"
FASTWAM_LIBERO_10_ROOT="${FASTWAM_LIBERO_10_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot}"
FASTWAM_TEXT_EMBEDDING_CACHE="${FASTWAM_TEXT_EMBEDDING_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"

FASTWAM_BASE_CHECKPOINT="${FASTWAM_BASE_CHECKPOINT:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224.pt}"
FASTWAM_DATASET_STATS="${FASTWAM_DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
FASTWAM_MODEL_BASE="${FASTWAM_MODEL_BASE:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/model_base}"

FASTWAM_TRACK4WORLD_REPO="${FASTWAM_TRACK4WORLD_REPO:-/mnt/homes/zhaoshizhen/lf/Track4World}"
FASTWAM_TRACK4WORLD_CHECKPOINT="${FASTWAM_TRACK4WORLD_CHECKPOINT:-/mnt/homes/zhaoshizhen/lf/Track4World/checkpoints/track4world_da3.pth}"
FASTWAM_DA3_MODEL="${FASTWAM_DA3_MODEL:-/mnt/homes/zhaoshizhen/lf/Track4World/checkpoints/DA3NESTED-GIANT-LARGE-1.1}"
FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH="${FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH:-/mnt/homes/zhaoshizhen/lf/.deps/fastwam_track4world}"

FASTWAM_LIBERO_REPO="${FASTWAM_LIBERO_REPO:-/home/zhaoshizhen/lf/third_party/libero_base/repos/LIBERO}"
FASTWAM_GEOMETRY_CACHE="${FASTWAM_GEOMETRY_CACHE:-/home/zhaoshizhen/lf/repos_new/FastWAM/feature_cache/libero_track4world_v2}"
FASTWAM_OUTPUT_ROOT="${FASTWAM_OUTPUT_ROOT:-/home/zhaoshizhen/lf/repos_new/FastWAM/outputs/libero_track4world}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${FASTWAM_OUTPUT_ROOT}/offline_geometry_16gpu}"

# "same" is required on multi-GPU runs: every rank extracts validation
# geometry on its own FastWAM device instead of all ranks using cuda:0.
FASTWAM_GEOMETRY_DEVICE="${FASTWAM_GEOMETRY_DEVICE:-same}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

NUM_GPUS="${NUM_GPUS:-16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-0}"
MAX_STEPS="${MAX_STEPS:-10000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
RESUME_STATE="${RESUME_STATE:-null}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
DRY_RUN="${DRY_RUN:-0}"
# =============================================================================

export FASTWAM_LIBERO_SPATIAL_ROOT FASTWAM_LIBERO_OBJECT_ROOT
export FASTWAM_LIBERO_GOAL_ROOT FASTWAM_LIBERO_10_ROOT
export FASTWAM_TEXT_EMBEDDING_CACHE FASTWAM_BASE_CHECKPOINT
export FASTWAM_DATASET_STATS FASTWAM_MODEL_BASE FASTWAM_OUTPUT_ROOT
export FASTWAM_TRACK4WORLD_REPO FASTWAM_TRACK4WORLD_CHECKPOINT
export FASTWAM_DA3_MODEL FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH
export FASTWAM_LIBERO_REPO FASTWAM_GEOMETRY_CACHE FASTWAM_GEOMETRY_DEVICE
export HF_ENDPOINT
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH:$FASTWAM_REPO_ROOT/src:${PYTHONPATH:-}"

print_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
}

paths_config="configs/paths/libero_track4world_env.yaml"
doctor_command=(python scripts/libero_track4world.py --paths "$paths_config" doctor --require-cache)
coverage_command=(python scripts/libero_track4world.py --paths "$paths_config" coverage --require-complete)
train_command=(
  bash scripts/train_zero1.sh "$NUM_GPUS"
  paths=libero_track4world_env
  task=libero_geometry_offline_2cam224
  "paths.geometry_device=$FASTWAM_GEOMETRY_DEVICE"
  "output_dir=$TRAIN_OUTPUT_DIR"
  "batch_size=$BATCH_SIZE_PER_GPU"
  "num_workers=$NUM_WORKERS_PER_GPU"
  "max_steps=$MAX_STEPS"
  "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  "mixed_precision=$MIXED_PRECISION"
  "seed=$SEED"
  "save_every=$SAVE_EVERY"
  "eval_every=$EVAL_EVERY"
  "resume=$RESUME_STATE"
  "wandb.enabled=$WANDB_ENABLED"
  "$@"
)

echo "FastWAM + offline Track4World geometry 16-GPU training"
echo "  data[0]        = $FASTWAM_LIBERO_SPATIAL_ROOT"
echo "  data[1]        = $FASTWAM_LIBERO_OBJECT_ROOT"
echo "  data[2]        = $FASTWAM_LIBERO_GOAL_ROOT"
echo "  data[3]        = $FASTWAM_LIBERO_10_ROOT"
echo "  geometry cache = $FASTWAM_GEOMETRY_CACHE"
echo "  base ckpt      = $FASTWAM_BASE_CHECKPOINT"
echo "  dataset stats  = $FASTWAM_DATASET_STATS"
echo "  Track4World    = $FASTWAM_TRACK4WORLD_CHECKPOINT"
echo "  DA3            = $FASTWAM_DA3_MODEL"
echo "  output         = $TRAIN_OUTPUT_DIR"
print_command "${doctor_command[@]}"
print_command "${coverage_command[@]}"
print_command "${train_command[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$FASTWAM_GEOMETRY_DEVICE" != "same" ]]; then
  echo "16-GPU geometry training requires FASTWAM_GEOMETRY_DEVICE=same" >&2
  exit 2
fi
if [[ ! -d "$FASTWAM_REPO_ROOT" ]]; then
  echo "Missing FASTWAM_REPO_ROOT directory: $FASTWAM_REPO_ROOT" >&2
  exit 2
fi

cd "$FASTWAM_REPO_ROOT"
"${doctor_command[@]}"
"${coverage_command[@]}"
exec "${train_command[@]}"
