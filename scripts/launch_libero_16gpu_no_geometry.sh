#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PATHS TO EDIT ON A NEW MACHINE
# These defaults describe the machine on which this integration was verified.
# Edit this block, or export the same variables before invoking the launcher.
# =============================================================================
FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-/mnt/homes/zhaoshizhen/lf/repos_new/FastWAM}"

FASTWAM_LIBERO_SPATIAL_ROOT="${FASTWAM_LIBERO_SPATIAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot}"
FASTWAM_LIBERO_OBJECT_ROOT="${FASTWAM_LIBERO_OBJECT_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot}"
FASTWAM_LIBERO_GOAL_ROOT="${FASTWAM_LIBERO_GOAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot}"
FASTWAM_LIBERO_10_ROOT="${FASTWAM_LIBERO_10_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot}"
FASTWAM_TEXT_EMBEDDING_CACHE="${FASTWAM_TEXT_EMBEDDING_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"

# This control run starts from the same immutable FastWAM checkpoint and stats
# used by the geometry-adapter run, but updates the original FastWAM DiTs.
FASTWAM_BASE_CHECKPOINT="${FASTWAM_BASE_CHECKPOINT:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224.pt}"
FASTWAM_DATASET_STATS="${FASTWAM_DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
FASTWAM_MODEL_BASE="${FASTWAM_MODEL_BASE:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/model_base}"
FASTWAM_OUTPUT_ROOT="${FASTWAM_OUTPUT_ROOT:-/home/zhaoshizhen/lf/repos_new/FastWAM/outputs/libero_track4world}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-${FASTWAM_OUTPUT_ROOT}/no_geometry_16gpu}"

# The shared path config contains these fields, although this baseline never
# instantiates Track4World or reads the geometry cache.
FASTWAM_TRACK4WORLD_REPO="${FASTWAM_TRACK4WORLD_REPO:-/mnt/homes/zhaoshizhen/lf/Track4World}"
FASTWAM_TRACK4WORLD_CHECKPOINT="${FASTWAM_TRACK4WORLD_CHECKPOINT:-/mnt/homes/zhaoshizhen/lf/Track4World/checkpoints/track4world_da3.pth}"
FASTWAM_DA3_MODEL="${FASTWAM_DA3_MODEL:-/mnt/homes/zhaoshizhen/lf/Track4World/checkpoints/DA3NESTED-GIANT-LARGE-1.1}"
FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH="${FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH:-/mnt/homes/zhaoshizhen/lf/.deps/fastwam_track4world}"
FASTWAM_LIBERO_REPO="${FASTWAM_LIBERO_REPO:-/home/zhaoshizhen/lf/third_party/libero_base/repos/LIBERO}"
FASTWAM_GEOMETRY_CACHE="${FASTWAM_GEOMETRY_CACHE:-/home/zhaoshizhen/lf/repos_new/FastWAM/feature_cache/libero_track4world_v2}"
FASTWAM_GEOMETRY_DEVICE="${FASTWAM_GEOMETRY_DEVICE:-same}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Official task defaults are explicit so the effective distributed run is clear.
NUM_GPUS="${NUM_GPUS:-16}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-16}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
MAX_STEPS="${MAX_STEPS:-null}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
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

print_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
}

require_file() {
  [[ -f "$2" ]] || { echo "Missing $1 file: $2" >&2; exit 2; }
}

require_dir() {
  [[ -d "$2" ]] || { echo "Missing $1 directory: $2" >&2; exit 2; }
}

command=(
  bash scripts/train_zero1.sh "$NUM_GPUS"
  paths=libero_track4world_env
  task=libero_uncond_2cam224_1e-4
  'data.train.dataset_dirs=${paths.train_datasets}'
  'data.train.text_embedding_cache_dir=${paths.text_embedding_cache}'
  '+data.train.pretrained_norm_stats=${paths.dataset_stats}'
  '+model.model_base_path=${paths.model_base}'
  model.skip_dit_load_from_pretrain=true
  model.action_dit_pretrained_path=null
  'initial_checkpoint=${paths.fastwam_checkpoint}'
  "output_dir=$TRAIN_OUTPUT_DIR"
  "batch_size=$BATCH_SIZE_PER_GPU"
  "num_workers=$NUM_WORKERS_PER_GPU"
  "num_epochs=$NUM_EPOCHS"
  "max_steps=$MAX_STEPS"
  "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
  "mixed_precision=$MIXED_PRECISION"
  "seed=$SEED"
  "resume=$RESUME_STATE"
  "wandb.enabled=$WANDB_ENABLED"
  "$@"
)

echo "FastWAM no-geometry 16-GPU control"
echo "  data[0]       = $FASTWAM_LIBERO_SPATIAL_ROOT"
echo "  data[1]       = $FASTWAM_LIBERO_OBJECT_ROOT"
echo "  data[2]       = $FASTWAM_LIBERO_GOAL_ROOT"
echo "  data[3]       = $FASTWAM_LIBERO_10_ROOT"
echo "  base ckpt     = $FASTWAM_BASE_CHECKPOINT"
echo "  dataset stats = $FASTWAM_DATASET_STATS"
echo "  model base    = $FASTWAM_MODEL_BASE"
echo "  output        = $TRAIN_OUTPUT_DIR"
print_command "${command[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

require_dir FASTWAM_REPO_ROOT "$FASTWAM_REPO_ROOT"
require_dir LIBERO_SPATIAL "$FASTWAM_LIBERO_SPATIAL_ROOT"
require_dir LIBERO_OBJECT "$FASTWAM_LIBERO_OBJECT_ROOT"
require_dir LIBERO_GOAL "$FASTWAM_LIBERO_GOAL_ROOT"
require_dir LIBERO_10 "$FASTWAM_LIBERO_10_ROOT"
require_dir text_embedding_cache "$FASTWAM_TEXT_EMBEDDING_CACHE"
require_dir model_base "$FASTWAM_MODEL_BASE"
require_file FastWAM_checkpoint "$FASTWAM_BASE_CHECKPOINT"
require_file dataset_stats "$FASTWAM_DATASET_STATS"

cd "$FASTWAM_REPO_ROOT"
exec "${command[@]}"
