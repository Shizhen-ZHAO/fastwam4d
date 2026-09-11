#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PATHS TO EDIT ON A NEW MACHINE
# Evaluation consumes simulator RGB online; it never reads the geometry cache.
# Edit this block, or export the same variables before invoking the launcher.
# =============================================================================
FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-/mnt/homes/zhaoshizhen/lf/repos_new/FastWAM}"

FASTWAM_LIBERO_SPATIAL_ROOT="${FASTWAM_LIBERO_SPATIAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot}"
FASTWAM_LIBERO_OBJECT_ROOT="${FASTWAM_LIBERO_OBJECT_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot}"
FASTWAM_LIBERO_GOAL_ROOT="${FASTWAM_LIBERO_GOAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot}"
FASTWAM_LIBERO_10_ROOT="${FASTWAM_LIBERO_10_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot}"
FASTWAM_TEXT_EMBEDDING_CACHE="${FASTWAM_TEXT_EMBEDDING_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"

# Evaluation loads the immutable base checkpoint plus the adapter-only file
# produced by launch_libero_16gpu_offline_geometry.sh.
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
GEOMETRY_ADAPTER="${GEOMETRY_ADAPTER:-${FASTWAM_OUTPUT_ROOT}/offline_geometry_16gpu/checkpoints/weights/step_010000.pt}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${FASTWAM_OUTPUT_ROOT}/eval_online_geometry}"

FASTWAM_GEOMETRY_DEVICE="${FASTWAM_GEOMETRY_DEVICE:-same}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Task-parallel evaluation: one persistent FastWAM + Track4World worker per GPU.
NUM_GPUS="${NUM_GPUS:-16}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-10}"
SIGMA_SHIFT="${SIGMA_SHIFT:-5.0}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-null}"
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

print_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
}

eval_command=(
  python scripts/run_libero_geometry_eval.py
  --manager
  --paths configs/paths/libero_track4world_env.yaml
  task=libero_geometry_offline_2cam224
  "ckpt=$FASTWAM_BASE_CHECKPOINT"
  "EVALUATION.geometry_adapter=$GEOMETRY_ADAPTER"
  "EVALUATION.dataset_stats_path=$FASTWAM_DATASET_STATS"
  "EVALUATION.output_dir=$EVAL_OUTPUT_DIR"
  EVALUATION.compile_action_infer=false
  "EVALUATION.num_trials=$NUM_TRIALS"
  "EVALUATION.max_steps=$MAX_ENV_STEPS"
  "EVALUATION.num_inference_steps=$NUM_INFERENCE_STEPS"
  "EVALUATION.replan_steps=$REPLAN_STEPS"
  "EVALUATION.sigma_shift=$SIGMA_SHIFT"
  "MULTIRUN.num_gpus=$NUM_GPUS"
  'MULTIRUN.task_suite_names=[libero_10,libero_goal,libero_spatial,libero_object]'
  "$@"
)

echo "FastWAM + online Track4World LIBERO evaluation"
echo "  base ckpt        = $FASTWAM_BASE_CHECKPOINT"
echo "  geometry adapter = $GEOMETRY_ADAPTER"
echo "  dataset stats    = $FASTWAM_DATASET_STATS"
echo "  Track4World      = $FASTWAM_TRACK4WORLD_CHECKPOINT"
echo "  DA3              = $FASTWAM_DA3_MODEL"
echo "  LIBERO repo      = $FASTWAM_LIBERO_REPO"
echo "  output           = $EVAL_OUTPUT_DIR"
echo "  offline cache    = not used"
print_command "${eval_command[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

if [[ "$FASTWAM_GEOMETRY_DEVICE" != "same" ]]; then
  echo "Multi-GPU online evaluation requires FASTWAM_GEOMETRY_DEVICE=same" >&2
  exit 2
fi
if [[ ! -d "$FASTWAM_REPO_ROOT" ]]; then
  echo "Missing FASTWAM_REPO_ROOT directory: $FASTWAM_REPO_ROOT" >&2
  exit 2
fi
if [[ ! -f "$GEOMETRY_ADAPTER" ]]; then
  echo "Missing geometry adapter: $GEOMETRY_ADAPTER" >&2
  exit 2
fi

cd "$FASTWAM_REPO_ROOT"
exec "${eval_command[@]}"
