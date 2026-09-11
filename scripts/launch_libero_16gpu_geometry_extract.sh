#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PATHS TO EDIT ON A NEW MACHINE
# The four LeRobot roots and their order must be identical during extraction
# and offline training. The cache directory must be on shared POSIX storage.
# Edit this block, or export the same variables before invoking the launcher.
# =============================================================================
FASTWAM_REPO_ROOT="${FASTWAM_REPO_ROOT:-/mnt/homes/zhaoshizhen/lf/repos_new/FastWAM}"

FASTWAM_LIBERO_SPATIAL_ROOT="${FASTWAM_LIBERO_SPATIAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot}"
FASTWAM_LIBERO_OBJECT_ROOT="${FASTWAM_LIBERO_OBJECT_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot}"
FASTWAM_LIBERO_GOAL_ROOT="${FASTWAM_LIBERO_GOAL_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot}"
FASTWAM_LIBERO_10_ROOT="${FASTWAM_LIBERO_10_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot}"
FASTWAM_TEXT_EMBEDDING_CACHE="${FASTWAM_TEXT_EMBEDDING_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"

# These FastWAM paths are checked by the shared machine preflight even though
# feature extraction itself primarily consumes Track4World, DA3, and RGB data.
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
EXTRACT_LOG_DIR="${EXTRACT_LOG_DIR:-${FASTWAM_OUTPUT_ROOT}/logs/geometry_extract}"

FASTWAM_GEOMETRY_DEVICE="${FASTWAM_GEOMETRY_DEVICE:-same}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Every process sees exactly one physical GPU and therefore uses cuda:0 inside.
NUM_GPUS="${NUM_GPUS:-16}"
LOG_EVERY="${LOG_EVERY:-100}"
VERIFY_COUNT="${VERIFY_COUNT:-1000}"
RUN_PARITY="${RUN_PARITY:-1}"
PARITY_COUNT="${PARITY_COUNT:-32}"
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$FASTWAM_TRACK4WORLD_EXTRA_PYTHONPATH:$FASTWAM_REPO_ROOT/src:${PYTHONPATH:-}"

print_command() {
  printf '[command]'
  printf ' %q' "$@"
  printf '\n'
}

if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: $NUM_GPUS" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  if (( ${#gpu_ids[@]} < NUM_GPUS )); then
    echo "NUM_GPUS=$NUM_GPUS but CUDA_VISIBLE_DEVICES has only ${#gpu_ids[@]} entries" >&2
    exit 2
  fi
else
  gpu_ids=()
  for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    gpu_ids+=("$gpu")
  done
fi

paths_config="configs/paths/libero_track4world_env.yaml"
doctor_command=(python scripts/libero_track4world.py --paths "$paths_config" doctor)
coverage_command=(python scripts/libero_track4world.py --paths "$paths_config" coverage --require-complete)
verify_command=(python scripts/libero_track4world.py --paths "$paths_config" verify --count "$VERIFY_COUNT")
parity_command=(python scripts/libero_track4world.py --paths "$paths_config" parity --device cuda:0 --count "$PARITY_COUNT")

echo "Full LIBERO Track4World feature extraction"
echo "  data[0]        = $FASTWAM_LIBERO_SPATIAL_ROOT"
echo "  data[1]        = $FASTWAM_LIBERO_OBJECT_ROOT"
echo "  data[2]        = $FASTWAM_LIBERO_GOAL_ROOT"
echo "  data[3]        = $FASTWAM_LIBERO_10_ROOT"
echo "  Track4World    = $FASTWAM_TRACK4WORLD_CHECKPOINT"
echo "  DA3            = $FASTWAM_DA3_MODEL"
echo "  geometry cache = $FASTWAM_GEOMETRY_CACHE"
echo "  logs           = $EXTRACT_LOG_DIR"
echo "  GPU mapping    = ${gpu_ids[*]:0:NUM_GPUS}"
print_command "${doctor_command[@]}"
for ((shard = 0; shard < NUM_GPUS; shard++)); do
  extract_command=(
    env "CUDA_VISIBLE_DEVICES=${gpu_ids[$shard]}"
    python scripts/libero_track4world.py
    --paths "$paths_config"
    extract
    --num-shards "$NUM_GPUS"
    --shard-id "$shard"
    --device cuda:0
    --log-every "$LOG_EVERY"
  )
  print_command "${extract_command[@]}"
done
print_command "${coverage_command[@]}"
print_command "${verify_command[@]}"
if [[ "$RUN_PARITY" == "1" ]]; then
  print_command "${parity_command[@]}"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
if [[ ! -d "$FASTWAM_REPO_ROOT" ]]; then
  echo "Missing FASTWAM_REPO_ROOT directory: $FASTWAM_REPO_ROOT" >&2
  exit 2
fi

cd "$FASTWAM_REPO_ROOT"
mkdir -p "$EXTRACT_LOG_DIR"
"${doctor_command[@]}"

pids=()
stop_workers() {
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap 'stop_workers; exit 130' INT TERM

for ((shard = 0; shard < NUM_GPUS; shard++)); do
  log_file="$EXTRACT_LOG_DIR/shard_${shard}.log"
  echo "Starting shard $shard/$NUM_GPUS on physical GPU ${gpu_ids[$shard]}: $log_file"
  env "CUDA_VISIBLE_DEVICES=${gpu_ids[$shard]}" \
    python scripts/libero_track4world.py \
      --paths "$paths_config" \
      extract \
      --num-shards "$NUM_GPUS" \
      --shard-id "$shard" \
      --device cuda:0 \
      --log-every "$LOG_EVERY" \
      >"$log_file" 2>&1 &
  pids+=("$!")
done

failed=0
for ((shard = 0; shard < NUM_GPUS; shard++)); do
  if wait "${pids[$shard]}"; then
    echo "Shard $shard completed: $EXTRACT_LOG_DIR/shard_${shard}.log"
  else
    status=$?
    echo "Shard $shard failed with status $status: $EXTRACT_LOG_DIR/shard_${shard}.log" >&2
    failed=1
  fi
done
trap - INT TERM

if (( failed != 0 )); then
  echo "At least one extraction shard failed; coverage/parity were not run." >&2
  exit 1
fi

"${coverage_command[@]}"
"${verify_command[@]}"
if [[ "$RUN_PARITY" == "1" ]]; then
  # Workers have exited, so physical GPU 0 is available for online re-extraction.
  env CUDA_VISIBLE_DEVICES="${gpu_ids[0]}" "${parity_command[@]}"
fi

echo "Full feature extraction and validation completed successfully."
