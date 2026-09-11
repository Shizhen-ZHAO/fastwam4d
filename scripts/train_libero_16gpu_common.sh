#!/usr/bin/env bash
set -euo pipefail
# Shared launcher; invoke a variant script, not this file directly.
TASK="${1:?Use train_libero_joint_16gpu.sh, train_libero_uncond_16gpu.sh or train_libero_idm_16gpu.sh}"
shift
for value in "${NNODES}" "${GPUS_PER_NODE}" "${NODE_RANK}" "${MASTER_PORT}"; do
  [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid integer: ${value}" >&2; exit 2; }
done
if (( NNODES < 1 || GPUS_PER_NODE < 1 || NNODES * GPUS_PER_NODE != 16 || NODE_RANK >= NNODES || MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo 'Require NNODES * GPUS_PER_NODE = 16, valid NODE_RANK and MASTER_PORT.' >&2
  exit 2
fi
if (( NNODES > 1 )); then
  : "${MASTER_ADDR:?Set MASTER_ADDR to the rank-0 node IP on every node}"
  : "${RUN_ID:?Set the same RUN_ID on every node; OUTPUT_DIR must be shared and identical}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
fi
export RUN_ID CUDA_HOME
export PATH="${FASTWAM_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${REPO_ROOT}"
printf 'Task: %s\nData: %s\nText cache: %s\nWan model base: %s\nActionDiT checkpoint: %s\nDataset stats: %s\nResume: %s\nOutput: %s\n' \
  "${TASK}" "${DATA_ROOT}" "${TEXT_CACHE}" "${MODEL_BASE}" "${ACTION_DIT_CHECKPOINT}" "${DATASET_STATS}" "${RESUME_STATE}" "${OUTPUT_DIR}"

# Official train_zero1.sh entrypoint/config, with explicit topology forwarding.
# accelerate --num_processes is GLOBAL world size, not GPUs per node.
command=(
  accelerate launch
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
  --deepspeed_multinode_launcher standard
  --num_processes 16
  --num_machines "${NNODES}"
  --machine_rank "${NODE_RANK}"
  --main_process_ip "${MASTER_ADDR}"
  --main_process_port "${MASTER_PORT}"
  scripts/train.py
  "task=${TASK}"
  "data.train.dataset_dirs=[${DATA_ROOT}/libero_spatial_no_noops_lerobot,${DATA_ROOT}/libero_object_no_noops_lerobot,${DATA_ROOT}/libero_goal_no_noops_lerobot,${DATA_ROOT}/libero_10_no_noops_lerobot]"
  "data.train.text_embedding_cache_dir=${TEXT_CACHE}"
  "+data.train.pretrained_norm_stats=${DATASET_STATS}"
  "model.action_dit_pretrained_path=${ACTION_DIT_CHECKPOINT}"
  "output_dir=${OUTPUT_DIR}"
  "resume=${RESUME_STATE}"
  "batch_size=${BATCH_SIZE}"
  "gradient_accumulation_steps=${GRAD_ACCUM}"
  "num_workers=${NUM_WORKERS}"
  "max_steps=${MAX_STEPS}"
  "wandb.name=${TASK}"
  "$@"
)
# Print the command only: no training or GPU access.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
