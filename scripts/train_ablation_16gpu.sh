#!/usr/bin/env bash
set -euo pipefail
# GEOMETRY_ENABLED is the ONLY experimental switch. Use same seed and settings.
source "$(dirname "${BASH_SOURCE[0]}")/geometry_paths.sh"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
GEOMETRY_ENABLED="${GEOMETRY_ENABLED:-true}"
[[ "$GEOMETRY_ENABLED" == true || "$GEOMETRY_ENABLED" == false ]] || exit 2
NNODES="${NNODES:-1}"; GPUS_PER_NODE="${GPUS_PER_NODE:-16}"; NODE_RANK="${NODE_RANK:-0}"
for n in "$NNODES" "$GPUS_PER_NODE" "$NODE_RANK"; do [[ "$n" =~ ^(0|[1-9][0-9]*)$ ]] || exit 2; done
(( NNODES > 0 && GPUS_PER_NODE > 0 && NODE_RANK < NNODES && NNODES * GPUS_PER_NODE == 16 )) || exit 2
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
if (( NNODES > 1 )); then
  : "${RUN_ID:?Use same RUN_ID on every node}" "${OUTPUT_DIR:?Use same shared OUTPUT_DIR on every node}"
  [[ "$MASTER_ADDR" != 127.0.0.1 && "$MASTER_ADDR" != localhost ]] || exit 2
fi
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then OUTPUT_DIR="\${paths.output_root}/geometry_${GEOMETRY_ENABLED}/${RUN_ID}"; fi
# Same bf16, ZeRO-1, per-rank B4, GAS2 and official train.py as baseline launcher.
command=(accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml
  --deepspeed_multinode_launcher standard --num_processes 16 --num_machines "$NNODES" --machine_rank "$NODE_RANK"
  --main_process_ip "$MASTER_ADDR" --main_process_port "${MASTER_PORT:-29500}"
  scripts/geometry_ablation.py train --paths "$GEOMETRY_PATHS"
  "geometry_enabled=${GEOMETRY_ENABLED}" "batch_size=${BATCH_SIZE:-4}" "gradient_accumulation_steps=${GRAD_ACCUM:-2}"
  "num_workers=${NUM_WORKERS:-8}" "max_steps=${MAX_STEPS:-null}" "resume=${RESUME_STATE:-null}"
  "output_dir=${OUTPUT_DIR}" "seed=${SEED:-42}" "$@")
printf '%q ' "${command[@]}"; printf '\n'
[[ "${DRY_RUN:-0}" == 1 ]] || exec "${command[@]}"
