#!/usr/bin/env bash
# Inner phase launcher: $1 = output root (required), remaining args passed to run_pair.py.
set -euo pipefail
source /tmp/align_env
cd "$TARGET_REPO"
export ACTION_DIT_CHECKPOINT=/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export TEXT_CACHE=/new-interaction/share/user_folder/jiahao.ljh/data/fastwam-text-emb/libero
export DATASET_STATS="$REVIEW_ROOT/libero_reference_dataset_stats.json"
export NNODES=1 GPUS_PER_NODE=16 NODE_RANK=0
source scripts/libero_cluster_paths.sh
ROOT="${1:?output root}"
shift
exec python scripts/alignment/run_pair.py --reference-repo "$REFERENCE_REPO" --output-root "$ROOT" "$@"
