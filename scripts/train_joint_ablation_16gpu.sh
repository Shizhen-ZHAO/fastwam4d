#!/usr/bin/env bash
set -euo pipefail
export FASTWAM_VARIANT=joint
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_ablation_16gpu.sh" "$@"
