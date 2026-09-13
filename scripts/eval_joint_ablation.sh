#!/usr/bin/env bash
set -euo pipefail
export FASTWAM_VARIANT=joint
exec bash "$(dirname "${BASH_SOURCE[0]}")/eval_ablation.sh" "$@"
