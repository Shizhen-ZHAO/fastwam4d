#!/usr/bin/env bash
set -euo pipefail
# 三个训练入口使用同一份路径、环境、拓扑及启动实现。
export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
exec bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_joint_2cam224_1e-4 "$@"
