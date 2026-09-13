#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
exec "${PYTHON_BIN:-python}" "$ROOT_DIR/experiments/libero_plus/parallel_workers.py" "${1:?task file required}"
