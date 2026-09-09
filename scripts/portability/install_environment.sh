#!/usr/bin/env bash
# Default: print a plan. One-command new environment installation:
#   bash scripts/portability/install_environment.sh --execute
# Explicitly opted-in active conda/venv:
#   bash scripts/portability/install_environment.sh --active-env --execute
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: bash scripts/portability/install_environment.sh [--execute] [--env-name NAME]' \
        '       bash scripts/portability/install_environment.sh --active-env [--python PATH] [--execute]' \
        'Default: print only. --execute creates a NEW repo-local conda environment (Python 3.10).' \
        'Default name: fastwam4d-cu121, under <FastWAM>/.conda/envs/.' \
        '--active-env explicitly authorizes installation into the activated conda/venv.' \
        '--python selects that active environment interpreter; it never authorizes a global install.' \
        'No model weights or datasets are downloaded by this script.'
}

fail() { printf 'Environment setup failed: %s\n' "$*" >&2; exit 1; }
execute=0
active=0
env_name=fastwam4d-cu121
name_given=0
python_bin=''
while (($#)); do
    case "$1" in
        --execute) execute=1; shift ;;
        --active-env) active=1; shift ;;
        --env-name)
            (($# >= 2)) && [[ -n "$2" ]] || fail '--env-name needs a value'
            env_name=$2; name_given=1; shift 2 ;;
        --python)
            (($# >= 2)) && [[ -n "$2" ]] || fail '--python needs a value'
            python_bin=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
[[ "$env_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail 'Invalid environment name'
(( ! active || ! name_given )) || fail '--env-name and --active-env are mutually exclusive'
[[ -z "$python_bin" ]] || (( active )) || fail '--python requires --active-env'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$script_dir/../.." && pwd -P)
requirements="$root/requirements/fastwam4d-cu121.txt"
prefix="$root/.conda/envs/$env_name"
if (( active )); then
    prefix=${VIRTUAL_ENV:-${CONDA_PREFIX:-}}
    [[ -n "$prefix" ]] || fail '--active-env requires an activated conda environment or venv'
    python_bin=${python_bin:-$prefix/bin/python}
else
    [[ ! -e "$prefix" && ! -L "$prefix" ]] || fail "Refusing to reuse existing environment: $prefix"
    python_bin="$prefix/bin/python"
fi

run() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    if (( execute )); then "$@"; fi
}

printf 'FastWAM Python 3.10 / torch 2.5.1 / CUDA 12.1 setup\nTarget: %s\n' "$prefix"
if (( ! execute )); then
    printf 'PLAN ONLY: pass --execute to run these commands.\n'
fi

# Prevent user pip configuration or a workspace PYTHONPATH from redirecting an
# installation/import into an unrelated environment. This affects this child only.
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null
export DS_BUILD_OPS=0
if (( active && execute )); then
    [[ -x "$python_bin" ]] || fail "Not an executable interpreter: $python_bin"
    "$python_bin" - "$prefix" <<'PY'
import os
from pathlib import Path
import sys

prefix = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != prefix:
    raise SystemExit('--python must belong to the explicitly activated environment')
if sys.version_info[:2] != (3, 10):
    raise SystemExit('Python 3.10 is required; the active interpreter will not be replaced')
if sys.prefix == sys.base_prefix and not (prefix / 'conda-meta').is_dir():
    raise SystemExit('Refusing to install into a global/system interpreter')
conda_exe = os.environ.get('CONDA_EXE')
if ((prefix / 'conda-meta').is_dir() and
        (os.environ.get('CONDA_DEFAULT_ENV') == 'base' or
         (conda_exe and prefix == Path(conda_exe).resolve().parent.parent))):
    raise SystemExit('Refusing to modify the base conda environment')
print(f'Validated active Python 3.10 environment: {prefix}')
PY
fi
if (( ! active )); then
    if (( execute )); then command -v conda >/dev/null || fail 'conda is required to create a new environment'; fi
    run conda create --yes --prefix "$prefix" python=3.10 pip
fi
run "$python_bin" "$script_dir/bootstrap_sources.py" --root "$root"
run "$python_bin" -m pip install setuptools==80.10.2 wheel==0.47.0
# Install torch before DeepSpeed's metadata/build step, with no optional CUDA op build.
run "$python_bin" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121
run "$python_bin" -m pip install --no-build-isolation -r "$requirements"
run "$python_bin" -m pip install --no-build-isolation --no-deps -e "$root" -e "$root/third_party/LIBERO"
if [[ -f "$root/third_party/utils3d/pyproject.toml" || -f "$root/third_party/utils3d/setup.py" ]]; then
    run "$python_bin" -m pip install --no-build-isolation --no-deps -e "$root/third_party/utils3d"
else
    printf '  utils3d is a package at its repository root; register its parent source path.\n'
fi
printf '  Register Track4World, utils3d and Pi3 source paths in the target environment.\n'
if (( execute )); then
    "$python_bin" - "$root" <<'PY'
from pathlib import Path
import sys
import sysconfig

root = Path(sys.argv[1]).resolve()
site = Path(sysconfig.get_path('purelib')).resolve()
if not site.is_relative_to(Path(sys.prefix).resolve()):
    raise SystemExit(f'Refusing source-path registration outside target environment: {site}')
paths = [root / 'third_party', root / 'third_party/Track4World', root / 'third_party/Pi3']
if any('\n' in str(path) or '\r' in str(path) for path in paths):
    raise SystemExit('Repository paths containing newlines are unsupported')
content = ''.join(f'{path}\n' for path in paths)
destination = site / 'fastwam4d_sources.pth'
if destination.is_symlink() or (destination.exists() and destination.read_text() != content):
    raise SystemExit(f'Existing source registration differs; preserving {destination}')
if not destination.exists():
    with destination.open('x') as stream:
        stream.write(content)
print(f'Registered source paths: {destination}')
PY
    "$python_bin" - <<'PY'
import torch
import torchvision
assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__
assert torch.version.cuda == '12.1', torch.version.cuda
assert torchvision.__version__.split('+')[0] == '0.20.1', torchvision.__version__
from fastwam.models.wan22.geometry_adapter import GeometryTokenizer
from track4world.nets.model import Track4World
print('Torch/CUDA versions and geometry source imports verified (no model construction).')
PY
    if (( ! active )); then printf 'Ready. Activate with: conda activate %q\n' "$prefix"; fi
    printf 'CPU tests: %q -m pytest -q %q\n' "$python_bin" "$root/tests"
else
    printf '  Verify torch/CUDA versions and import geometry modules without constructing models.\n'
fi
