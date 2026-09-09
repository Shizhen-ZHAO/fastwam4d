#!/usr/bin/env python3
"""Print a network-free asset plan; download only with --execute.

Examples (paths default to this checkout, independently of the working directory)::

    python scripts/portability/download_assets.py
    python scripts/portability/download_assets.py --models --data
    python scripts/portability/download_assets.py --data --execute

Only the download child imports huggingface_hub. It receives HF_ENDPOINT before
Python starts, clears proxy/offline/debug settings, and uses the Hub's own cache
and interrupted-download recovery (no force_download or deprecated resume flag).
Tokens may be supplied through the usual HF environment/cache, never CLI flags.

Provenance: official HF /api/models/{repo} and /api/datasets/{repo} metadata
checked 2026-09-09. The official LIBERO GitHub README links to
https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets ; its pinned siblings
contain 130 *_demo.hdf5 files in five directories, matching all 130 local cache
.metadata revisions. No ZIP, extraction, LeRobot conversion, or model loading.

VAE/T5 use the official Wan-AI/Wan2.2-TI2V-5B .pth files, verified through
https://huggingface.co/api/models/Wan-AI/Wan2.2-TI2V-5B and pinned below. These
files use the native loader's redirect_common_files=False layout; no weight
format conversion or unrelated diffusion weights are needed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
from typing import Mapping, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENDPOINT = "https://hf-mirror.com"
LIBERO_REPO = "yifengzhu-hf/LIBERO-datasets"
LIBERO_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
LIBERO_SUITE_TASKS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
    "libero_10": 10,
}
LIBERO_ALLOW_PATTERNS = tuple(f"{suite}/*_demo.hdf5" for suite in LIBERO_SUITE_TASKS)
_REMOVED_ENV = {
    "http_proxy", "https_proxy", "all_proxy", "hf_hub_offline",
    "hf_datasets_offline", "transformers_offline", "hf_debug",
}


@dataclass(frozen=True)
class Asset:
    name: str
    repo_id: str
    revision: str
    subdir: str
    files: tuple[str, ...]
    repo_type: str = "model"
    method: str = "hf_hub_download"
    verification: str = "Verified against official HF API file listing."


MODEL_ASSETS = (
    Asset(
        "fastwam", "yuanty/fastwam", "8eaceeb24c3cc92ff2a9c9a9d266a4941b836705",
        "fastwam", ("libero_uncond_2cam224.pt", "libero_uncond_2cam224_dataset_stats.json"),
    ),
    Asset(
        "track4world", "TencentARC/Track4World", "93ae34410efd05d1b7abfdb0f749bc93f7655404",
        "track4world", ("track4world_da3.pth",),
    ),
    Asset(
        "da3", "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "b2359bdf726fb44ef62acca04d629dcf158053e7", "DA3NESTED-GIANT-LARGE-1.1",
        ("config.json", "model.safetensors"),
    ),
    Asset(
        "vae_t5", "Wan-AI/Wan2.2-TI2V-5B", "921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        "model_base/Wan-AI/Wan2.2-TI2V-5B",
        ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"),
        verification=(
            "Official HF API file listing verified on 2026-09-09. Native .pth "
            "weights for redirect_common_files=False; no format conversion."
        ),
    ),
    Asset(
        "tokenizer", "Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a",
        "model_base/Wan-AI/Wan2.1-T2V-1.3B",
        tuple(f"google/umt5-xxl/{name}" for name in (
            "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json",
        )),
    ),
)
DATA_ASSET = Asset(
    "libero", LIBERO_REPO, LIBERO_REVISION, "", LIBERO_ALLOW_PATTERNS,
    repo_type="dataset", method="snapshot_download",
    verification=(
        "Official LIBERO GitHub README links to this HF dataset; pinned API file "
        "listing and local .metadata agree: spatial/object/goal/90/10 = 10/10/10/90/10 "
        "original HDF5 tasks (130 total)."
    ),
)


def validate_endpoint(value: str) -> str:
    """Reject credentials/query strings before an endpoint can enter a plan/log."""
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
        or parsed.password is not None or parsed.query or parsed.fragment
        or parsed.path not in ("", "/") or any(c.isspace() for c in value)
        or "\\" in value or parsed.port == 0
    ):
        raise ValueError("endpoint must be an HTTPS origin without credentials, query, or path")
    return value.rstrip("/")


def validate_root(value: str | Path) -> Path:
    raw = os.fspath(value)
    if not raw.strip() or "\x00" in raw or "://" in raw:
        raise ValueError("download root must be a nonempty local directory path")
    path = Path(raw).expanduser().resolve()
    if path == Path(path.anchor):
        raise ValueError("download root must not be the filesystem root")
    for ancestor in (path, *path.parents):
        if ancestor.exists() and not ancestor.is_dir():
            raise ValueError("download root or its parent is a file")
    return path


def _relative_path(value: str, *, pattern: bool = False) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value or path.is_absolute() or PureWindowsPath(value).drive
        or "\\" in value or any(part in ("", ".", "..") for part in value.split("/"))
        or any(ord(c) < 32 for c in value) or ":" in value
        or (not pattern and any(c in value for c in "*?[]"))
    ):
        raise ValueError("unsafe repository-relative asset path")
    return path


def _check_local_path(path: Path, root: Path, *, directory: bool = False) -> None:
    """Reject escaping symlinks, file/directory collisions, and cache symlinks."""
    path.relative_to(root)
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinks inside a download destination are not supported")
        is_directory = directory or current != path
        if current.exists() and (current.is_dir() != is_directory):
            raise ValueError("asset destination has a file/directory collision")


def _validate_destination(asset: Asset, root: Path) -> Path:
    dest = root / _relative_path(asset.subdir) if asset.subdir else root
    _check_local_path(dest, root, directory=True)
    # Hub writes metadata and partial files here; check existing descendants too.
    cache = dest / ".cache"
    _check_local_path(cache, root, directory=True)
    if cache.exists():
        for base, dirs, files in os.walk(cache, followlinks=False):
            for name in (*dirs, *files):
                if (Path(base) / name).is_symlink():
                    raise ValueError("symlinks inside the Hub local cache are not supported")
    for filename in asset.files:
        relative = _relative_path(filename, pattern=asset.method == "snapshot_download")
        if asset.method == "snapshot_download":
            _check_local_path(dest / relative.parent, root, directory=True)
            for candidate in dest.glob(filename):
                _check_local_path(candidate, root)
        else:
            _check_local_path(dest / relative, root)
    return dest


def build_plan(
    assets_root: str | Path = PROJECT_ROOT / "assets_local",
    data_root: str | Path = PROJECT_ROOT / "datasets/libero",
    *, models: bool = False, data: bool = False, endpoint: str = DEFAULT_ENDPOINT,
) -> dict:
    """Only inspect paths and static metadata: no imports, mkdirs, or network."""
    endpoint = validate_endpoint(endpoint)
    assets_root, data_root = validate_root(assets_root), validate_root(data_root)
    selected = list(MODEL_ASSETS) if models or not data else []
    if data:
        selected.append(DATA_ASSET)
    entries = []
    for asset in selected:
        root = data_root if asset.repo_type == "dataset" else assets_root
        dest = _validate_destination(asset, root)
        source_prefix = "datasets/" if asset.repo_type == "dataset" else ""
        entry = {
            "name": asset.name, "repo_id": asset.repo_id, "repo_type": asset.repo_type,
            "revision": asset.revision, "revision_pinned": bool(re.fullmatch(r"[0-9a-f]{40}", asset.revision)),
            "method": asset.method, "local_dir": str(dest),
            "source": f"https://huggingface.co/{source_prefix}{asset.repo_id}/tree/{asset.revision}",
            "verification": asset.verification,
        }
        if asset.method == "snapshot_download":
            entry.update(allow_patterns=list(asset.files), suite_tasks=dict(LIBERO_SUITE_TASKS), expected_files=130)
        else:
            entry.update(files=list(asset.files), destinations=[str(dest / name) for name in asset.files])
        entries.append(entry)
    return {
        "endpoint": endpoint, "assets_root": str(assets_root), "data_root": str(data_root),
        "assets": entries,
    }


def download_environment(parent: Mapping[str, str], endpoint: str) -> dict[str, str]:
    """Return a child environment, leaving the caller's environment untouched."""
    env = {key: value for key, value in parent.items() if key.lower() not in _REMOVED_ENV}
    env.update({
        "HF_ENDPOINT": validate_endpoint(endpoint), "HF_HUB_VERBOSITY": "error",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_HUB_DISABLE_XET": "1",
    })
    return env


def _download(plan: dict) -> None:
    """Worker only: environment is already set before huggingface_hub import."""
    if os.environ.get("HF_ENDPOINT") != plan["endpoint"] or any(
        key.lower() in _REMOVED_ENV for key in os.environ
    ):
        raise RuntimeError("downloads require the configured child environment")
    from huggingface_hub import hf_hub_download, snapshot_download

    for entry in plan["assets"]:
        kwargs = {
            "repo_id": entry["repo_id"], "repo_type": entry["repo_type"],
            "revision": entry["revision"], "local_dir": entry["local_dir"],
            "endpoint": plan["endpoint"], "local_files_only": False,
        }
        if entry["method"] == "snapshot_download":
            snapshot_download(**kwargs, allow_patterns=entry["allow_patterns"])
            root = Path(entry["local_dir"])
            for suite, count in entry["suite_tasks"].items():
                paths = list((root / suite).glob("*_demo.hdf5"))
                if len(paths) != count or any(not p.is_file() or p.is_symlink() for p in paths):
                    raise RuntimeError("LIBERO suite file count does not match the pinned manifest")
        else:
            for filename in entry["files"]:
                hf_hub_download(**kwargs, filename=filename)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse normally echoes arbitrary arguments (possibly pasted secrets).
        super().error("invalid arguments; use --help for supported options")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="download; default only prints the plan")
    parser.add_argument("--models", action="store_true", help="include the model assets (default if neither selected)")
    parser.add_argument("--data", action="store_true", help="include all five original LIBERO HDF5 suites; may combine with --models")
    parser.add_argument("--assets-root", default=str(PROJECT_ROOT / "assets_local"))
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "datasets/libero"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="HTTPS Hub origin (default: %(default)s)")
    parser.add_argument("--_download-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._download_worker and not args.execute:
        parser.error("worker requires execute")
    try:
        plan = build_plan(args.assets_root, args.data_root, models=args.models, data=args.data, endpoint=args.endpoint)
    except (ValueError, OSError, RuntimeError):
        print("Invalid endpoint or download path; check roots, symlinks, and file/directory collisions.", file=sys.stderr)
        return 2
    if args._download_worker:
        try:
            _download(plan)
        except Exception:
            # Never echo exception text: libraries may include tokens or request URLs.
            return 1
        return 0
    plan["dry_run"] = not args.execute
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        return 0
    command = [
        sys.executable, str(Path(__file__).resolve()), "--execute", "--_download-worker",
        "--assets-root", plan["assets_root"], "--data-root", plan["data_root"],
        "--endpoint", plan["endpoint"],
    ]
    if args.models:
        command.append("--models")
    if args.data:
        command.append("--data")
    try:
        result = subprocess.run(
            command, env=download_environment(os.environ, plan["endpoint"]),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        print("Could not start the asset download child process.", file=sys.stderr)
        return 1
    if result.returncode:
        print(
            "Asset download failed. Check huggingface_hub installation, source access, "
            "mirror connectivity, disk space, and the manifest's verification notes. "
            "Re-run --execute to resume using the Hub cache; library output is hidden to protect credentials.",
            file=sys.stderr,
        )
        return 1
    print("Requested asset downloads completed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
