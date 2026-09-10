#!/usr/bin/env python3
"""Launch geometry training after printing and validating all machine paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from libero_track4world import load_paths, print_paths, validate_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("offline", "online"), default="offline")
    parser.add_argument(
        "--paths",
        default=str(REPO_ROOT / "configs/paths/libero_track4world_local.yaml"),
    )
    args, hydra_args = parser.parse_known_args()
    paths = load_paths(args.paths)
    print_paths(paths)
    validate_paths(paths, cache_required=args.mode == "offline")
    print(f"Launching {args.mode} geometry training after path preflight.", flush=True)

    path_file = Path(paths["_config_path"])
    expected_parent = (REPO_ROOT / "configs/paths").resolve()
    if path_file.parent != expected_parent:
        raise ValueError(f"Training path config must be inside {expected_parent}")
    if not any(arg.startswith("paths=") for arg in hydra_args):
        hydra_args.append(f"paths={path_file.stem}")
    if not any(arg.startswith("task=") for arg in hydra_args):
        hydra_args.append(f"task=libero_geometry_{args.mode}_2cam224")

    extra_path = str(Path(paths["track4world_extra_pythonpath"]).resolve())
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    os.environ.setdefault("HF_ENDPOINT", str(paths["hf_endpoint"]))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(paths["model_base"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    sys.argv = [str(REPO_ROOT / "scripts/train.py"), *hydra_args]
    runpy.run_path(str(REPO_ROOT / "scripts/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
