#!/usr/bin/env python3
"""Launch official LIBERO evaluation after portable path preflight."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from libero_track4world import load_paths, print_paths, validate_paths


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--paths",
        default=str(REPO_ROOT / "configs/paths/libero_track4world_local.yaml"),
    )
    args, hydra_args = parser.parse_known_args()
    paths = load_paths(args.paths)
    print_paths(paths)
    validate_paths(paths)
    print("Launching online LIBERO evaluation after path preflight.", flush=True)

    path_file = Path(paths["_config_path"])
    expected_parent = (REPO_ROOT / "configs/paths").resolve()
    if path_file.parent != expected_parent:
        raise ValueError(f"Evaluation path config must be inside {expected_parent}")
    if not any(arg.startswith("paths=") for arg in hydra_args):
        hydra_args.append(f"paths={path_file.stem}")

    # The geometry wrapper defaults to the eager inference path that is covered
    # by the online Track4World regression and simulator smoke tests.  Keep this
    # as a CLI default (rather than changing the official sim config) so users
    # can still opt into compilation explicitly after validating it locally.
    if not any(arg.startswith("EVALUATION.compile_action_infer=") for arg in hydra_args):
        hydra_args.append("EVALUATION.compile_action_infer=false")

    libero_repo = Path(paths["libero_repo"]).resolve()
    for entry in (
        paths["track4world_extra_pythonpath"],
        str(libero_repo),
        str(REPO_ROOT / "experiments/libero"),
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    # LIBERO insists on a config file. Generate it from the one machine path
    # config instead of maintaining a second set of absolute paths.
    benchmark_root = libero_repo / "libero/libero"
    runtime_config = Path(paths["output_root"]) / ".libero_runtime"
    runtime_config.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(Path(paths["train_datasets"][0]).parent),
            "assets": str(benchmark_root / "assets"),
        },
        runtime_config / "config.yaml",
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(runtime_config)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_ENDPOINT", str(paths["hf_endpoint"]))
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(paths["model_base"])

    sys.argv = [str(REPO_ROOT / "experiments/libero/eval_libero_single.py"), *hydra_args]
    runpy.run_path(
        str(REPO_ROOT / "experiments/libero/eval_libero_single.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
