#!/usr/bin/env python3
"""Portable Track4World feature-cache tools for FastWAM LIBERO.

This entry point deliberately reads every machine-specific location from one
paths YAML. It never downloads checkpoints; all model files must already exist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from omegaconf import OmegaConf
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def load_paths(path: str) -> dict:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing paths config: {config_path}")
    value = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError("Paths YAML must contain a mapping")
    value["_config_path"] = str(config_path)
    return value


def geometry_config(paths: dict, *, device: str | None = None) -> dict:
    # Extraction and model training must never maintain duplicate geometry
    # hyperparameters. Resolve the exact model config against this machine's
    # path file, then optionally override only the extraction device.
    model_config = OmegaConf.load(REPO_ROOT / "configs/model/fastwam_geometry.yaml")
    root = OmegaConf.create({"paths": paths, "model": model_config})
    config = OmegaConf.to_container(root.model.geometry, resolve=True)
    if not isinstance(config, dict):
        raise ValueError("model.geometry must resolve to a mapping")
    if device is not None:
        config["extractor"]["device"] = device
    return config


def path_rows(paths: dict):
    scalar_keys = (
        "text_embedding_cache",
        "fastwam_checkpoint",
        "dataset_stats",
        "model_base",
        "track4world_repo",
        "track4world_checkpoint",
        "da3_model",
        "track4world_extra_pythonpath",
        "libero_repo",
        "geometry_cache",
        "output_root",
    )
    rows = [("paths_config", paths["_config_path"])]
    rows.extend((f"train_dataset[{i}]", value) for i, value in enumerate(paths["train_datasets"]))
    rows.extend((key, paths[key]) for key in scalar_keys)
    rows.append(("geometry_device", paths["geometry_device"]))
    rows.append(("hf_endpoint", paths["hf_endpoint"]))
    return rows


def print_paths(paths: dict):
    print("Resolved machine paths (printed before execution):")
    width = max(len(key) for key, _ in path_rows(paths))
    for key, value in path_rows(paths):
        print(f"  {key:<{width}}  {value}")


def validate_paths(paths: dict, *, cache_required: bool = False):
    files = ("fastwam_checkpoint", "dataset_stats", "track4world_checkpoint")
    directories = (
        "model_base",
        "track4world_repo",
        "da3_model",
        "track4world_extra_pythonpath",
        "libero_repo",
        "text_embedding_cache",
    )
    failures = []
    for key in files:
        if not Path(paths[key]).expanduser().is_file():
            failures.append(f"{key} is not a file: {paths[key]}")
    for key in directories:
        if not Path(paths[key]).expanduser().is_dir():
            failures.append(f"{key} is not a directory: {paths[key]}")
    track4world_root = Path(paths["track4world_repo"]).expanduser()
    for relative in (
        "track4world/nets/model.py",
        "track4world/nets/external/depth_anything_3/api.py",
        "track4world/nets/external/pi3/models/pi3.py",
        "utils3d/__init__.py",
    ):
        if not (track4world_root / relative).is_file():
            failures.append(f"track4world_repo missing {relative}: {track4world_root}")
    for index, root in enumerate(paths["train_datasets"]):
        root = Path(root).expanduser()
        for relative in ("meta/info.json", "meta/episodes.jsonl", "data", "videos"):
            if not (root / relative).exists():
                failures.append(f"train_dataset[{index}] missing {relative}: {root}")
    if cache_required and not (Path(paths["geometry_cache"]) / "manifest.json").is_file():
        failures.append(f"geometry cache manifest is missing: {paths['geometry_cache']}")
    if failures:
        raise FileNotFoundError("Path preflight failed:\n  " + "\n  ".join(failures))


def build_history(paths: dict):
    from fastwam.datasets.libero_geometry import LeRobotGeometryHistoryDataset

    config = geometry_config(paths)
    return LeRobotGeometryHistoryDataset(
        paths["train_datasets"],
        history_length=config["history_length"],
        history_stride=config["history_stride"],
        image_size=config["extractor"]["image_size"],
    )


def selected_indices(length: int, args) -> range:
    if getattr(args, "indices", None):
        values = [int(value) for value in args.indices.split(",")]
        if any(value < 0 or value >= length for value in values):
            raise ValueError(f"Every explicit index must be in [0, {length})")
        if len(values) != len(set(values)):
            raise ValueError("Explicit indices must be unique")
        return values
    start = int(args.start)
    stop = length if args.count is None else min(length, start + int(args.count))
    if not 0 <= start < length or stop <= start:
        raise ValueError(f"Invalid range start={start}, stop={stop}, dataset length={length}")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    total = stop - start
    chunk = (total + args.num_shards - 1) // args.num_shards
    shard_start = min(stop, start + args.shard_id * chunk)
    shard_stop = min(stop, shard_start + chunk)
    return range(shard_start, shard_stop)


def command_doctor(paths: dict, args):
    validate_paths(paths, cache_required=args.require_cache)
    repo = Path(paths["track4world_repo"]).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import utils3d
        from track4world.nets import model as track4world_model
        from fastwam.models.wan22.track4world_online import require_module_origin

        require_module_origin(track4world_model, repo / "track4world/nets/model.py")
        require_module_origin(utils3d, repo / "utils3d/__init__.py")
    except Exception as exc:
        raise RuntimeError(
            "Track4World Python import preflight failed. Complete its utils3d/Pi3 setup "
            "and install runtime dependencies in the FastWAM environment."
        ) from exc
    history = build_history(paths)
    totals = []
    for root, dataset in zip(paths["train_datasets"], history.dataset._datasets):
        totals.append({"root": root, "frames": len(dataset), "episodes": dataset.meta.total_episodes})
    expected_fps = float(geometry_config(paths)["history_fps"])
    if history.fps != expected_fps:
        raise ValueError(
            f"Geometry history_fps={expected_fps} but LeRobot data fps={history.fps}"
        )
    print(json.dumps({
        "status": "ok",
        "fps": history.fps,
        "total_frames": len(history),
        "track4world_model": str(Path(track4world_model.__file__).resolve()),
        "utils3d": str(Path(utils3d.__file__).resolve()),
        "datasets": totals,
    }, indent=2))


def command_extract(paths: dict, args):
    validate_paths(paths)
    if not torch.cuda.is_available():
        raise RuntimeError("Track4World extraction requires CUDA")
    history = build_history(paths)
    config = geometry_config(paths, device=args.device)
    from fastwam.datasets.geometry_cache import LeRobotGeometryCache
    from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor

    print("Fingerprinting dataset/video bytes, model weights and extraction code for cache v2...", flush=True)
    cache = LeRobotGeometryCache(paths["geometry_cache"], history, config, create=True)
    indices = selected_indices(len(history), args)
    extractor = OnlineTrack4WorldExtractor(**config["extractor"])
    written = skipped = 0
    started = time.monotonic()
    try:
        for position, index in enumerate(indices, start=1):
            if cache.contains(index) and not args.verify_existing:
                skipped += 1
                continue
            sample = history[index]
            online_inputs = {
                "images": sample["history_images"].unsqueeze(0),
                "timestamps": sample["history_timestamps"].unsqueeze(0),
                "valid": sample["history_valid"].unsqueeze(0),
            }
            raw = extractor(**online_inputs, output_device="cpu")
            raw = {key: value[0] for key, value in raw.items()}
            if cache.contains(index):
                cached = cache.read(index)
                for key in raw:
                    if not torch.equal(raw[key], cached[key]):
                        raise ValueError(f"Existing cache differs at index={index}, key={key}")
                skipped += 1
            else:
                cache.write(index, raw)
                cached = cache.read(index)
                if any(not torch.equal(raw[key], cached[key]) for key in raw):
                    raise ValueError(f"Lossless cache round trip failed at index={index}")
                written += 1
            if position == 1 or position % args.log_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"progress={position}/{len(indices)} index={index} written={written} "
                    f"skipped={skipped} last_extract_s={extractor.last_seconds:.2f} "
                    f"elapsed_s={elapsed:.1f} quality={extractor.last_quality}",
                    flush=True,
                )
    finally:
        extractor.close()
    print(json.dumps({"status": "ok", "written": written, "skipped": skipped, "seconds": time.monotonic() - started}, indent=2))


def command_verify(paths: dict, args):
    validate_paths(paths, cache_required=True)
    history = build_history(paths)
    from fastwam.datasets.geometry_cache import LeRobotGeometryCache

    config = geometry_config(paths)
    cache = LeRobotGeometryCache(paths["geometry_cache"], history, config, create=False)
    indices = selected_indices(len(history), args)
    checked = 0
    for index in indices:
        raw = cache.read(index)
        identity = history.metadata(index)
        points = int(config["extractor"]["grid_size"]) ** 2
        if raw["scene"].shape[:2] != (config["num_views"], points) or raw["track"].shape[:3] != (
            config["num_views"], config["history_length"], points
        ):
            raise ValueError(f"Unexpected cached tensor shapes at index={index}")
        if identity["history_frame_indices"][-1] != identity["frame_index"]:
            raise ValueError(f"History does not end at current frame for index={index}")
        checked += 1
    print(json.dumps({"status": "ok", "checked": checked}, indent=2))


def command_coverage(paths: dict, args):
    validate_paths(paths, cache_required=True)
    history = build_history(paths)
    from fastwam.datasets.geometry_cache import LeRobotGeometryCache

    cache = LeRobotGeometryCache(paths["geometry_cache"], history, geometry_config(paths), create=False)
    coverage = cache.coverage()
    coverage["status"] = "complete" if coverage["missing_frames"] == 0 else "incomplete"
    print(json.dumps(coverage, indent=2))
    if args.require_complete and coverage["missing_frames"]:
        raise SystemExit(2)


def command_parity(paths: dict, args):
    """Re-extract selected RGB histories and compare with their offline cache."""
    if args.indices is None and args.count is None:
        raise ValueError("parity requires --indices or --count; full-dataset parity is expensive")
    validate_paths(paths, cache_required=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Track4World parity validation requires CUDA")
    history = build_history(paths)
    config = geometry_config(paths, device=args.device)
    from fastwam.datasets.geometry_cache import LeRobotGeometryCache
    from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor

    cache = LeRobotGeometryCache(paths["geometry_cache"], history, config, create=False)
    indices = selected_indices(len(history), args)
    extractor = OnlineTrack4WorldExtractor(**config["extractor"])
    report = {}
    failures = []
    try:
        for index in indices:
            sample = history[index]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                online = extractor(
                    sample["history_images"].unsqueeze(0),
                    sample["history_timestamps"].unsqueeze(0),
                    sample["history_valid"].unsqueeze(0),
                    output_device="cpu",
                )
            online = {key: value[0] for key, value in online.items()}
            offline = cache.read(index)
            for key in sorted(offline):
                actual, expected = online[key], offline[key]
                key_report = report.setdefault(
                    key,
                    {
                        "max_mask_flips": 0,
                        "max_relative_rmse": 0.0,
                        "min_cosine": 1.0,
                    },
                )
                if actual.dtype == torch.bool:
                    flips = int((actual != expected).sum())
                    key_report["max_mask_flips"] = max(key_report["max_mask_flips"], flips)
                    if flips:
                        failures.append(f"index={index} key={key} mask_flips={flips}")
                    continue
                actual = actual.float().flatten()
                expected = expected.float().flatten()
                if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
                    failures.append(f"index={index} key={key} contains non-finite values")
                    continue
                rmse = (actual - expected).square().mean().sqrt()
                reference_rms = expected.square().mean().sqrt()
                relative_rmse = float(rmse / reference_rms.clamp_min(1e-12))
                norm_product = actual.square().sum().sqrt() * expected.square().sum().sqrt()
                cosine = 1.0 if norm_product <= 1e-12 else float((actual @ expected) / norm_product)
                key_report["max_relative_rmse"] = max(
                    key_report["max_relative_rmse"], relative_rmse
                )
                key_report["min_cosine"] = min(key_report["min_cosine"], cosine)
                if relative_rmse > args.max_relative_rmse or cosine < args.min_cosine:
                    failures.append(
                        f"index={index} key={key} relative_rmse={relative_rmse:.6g} "
                        f"cosine={cosine:.9f}"
                    )
            print(f"parity_checked index={index}", flush=True)
    finally:
        extractor.close()
    result = {
        "status": "ok" if not failures else "failed",
        "checked": len(indices),
        "max_relative_rmse_limit": args.max_relative_rmse,
        "min_cosine_limit": args.min_cosine,
        "features": report,
        "failures": failures,
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(3)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(REPO_ROOT / "configs/paths/libero_track4world_local.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--require-cache", action="store_true")
    coverage = subparsers.add_parser("coverage")
    coverage.add_argument("--require-complete", action="store_true")
    for name in ("extract", "verify", "parity"):
        command = subparsers.add_parser(name)
        command.add_argument("--start", type=int, default=0)
        command.add_argument("--count", type=int)
        command.add_argument("--num-shards", type=int, default=1)
        command.add_argument("--shard-id", type=int, default=0)
        command.add_argument("--indices", help="comma-separated exact global sample indices")
    extract = subparsers.choices["extract"]
    extract.add_argument("--device")
    extract.add_argument("--log-every", type=int, default=10)
    extract.add_argument("--verify-existing", action="store_true")
    parity = subparsers.choices["parity"]
    parity.add_argument("--device")
    parity.add_argument("--max-relative-rmse", type=float, default=5e-3)
    parity.add_argument("--min-cosine", type=float, default=0.999)
    return parser


def main():
    args = make_parser().parse_args()
    paths = load_paths(args.paths)
    print_paths(paths)
    extra_path = str(Path(paths["track4world_extra_pythonpath"]).expanduser().resolve())
    if extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    os.environ.setdefault("HF_ENDPOINT", str(paths["hf_endpoint"]))
    command = {
        "doctor": command_doctor,
        "extract": command_extract,
        "verify": command_verify,
        "coverage": command_coverage,
        "parity": command_parity,
    }[args.command]
    command(paths, args)


if __name__ == "__main__":
    main()
