"""Full LIBERO causal-window extraction with a directly usable training config.

All FIVE suites, all episodes and every decision frame by default, including
episode tails. No VAE/T5/FastWAM is loaded. One compressed HDF5 file per episode.
plan is read-only; extract/verify/status support disjoint episode-level shards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate

from experiment_vae_geometry_cached import load_config, seed_all
from fastwam.datasets.libero_geometry import CAMERA_KEYS, LiberoHDF5HistoryDataset
from fastwam.datasets.geometry_cache import atomic_create
from fastwam.datasets.geometry_cache_hdf5 import HDF5GeometryFeatureCache


SUITE_TASKS = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
               "libero_10": 10, "libero_90": 90}
DEFAULT_DATA = "/mnt/homes/zhaoshizhen/datasets/libero/official_hf"
DEFAULT_CACHE = "outputs/libero_geometry_full_hdf5"


def discover_files(root, suites, *, smoke=False):
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("Select nonempty, unique suite names")
    result = []
    for suite in suites:
        if suite not in SUITE_TASKS:
            raise ValueError(f"Unknown LIBERO suite: {suite}")
        files = sorted((Path(root) / suite).glob("*_demo.hdf5"))
        if len(files) != SUITE_TASKS[suite]:
            raise ValueError(f"Incomplete {suite}: found {len(files)} tasks, expected {SUITE_TASKS[suite]}")
        result.extend(files[:1] if smoke else files)
    return [str(p.resolve()) for p in result]


def scan_inventory(files, stride, horizon, *, episode_ids=None):
    """Validate every selected episode's keys and lengths without loading RGB."""
    result = {}
    for source in files:
        suite = Path(source).parent.name
        row = result.setdefault(suite, dict(tasks=0, episodes=0, decision_frames=0,
                                            extraction_windows=0, training_eligible_windows=0))
        row["tasks"] += 1
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            names = sorted(data, key=lambda s: int(s.rsplit("_", 1)[-1]))
            actual_ids = {int(s.rsplit("_", 1)[-1]) for s in names}
            if actual_ids != set(range(50)):
                raise ValueError(f"Expected all 50 demo_0..demo_49 episodes in {source}")
            for name in names:
                if episode_ids is not None and int(name.rsplit("_", 1)[-1]) not in episode_ids:
                    continue
                demo = data[name]
                n = len(demo["actions"])
                if demo["actions"].shape != (n, 7) or n < 1:
                    raise ValueError(f"Invalid action shape: {source}:{name}")
                for key in CAMERA_KEYS:
                    image = demo[f"obs/{key}"]
                    if image.ndim != 4 or image.shape[-1] != 3 or len(image) < n or image.dtype.name != "uint8":
                        raise ValueError(f"Invalid RGB data: {source}:{name}:{key}")
                for key, dim in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
                    value = demo[f"obs/{key}"]
                    if len(value) < n or value.shape[1:] != (dim,):
                        raise ValueError(f"Invalid proprio data: {source}:{name}:{key}")
                row["episodes"] += 1
                row["decision_frames"] += n
                row["extraction_windows"] += len(range(0, n, stride))
                row["training_eligible_windows"] += len(range(0, n - horizon, stride))
    return result


def shard_for_episode(source, episode, shards):
    # Includes suite/task to avoid collisions across demo_0 in different tasks.
    digest = hashlib.sha256(f"{source}\0{episode}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % shards


def selected_work(dataset, shard_index, shards, *, smoke=False):
    indices = []
    for i, (source, episode, t) in enumerate(dataset.samples):
        if shard_for_episode(source, episode, shards) != shard_index:
            continue
        if smoke:
            n = dataset.episode_lengths[(source, episode)]
            if t not in {0, (dataset.history_length - 1) * dataset.history_stride, n - 1}:
                continue
        indices.append(i)
    return indices


def raw_bytes_per_window(geometry):
    v, p, length = int(geometry.num_views), int(geometry.extractor.grid_size) ** 2, int(geometry.history_length)
    floats = v * (p * (1024 + 6) + 3072 + 13 + length * p * (256 + 11))
    bools = v * (p + 1 + length * p)
    return 4 * floats + bools


def nearest_existing(path):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    return path


def create_if_equal(path, value):
    """Immutable shared config, safe for simultaneous shard workers."""
    if Path(path).exists():
        old = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if old != value:
            raise ValueError(f"Existing generated configuration differs: {path}; use a new cache directory")
        return
    try:
        atomic_create(path, text_value=OmegaConf.to_yaml(OmegaConf.create(value)))
    except FileExistsError:
        create_if_equal(path, value)


def make_configs(args, files):
    cfg = load_config(args.base_config)
    cfg.cache_backend = "hdf5"
    cfg.cache_dir = str(Path(args.cache_dir).resolve())
    cfg.output_dir = "outputs/libero_full_cached_train" if not args.smoke else "outputs/libero_full_cached_smoke_train"
    cfg.online_test_output_dir = cfg.output_dir + "_online_test"
    cfg.data.files = files
    cfg.data.sample_stride = args.sample_stride
    cfg.data.load_history = False
    cfg.data.history_only = False
    cfg.train_samples = 0
    cfg.train_indices = None
    cfg.num_workers = args.num_workers
    cfg.steps = args.train_steps
    cfg.verify_cache_before_training = False
    cfg.validation_episode_ids = list(args.validation_episode_ids)
    cfg.data.episode_ids = ([0] if args.smoke else
                            [i for i in range(50) if i not in args.validation_episode_ids])
    if not cfg.data.episode_ids or (args.smoke and 0 in args.validation_episode_ids):
        raise ValueError("No training episodes remain, or smoke demo_0 overlaps validation")
    # Extraction covers every episode, including held-out episodes, but never
    # uses their future frames. Caching is not training: the split is below.
    extract_args = OmegaConf.to_container(cfg.data, resolve=True)
    extract_args.update(history_only=True, episode_ids=[0] if args.smoke else None)
    dataset = LiberoHDF5HistoryDataset(**extract_args)
    if args.smoke:
        training = LiberoHDF5HistoryDataset(**OmegaConf.to_container(cfg.data, resolve=True))
        t = (dataset.history_length - 1) * dataset.history_stride
        cfg.train_indices = [i for i, (_, ep, frame) in enumerate(training.samples) if ep == "demo_0" and frame == t]
        if len(cfg.train_indices) != len(files):
            raise ValueError("Smoke requires a full-history training window in each selected task")
    return cfg, dataset


def run_extraction(cache, indices, *, device, min_free_gib=50, log_every=25,
                   verify_existing=False, extractor_factory=None):
    """Shared real and unit-test loop. Only the factory differs in CPU tests."""
    completed = cache.completed_indices(indices)
    if verify_existing:
        for index in sorted(completed):
            cache.read(index)
    extractor = None
    created, bytes_raw, extraction_seconds = 0, 0, 0.0
    start = time.monotonic()
    try:
        for position, index in enumerate(indices):
            if index in completed:
                continue
            free = shutil.disk_usage(cache.root).free
            if free < min_free_gib * 2**30:
                raise RuntimeError(f"Free space {free / 2**30:.1f} GiB is below reserve; stopped safely before next window")
            if extractor is None:
                if extractor_factory is None:
                    from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor
                    extractor_factory = OnlineTrack4WorldExtractor
                # Keep the online device in generated training/checkpoint config
                # stable; shard extraction devices may be different.
                ext = dict(cache.extraction_options)
                ext["device"] = device
                extractor = extractor_factory(**ext)
            history = default_collate([cache.dataset.history_item(index)])
            raw_batched = extractor(history["history_images"], history["history_timestamps"], history["history_valid"])
            raw = {k: v[0] for k, v in raw_batched.items()}
            try:
                cache.write(index, raw)
                created += 1
            except FileExistsError:
                # Accidentally duplicated workers cannot overwrite a commit;
                # don't demand bitwise equality of two independent FP16 runs.
                cache.read(index)
                continue
            stored = cache.read(index)
            for key in raw:
                torch.testing.assert_close(raw[key].cpu(), stored[key], rtol=0, atol=0, equal_nan=True)
            extraction_seconds += float(extractor.last_seconds)
            bytes_raw += sum(v.numel() * v.element_size() for v in raw.values())
            if created == 1 or created % log_every == 0 or position == len(indices) - 1:
                print(json.dumps({"kind": "progress", "created": created, "already_present": len(completed),
                                  "position": position + 1, "assigned_windows": len(indices),
                                  "source": cache.dataset.samples[index],
                                  "mean_extraction_seconds": extraction_seconds / created,
                                  "elapsed_seconds": time.monotonic() - start,
                                  "free_gib": free / 2**30}), flush=True)
    finally:
        if extractor is not None:
            extractor.close()
    return {"created": created, "resumed": len(completed), "raw_bytes_created": bytes_raw,
            "extractor_calls": getattr(extractor, "calls", 0), "seconds": time.monotonic() - start,
            "mean_extraction_seconds": extraction_seconds / created if created else None,
            "tracker_initialized": extractor is not None, "serialization_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "prepare", "extract", "verify", "status"), default="plan")
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--base-config", default="configs/experiments/libero_vae_geometry_cached.yaml")
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITE_TASKS), default=list(SUITE_TASKS))
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--sample-stride", type=int, default=1, help="Default extracts every decision frame, including tails")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--validation-episode-ids", nargs="+", type=int, default=[45, 46, 47, 48, 49])
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=50)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-windows", type=int, help="Debug-only prefix limit; never marks an incomplete shard complete")
    parser.add_argument("--verify-existing", action="store_true", help="Also reread tensor checksums when resuming")
    parser.add_argument("--smoke", action="store_true", help="Explicitly partial: first task per suite, demo_0, start/history/tail")
    args = parser.parse_args()
    if (args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards or args.sample_stride < 1
            or args.num_workers < 0 or args.train_steps < 1 or args.log_every < 1 or args.min_free_gib < 0
            or (args.max_windows is not None and args.max_windows < 1)):
        parser.error("Invalid stride, shard, worker, step, reserve or limit")
    if (not set(args.validation_episode_ids) < set(range(50))
            or len(set(args.validation_episode_ids)) != len(args.validation_episode_ids)):
        parser.error("Validation must be unique episode IDs 0..49, leaving training episodes")
    if args.smoke and (args.cache_dir == DEFAULT_CACHE or args.sample_stride != 1):
        parser.error("--smoke requires a separate --cache-dir and sample-stride=1")
    torch.set_num_threads(4)
    files = discover_files(args.data_root, args.suites, smoke=args.smoke)
    cfg, dataset = make_configs(args, files)
    inventory = scan_inventory(files, args.sample_stride, int(cfg.data.horizon), episode_ids=[0] if args.smoke else None)
    all_indices = selected_work(dataset, args.shard_index, args.num_shards, smoke=args.smoke)
    indices = all_indices if args.max_windows is None else all_indices[:args.max_windows]
    raw_bytes = raw_bytes_per_window(cfg.geometry)
    free = shutil.disk_usage(nearest_existing(args.cache_dir)).free
    plan = {"mode": args.mode, "suites": inventory, "files": len(files),
            "episodes": len(dataset.episode_lengths), "source_decision_windows": len(dataset),
            "assigned_windows": len(all_indices), "selected_windows_this_run": len(indices),
            "smoke": args.smoke, "sample_stride": args.sample_stride, "includes_episode_tails": True,
            "all_five_suites": set(args.suites) == set(SUITE_TASKS) and not args.smoke,
            "cache_dir": cfg.cache_dir, "shard_index": args.shard_index, "num_shards": args.num_shards,
            "raw_bytes_per_window": raw_bytes, "uncompressed_payload_bytes_all_selected_sources": len(dataset) * raw_bytes,
            "uncompressed_payload_bytes_this_run": len(indices) * raw_bytes,
            "disk_free_bytes": free, "compression": "lossless LZF + byte shuffle, FP32 + bool",
            "disk_warning": "Compressed size is data-dependent; uncompressed payload alone exceeds free space"
                            if len(dataset) * raw_bytes > free else None,
            "training_config": str(Path(cfg.cache_dir) / "training_config.yaml"),
            "train_episode_ids": list(cfg.data.episode_ids), "validation_episode_ids": list(cfg.validation_episode_ids)}
    print(json.dumps(plan, indent=2), flush=True)
    if args.mode == "plan":
        return
    geometry = OmegaConf.to_container(cfg.geometry, resolve=True)
    cache = HDF5GeometryFeatureCache(cfg.cache_dir, dataset, geometry, create=args.mode in {"prepare", "extract"})
    cache.extraction_options = geometry["extractor"]
    if args.mode in {"prepare", "extract"}:
        create_if_equal(Path(cfg.cache_dir) / "training_config.yaml", OmegaConf.to_container(cfg, resolve=True))
        create_if_equal(Path(cfg.cache_dir) / "dataset_inventory.yaml", inventory)
    if args.mode == "prepare":
        return
    seed_all(int(cfg.seed))
    run = {}
    if args.mode == "extract":
        run = run_extraction(cache, indices, device=args.device, min_free_gib=args.min_free_gib,
                             log_every=args.log_every, verify_existing=args.verify_existing)
    elif args.mode == "verify":
        for position, index in enumerate(indices):
            cache.read(index)
            if (position + 1) % args.log_every == 0:
                print(json.dumps({"verified": position + 1, "selected": len(indices)}), flush=True)
        run = {"verified_windows": len(indices), "tracker_initialized": False}
    completed = cache.completed_indices(all_indices)
    run.update(kind="completion", mode=args.mode, shard_index=args.shard_index, num_shards=args.num_shards,
               assigned_windows=len(all_indices), completed_windows=len(completed),
               shard_complete=len(completed) == len(all_indices), smoke=args.smoke,
               entire_selected_cache_complete=(args.num_shards == 1 and len(completed) == len(all_indices)),
               cache_contract_id=cache.contract_id)
    run["full_libero_complete"] = (not args.smoke and args.sample_stride == 1
                                    and set(args.suites) == set(SUITE_TASKS)
                                    and run["entire_selected_cache_complete"])
    # Every run gets a new report. A prefix smoke/partial run cannot overwrite a
    # misleading global COMPLETE marker. status and verify do not write cache.
    if args.mode == "extract":
        atomic_create(Path(cfg.cache_dir) / "reports" / f"shard{args.shard_index}_{time.time_ns()}.json", json_value=run)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
