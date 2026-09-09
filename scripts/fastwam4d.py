#!/usr/bin/env python3
"""Portable front door: configure, extract, train offline/online, test, rollout.

Does not import Torch, construct models, or use a cache when printing --dry-run.
Paths in the generated configuration are local to this machine. Legacy scripts
remain available, but their old hard-coded launch wrappers are not used here.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
          "libero_10": 10, "libero_90": 90}
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def absolute(path, root=ROOT):
    value = Path(path).expanduser()
    return str((value if value.is_absolute() else root / value).resolve())


def discover(data_root, suites):
    if not suites or len(set(suites)) != len(suites):
        raise ValueError("Choose distinct nonempty LIBERO suites")
    files = []
    for suite in suites:
        found = sorted((Path(data_root) / suite).glob("*_demo.hdf5"))
        if len(found) != SUITES[suite]:
            raise ValueError(f"{suite}: expected {SUITES[suite]} HDF5 task files, got {len(found)}; download data first")
        files.extend(str(p.resolve()) for p in found)
    return files


def resolve_config(cfg, root=ROOT):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if "paths" not in cfg:
        raise ValueError("Use the portable config or a full-cache config generated from it")
    for key in cfg.paths:
        cfg.paths[key] = absolute(cfg.paths[key], root)
    for key in ("cache_dir", "output_dir", "online_test_output_dir"):
        cfg[key] = absolute(cfg[key], root)
    cfg.data.text_cache_dir = absolute(cfg.data.text_cache_dir, root)
    cfg.data.files = [absolute(p, root) for p in cfg.data.files]
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def child_environment(cfg, root=ROOT, environ=None):
    env = dict(os.environ if environ is None else environ)
    for key in PROXY_KEYS:
        env.pop(key, None)
    # This pinned utils3d repository IS the package. Add its parent, never the
    # package itself: its io/numpy/torch directories would shadow Python modules.
    search = [str(root), str(root / "src"), cfg.paths.track4world_repo,
              str(Path(cfg.paths.utils3d_repo).parent), cfg.paths.libero_repo]
    if env.get("PYTHONPATH"):
        search.append(env["PYTHONPATH"])
    env.update(PYTHONPATH=os.pathsep.join(search), HF_ENDPOINT="https://hf-mirror.com",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true",
               DIFFSYNTH_MODEL_BASE_PATH=cfg.model_base_path,
               TOKENIZERS_PARALLELISM="false")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MUJOCO_GL", "egl")
    # Never write ~/.libero; the launcher initializes its own config directory.
    env["LIBERO_CONFIG_PATH"] = str(root / "configs" / "local" / "libero_runtime")
    return env


def publish_config(path, cfg):
    path = Path(path)
    if path.exists():
        if OmegaConf.to_container(OmegaConf.load(path), resolve=True) != OmegaConf.to_container(cfg, resolve=True):
            raise FileExistsError(f"Refusing to replace {path}; choose a new config path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(OmegaConf.to_yaml(cfg))


def prepare_libero_config(cfg, env):
    base = Path(cfg.paths.libero_repo) / "libero" / "libero"
    paths = dict(benchmark_root=str(base), bddl_files=str(base / "bddl_files"),
                 init_states=str(base / "init_files"), assets=str(base / "assets"),
                 datasets=str(cfg.paths.data_root))
    publish_config(Path(env["LIBERO_CONFIG_PATH"]) / "config.yaml", OmegaConf.create(paths))


def build_command(mode, cfg, config_path, args, extra, root=ROOT):
    python = sys.executable
    if mode in {"plan", "extract", "status", "verify"}:
        cmd = [python, str(root / "scripts/extract_libero_geometry_full.py"),
               "--mode", mode, "--base-config", str(config_path),
               "--data-root", cfg.paths.data_root, "--cache-dir", absolute(args.cache_dir, root) if args.cache_dir else cfg.cache_dir,
               "--device", cfg.geometry.extractor.device, "--sample-stride", str(cfg.data.sample_stride),
               "--train-steps", str(cfg.steps), "--num-workers", str(cfg.num_workers),
               "--validation-episode-ids", *map(str, cfg.validation_episode_ids),
               "--suites", *list(cfg.portable_suites)]
        return cmd + extra
    if mode == "rollout-online":
        if not args.adapter:
            raise ValueError("rollout-online requires --adapter")
        return [python, str(root / "experiments/libero/eval_libero_single.py"),
                f"ckpt={cfg.base_checkpoint}", f"EVALUATION.dataset_stats_path={cfg.data.stats_path}",
                f"model.redirect_common_files={str(bool(cfg.get('redirect_common_files', True))).lower()}",
                f"+EVALUATION.geometry_adapter={absolute(args.adapter, root)}",
                f"EVALUATION.device={cfg.model_device}",
                f"EVALUATION.output_dir={absolute(args.output_dir or 'outputs/libero_rollout', root)}",
                f"EVALUATION.num_inference_steps={args.inference_steps or cfg.inference_steps}",
                f"EVALUATION.task_suite_name={args.suite}", f"EVALUATION.task_id={args.task_id}",
                f"EVALUATION.num_trials={args.num_trials}"] + extra
    if extra:
        raise ValueError(f"Unrecognized arguments: {extra}")
    runner_mode = {"train-offline": "train", "train-online": "train-online", "test-online": "test"}[mode]
    if mode == "test-online" and not args.adapter:
        raise ValueError("test-online requires --adapter")
    cmd = [python, str(root / "scripts/experiment_vae_geometry_cached.py"), runner_mode,
           "--config", str(config_path)]
    for key in ("adapter", "output_dir", "steps", "batch_size", "num_workers", "train_samples", "inference_steps"):
        value = getattr(args, key, None)
        if value is not None:
            if key in {"adapter", "output_dir"}:
                value = absolute(value, root)
            cmd += ["--" + key.replace("_", "-"), str(value)]
    return cmd


def doctor(cfg):
    vae_relative = ("DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
                    if cfg.get("redirect_common_files", True)
                    else "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
    paths = [cfg.base_checkpoint, cfg.data.stats_path, cfg.text_weights,
             str(Path(cfg.model_base_path) / vae_relative),
             str(Path(cfg.tokenizer_path) / "tokenizer_config.json"),
             cfg.geometry.extractor.checkpoint_path,
             str(Path(cfg.geometry.extractor.da3_path) / "config.json"),
             str(Path(cfg.geometry.extractor.da3_path) / "model.safetensors"),
             str(Path(cfg.paths.track4world_repo) / "track4world/nets/model.py"),
             str(Path(cfg.paths.libero_repo) / "libero/libero/assets"),
             str(Path(cfg.paths.track4world_repo) / "track4world/nets/external/pi3/models/pi3.py")]
    missing = [p for p in paths if not Path(p).exists()]
    import torch
    devices = torch.cuda.device_count()
    invalid_devices = []
    for value in (cfg.model_device, cfg.geometry.extractor.device):
        dev = torch.device(value)
        if dev.type != "cuda" or (dev.index or 0) >= devices:
            invalid_devices.append(value)
    files = discover(cfg.paths.data_root, cfg.portable_suites)
    print(json.dumps({"python": sys.executable, "torch": torch.__version__, "visible_gpus": devices,
                      "task_files": len(files), "missing_paths": missing,
                      "invalid_devices": invalid_devices, "weights_loaded": False}, indent=2))
    return 1 if missing or invalid_devices else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("configure", "doctor", "plan", "extract", "status", "verify",
                                         "train-offline", "train-online", "test-online", "rollout-online"))
    parser.add_argument("--config", default="configs/local/libero.yaml")
    parser.add_argument("--template", default="configs/experiments/libero_portable.yaml")
    parser.add_argument("--assets-root")
    parser.add_argument("--data-root")
    parser.add_argument("--cache-dir")
    parser.add_argument("--model-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--suites", nargs="+", choices=tuple(SUITES))
    parser.add_argument("--set", dest="settings", nargs="*", default=[], help="configure only: dotted YAML overrides")
    parser.add_argument("--adapter")
    parser.add_argument("--output-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--suite", choices=tuple(SUITES), default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print only; no output files, models, downloads or subprocesses")
    args, extra = parser.parse_known_args(argv)
    config_path = Path(absolute(args.config))
    if args.mode == "configure":
        if extra:
            parser.error(f"Unrecognized arguments: {extra}")
        cfg = OmegaConf.merge(OmegaConf.load(absolute(args.template)), OmegaConf.from_dotlist(args.settings))
        for key in ("assets_root", "data_root"):
            if getattr(args, key) is not None:
                cfg.paths[key] = getattr(args, key)
        for key in ("cache_dir", "model_device", "steps", "batch_size", "num_workers", "train_samples", "inference_steps"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        if args.geometry_device:
            cfg.geometry.extractor.device = args.geometry_device
        if args.suites:
            cfg.portable_suites = args.suites
        cfg = resolve_config(cfg)
        cfg.data.files = discover(cfg.paths.data_root, cfg.portable_suites)
        if not args.dry_run:
            publish_config(config_path, cfg)
        print(OmegaConf.to_yaml(cfg))
        print(f"{'Would write' if args.dry_run else 'Configuration'}: {config_path}")
        return 0
    if any(getattr(args, key) is not None for key in ("assets_root", "data_root", "model_device", "geometry_device", "suites")) or args.settings:
        parser.error("Path/device/suite/--set overrides belong to configure; use a new config file")
    if args.cache_dir and args.mode not in {"plan", "extract", "status", "verify"}:
        parser.error("--cache-dir is supported by configure/plan/extract/status/verify; training reads its config")
    cfg = resolve_config(OmegaConf.load(config_path))
    if args.mode == "doctor":
        if args.dry_run or extra:
            parser.error("doctor is already read-only; no extra arguments")
        return doctor(cfg)
    command = build_command(args.mode, cfg, config_path, args, extra)
    env = child_environment(cfg)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if args.mode == "rollout-online":
        prepare_libero_config(cfg, env)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
