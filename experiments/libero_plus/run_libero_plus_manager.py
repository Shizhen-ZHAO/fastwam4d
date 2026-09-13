"""Plus manager：使用最终 Hydra 配置准备环境、清单与常驻 worker。"""
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from experiments.libero_plus.eval_utils import (
    expanded_path, resolve_stats_path, validate_checkpoint_config, validate_eval_config, validate_model_assets,
)
from experiments.libero_plus.task_utils import file_sha256, read_task_file, validate_gpu_ids

for name, resolver in (("eval", eval), ("max", max), ("split", lambda s, i: s.split("/")[int(i)])):
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)


def git_revision(path):
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def prepare_benchmark(output_dir):
    root = expanded_path(os.environ["LIBERO_PLUS_ROOT"])
    paths = {
        "assets": expanded_path(os.environ["LIBERO_PLUS_ASSETS_DIR"]),
        "bddl_files": expanded_path(os.environ.get("LIBERO_PLUS_BDDL_DIR", root / "libero/libero/bddl_files")),
        "init_states": expanded_path(os.environ.get("LIBERO_PLUS_INIT_STATES_DIR", root / "libero/libero/init_files")),
        "benchmark_root": expanded_path(os.environ.get("LIBERO_PLUS_BENCHMARK_ROOT", root / "libero/libero")),
    }
    for path in [root, *paths.values()]:
        if not path.is_dir():
            raise FileNotFoundError(f"Plus 环境目录不存在：{path}")
    config_dir = output_dir / "libero_config"
    config_dir.mkdir()
    config = {key: str(value) for key, value in paths.items()}
    config["datasets"] = str(paths["benchmark_root"].parent / "datasets")
    OmegaConf.save(OmegaConf.create(config), config_dir / "config.yaml")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["LIBERO_ASSETS_DIR"] = str(paths["assets"])
    sys.path.insert(0, str(root))
    import libero.libero as libero
    from libero.libero import benchmark
    module_path = Path(libero.__file__).resolve()
    if root not in module_path.parents:
        raise ValueError(f"导入的 libero 不是指定 Plus 环境：{module_path}")
    for key, expected in paths.items():
        if expanded_path(libero.get_libero_path(key)) != expected:
            raise ValueError(f"LIBERO 配置解析不匹配：{key}")
    return benchmark.get_benchmark_dict(), {
        "libero_plus_root": str(root), "libero_module": str(module_path),
        "libero_revision": git_revision(root), "libero_paths": config,
    }


def prepare_tasks(cfg, output_dir, benchmarks):
    supplied = cfg.MULTIRUN.get("task_file")
    task_file = output_dir / "tasks.txt"
    suites = {}
    if supplied:
        source = expanded_path(supplied)
        tasks = read_task_file(source)
        # 不调用目标普通 manager 中会覆盖文件的 create_task_file。
        shutil.copyfile(source, task_file)
    else:
        tasks = []
        for suite in cfg.MULTIRUN.task_suite_names:
            suites[suite] = benchmarks[suite]()
            tasks.extend((suite, task_id) for task_id in range(int(suites[suite].n_tasks)))
        task_file.write_text("".join(f"{suite},{task_id}\n" for suite, task_id in tasks))
        tasks = read_task_file(task_file)
    manifest = []
    for suite, task_id in tasks:
        if suite not in suites:
            suites[suite] = benchmarks[suite]()
        if task_id >= int(suites[suite].n_tasks):
            raise ValueError(f"任务 ID 超出实际 Plus benchmark 范围：{suite},{task_id}")
        task = suites[suite].get_task(task_id)
        manifest.append({"suite": suite, "task_id": task_id, "name": str(task.name),
                         "bddl_file": str(task.bddl_file)})
    (output_dir / "task_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return task_file


def backup_results(output_dir):
    for flag, destination in (("OSS_BACKUP", "OSS_BACKUP_DIR"), ("SHARE_BACKUP", "SHARE_BACKUP_DIR")):
        if os.environ.get(flag, "false").lower() != "true":
            continue
        try:
            base = expanded_path(os.environ[destination])
            if base == output_dir or output_dir in base.parents:
                raise ValueError("备份目的地不能位于本次输出目录内")
            base.mkdir(parents=True, exist_ok=True)
            shutil.copytree(output_dir, base / output_dir.name)
            print(f"结果备份：{base / output_dir.name}", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"备份未完成，原结果保留在 {output_dir}：{exc}", file=sys.stderr)


def write_worker_config(resolved, output_dir):
    worker_config = dict(resolved)
    # cfg 快照不含 Hydra 内部节点，显式保留不切目录/不写共用日志的设置。
    worker_config["defaults"] = [{"override hydra/job_logging": "disabled"}, "_self_"]
    worker_config["hydra"] = {"job": {"chdir": False}, "run": {"dir": "."}, "output_subdir": None}
    path = output_dir / "worker_config.yaml"
    OmegaConf.save(OmegaConf.create(worker_config), path)
    return path


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus")
def main(cfg: DictConfig):
    variant = validate_eval_config(cfg)
    ids = validate_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES", ""), int(cfg.MULTIRUN.num_gpus))
    create_only = bool(cfg.MULTIRUN.create_only)
    if not create_only and cfg.ckpt is None:
        raise ValueError("请设置 CKPT 为对应 1d 模型的训练权重")
    if not create_only and shutil.which("tmux") is None:
        raise RuntimeError("未找到 tmux，请在评测环境安装 tmux")
    checkpoint_meta = None
    assets = None
    if not create_only:
        cfg.ckpt = str(expanded_path(cfg.ckpt))
        checkpoint_meta = validate_checkpoint_config(cfg)
        cfg.EVALUATION.dataset_stats_path = str(resolve_stats_path(cfg))
        assets = validate_model_assets(cfg)
    output_dir = expanded_path(cfg.EVALUATION.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"输出目录非空，请使用新 OUTPUT_DIR：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.EVALUATION.output_dir = str(output_dir)
    benchmarks, benchmark_meta = prepare_benchmark(output_dir)
    task_file = prepare_tasks(cfg, output_dir, benchmarks)
    cfg.MULTIRUN.task_file = str(task_file)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(OmegaConf.create(resolved), output_dir / "manager_config.yaml")
    meta = {
        "benchmark": "libero_plus", "eval_mode": variant, "ckpt": cfg.ckpt,
        "num_trials": int(cfg.EVALUATION.num_trials), "task_file_sha256": file_sha256(task_file),
        "repo_revision": git_revision(ROOT), "checkpoint_config": checkpoint_meta,
        "cuda_visible_devices": ids, "create_only": create_only, **benchmark_meta,
        "model_assets": assets,
    }
    if not create_only:
        meta["ckpt_sha256"] = file_sha256(cfg.ckpt)
        meta["dataset_stats_path"] = cfg.EVALUATION.dataset_stats_path
        meta["dataset_stats_sha256"] = file_sha256(cfg.EVALUATION.dataset_stats_path)
    (output_dir / "eval_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if create_only:
        print(f"仅生成任务清单与配置：{output_dir}")
        return
    # worker 直接加载已解析快照，避免环境/YAML/Hydra 三层覆盖再次产生漂移。
    write_worker_config(resolved, output_dir)
    env = os.environ.copy()
    env.update({"ROOT_DIR": str(ROOT), "OUTPUT_DIR": str(output_dir), "CKPT": str(cfg.ckpt),
                "CONFIG": HydraConfig.get().runtime.choices.task,
                "NUM_GPUS": str(len(ids)), "CUDA_VISIBLE_DEVICES": ",".join(ids),
                "NUM_TRIALS": str(cfg.EVALUATION.num_trials),
                "MAX_TASKS_PER_GPU": str(cfg.MULTIRUN.max_tasks_per_gpu),
                "PYTHON_BIN": sys.executable,
                "EXTRA_ARGS": shlex.join(["--config-path", str(output_dir), "--config-name", "worker_config"])})
    if "OMP_NUM_THREADS" not in env and int(cfg.MULTIRUN.max_tasks_per_gpu) > 1:
        env["OMP_NUM_THREADS"] = str(max(1, (os.cpu_count() or 1) * 3 // 4 //
                                             (len(ids) * int(cfg.MULTIRUN.max_tasks_per_gpu))))
    print(f"开始 {variant} 评测：{len(ids)} 卡 × {cfg.MULTIRUN.max_tasks_per_gpu} worker；输出 {output_dir}", flush=True)
    try:
        subprocess.run(["bash", str(ROOT / "experiments/libero_plus/run_libero_plus_parallel_workers.sh"),
                        str(task_file)], cwd=ROOT, env=env, check=True)
    finally:
        backup_results(output_dir)


if __name__ == "__main__":
    main()
