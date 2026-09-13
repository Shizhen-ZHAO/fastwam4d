import json
import logging
import os
import time
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments/libero"))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from experiments.libero.eval_libero_single import (
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from experiments.libero_plus.eval_utils import (
    get_task_init_states, repeat_initial_states, resolve_stats_path,
    validate_checkpoint_config, validate_eval_config,
)
from experiments.libero_plus.task_utils import read_task_file
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


def _write_worker_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, cls=NumpyEncoder)
    tmp_path.replace(path)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus.yaml")
def main(cfg: DictConfig):
    worker_start_time = time.time()
    task_file = os.environ.get("LIBERO_PLUS_WORKER_TASK_FILE")
    if not task_file:
        raise ValueError("LIBERO_PLUS_WORKER_TASK_FILE must be set.")
    task_file = Path(task_file)
    if not task_file.is_file():
        raise FileNotFoundError(f"Worker task file not found: {task_file}")

    worker_id = os.environ.get("LIBERO_PLUS_WORKER_ID", str(cfg.gpu_id))
    gpu_id = int(cfg.gpu_id)
    output_dir = Path(cfg.EVALUATION.output_dir)
    result_file = Path(
        os.environ.get(
            "LIBERO_PLUS_WORKER_RESULT_FILE",
            str(output_dir / "worker_results" / f"worker{worker_id}_gpu{gpu_id}_results.json"),
        )
    )

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed) + int(worker_id), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("LIBERO-plus persistent worker currently supports only env_num=1.")

    task_specs = read_task_file(task_file)
    payload = {
        "format_version": "libero_plus_worker_results_v1",
        "worker_id": worker_id,
        "gpu_id": gpu_id,
        "task_file": str(task_file),
        "result_file": str(result_file),
        "total_tasks": len(task_specs),
        "completed_tasks": 0,
        "failed_tasks": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
        "results": [],
    }
    _write_worker_results(result_file, payload)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    validate_eval_config(cfg)
    payload["checkpoint_config"] = validate_checkpoint_config(cfg)
    dataset_stats_path = resolve_stats_path(cfg)
    import torch
    payload["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                          "device": model_device, "dtype": str(model_dtype),
                          "worker_seed": None if cfg.seed is None else int(cfg.seed) + int(worker_id),
                          "episode_seed": cfg.seed}
    _write_worker_results(result_file, payload)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    benchmark_dict = benchmark.get_benchmark_dict()
    num_trials = int(cfg.EVALUATION.num_trials)

    for suite, task_id in task_specs:
        task_start_time = time.time()
        try:
            cfg.EVALUATION.task_suite_name = suite
            cfg.EVALUATION.task_id = task_id

            task_suite = benchmark_dict[suite]()
            task = task_suite.get_task(task_id)
            initial_states = repeat_initial_states(
                get_task_init_states(task_suite, task_id),
                num_trials,
            )

            video_dir = output_dir / suite / "videos"
            predicted_video_dir = output_dir / suite / "predicted_videos"
            if bool(cfg.EVALUATION.get("save_video", True)):
                video_dir.mkdir(parents=True, exist_ok=True)
            if bool(cfg.EVALUATION.get("visualize_future_video", False)):
                predicted_video_dir.mkdir(parents=True, exist_ok=True)

            result = {
                "task_suite": suite,
                "task_id": task_id,
                "task_name": task.name,
                "bddl_file": str(task.bddl_file),
                "task_description": None,
                "successes": 0,
                "total_episodes": num_trials,
                "gpu_id": gpu_id,
                "worker_id": worker_id,
                "success_episodes": [],
                "failure_episodes": [],
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": 0.0,
            }
            result.update(
                run_single_task(
                    task=task,
                    initial_states=initial_states,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    video_dir=video_dir,
                    predicted_video_dir=predicted_video_dir,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=model_device,
                )
            )
            result["duration"] = time.time() - task_start_time
            payload["results"].append(result)
            payload["completed_tasks"] = len(payload["results"])
            payload["duration"] = time.time() - worker_start_time
            _write_worker_results(result_file, payload)
            print(
                f"[worker {worker_id}] {suite} task_id={task_id} completed: "
                f"{result['successes']}/{num_trials} successes"
            )
        except Exception as exc:
            payload["failed_tasks"].append(
                {
                    "task_suite": suite,
                    "task_id": task_id,
                    "error": repr(exc),
                    "duration": time.time() - task_start_time,
                }
            )
            payload["duration"] = time.time() - worker_start_time
            _write_worker_results(result_file, payload)
            raise

    payload["duration"] = time.time() - worker_start_time
    _write_worker_results(result_file, payload)
    print(
        f"[worker {worker_id}] finished {payload['completed_tasks']}/"
        f"{payload['total_tasks']} tasks in {payload['duration']:.2f}s"
    )


if __name__ == "__main__":
    main()
