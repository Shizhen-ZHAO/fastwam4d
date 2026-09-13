"""保留参考轮转分片和 tmux 常驻 worker；直接启动脚本，避免 send-keys 竞争。"""
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_plus.task_utils import read_task_file, validate_gpu_ids, validate_worker_result


def tmux(*args, check=True):
    return subprocess.run(["tmux", *args], check=check, capture_output=True, text=True)


def stop_on_signal(signum, frame):
    raise KeyboardInterrupt(f"signal={signum}")


def prepare_workers(task_file, output_dir, gpu_ids, density):
    tasks = read_task_file(task_file)
    workers = []
    for directory in ("worker_tasks", "worker_results", "worker_status", "worker_launchers", "task_logs"):
        path = output_dir / directory
        path.mkdir(exist_ok=True)
        if any(path.iterdir()):
            raise FileExistsError(f"worker 目录非空，禁止混入历史结果：{path}")
    count = len(gpu_ids) * density
    for worker_id in range(count):
        shard = tasks[worker_id::count]
        shard_file = output_dir / "worker_tasks" / f"tasks_worker{worker_id}.txt"
        shard_file.write_text("".join(f"{suite},{task_id}\n" for suite, task_id in shard))
        if not shard:
            continue
        gpu = gpu_ids[worker_id % len(gpu_ids)]
        prefix = f"worker{worker_id}_gpu{gpu}"
        workers.append({"id": worker_id, "gpu": gpu, "tasks": shard, "task_file": shard_file,
                        "result": output_dir / "worker_results" / f"{prefix}_results.json",
                        "log": output_dir / "task_logs" / f"{prefix}.log",
                        "status": output_dir / "worker_status" / f"worker{worker_id}.status",
                        "started": output_dir / "worker_status" / f"worker{worker_id}.started",
                        "launcher": output_dir / "worker_launchers" / f"{prefix}.sh"})
    return workers


def write_launcher(worker, output_dir):
    # 只把本评测需要的环境写入脚本；不落盘其他凭据，不修改 tmux server 全局环境。
    names = {"PATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "CUDA_HOME", "PYTHONPATH",
             "DIFFSYNTH_MODEL_BASE_PATH", "DIFFSYNTH_SKIP_DOWNLOAD", "LIBERO_PLUS_ROOT",
             "LIBERO_PLUS_ASSETS_DIR", "LIBERO_ASSETS_DIR", "LIBERO_CONFIG_PATH",
             "MUJOCO_GL", "PYOPENGL_PLATFORM", "MUJOCO_EGL_DEVICE_ID", "OMP_NUM_THREADS",
             "TOKENIZERS_PARALLELISM", "HYDRA_FULL_ERROR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
             "FASTWAM_ACTION_ROPE_MODE", "FASTWAM_CORRUPT_TARGET", "FASTWAM_CORRUPT_SIGMA",
             "FASTWAM_PLUS_TRACE", "FASTWAM_PLUS_TRACE_REPO"}
    env = {key: os.environ[key] for key in names if key in os.environ}
    # 保留设备运行时与确定性配置，兼容集群定制 torch/PPU 环境。
    hardware_prefixes = ("CUDA_", "NVIDIA_", "TORCH_", "PYTORCH_", "PPU_", "PPL_", "DIPU_", "DIOPI_", "NCCL_",
                         "CUBLAS_", "CUDNN_", "MUSA_", "HIP_", "OMP_", "MKL_", "OPENBLAS_")
    env.update({key: value for key, value in os.environ.items() if key.startswith(hardware_prefixes)})
    env.update({"CUDA_VISIBLE_DEVICES": worker["gpu"], "LIBERO_PLUS_WORKER_ID": str(worker["id"]),
                "LIBERO_PLUS_WORKER_TASK_FILE": str(worker["task_file"]),
                "LIBERO_PLUS_WORKER_RESULT_FILE": str(worker["result"])})
    # env -i 清除旧 tmux server 的残留实验开关；保留该用户的系统身份信息。
    for name in ("HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL"):
        if name in os.environ:
            env[name] = os.environ[name]
    worker_entry = ROOT / "experiments/libero_plus/eval_libero_plus_worker.py"
    if os.environ.get("FASTWAM_PLUS_TRACE", "0") == "1":
        worker_entry = ROOT / "experiments/libero_plus/trace_worker.py"
    command = ["env", "-i", *(f"{key}={value}" for key, value in sorted(env.items())),
               os.environ.get("PYTHON_BIN", sys.executable), str(worker_entry),
               *shlex.split(os.environ["EXTRA_ARGS"]), f"gpu_id={worker['gpu']}"]
    script = ("#!/usr/bin/env bash\nset -u\n"
              f"cd {shlex.quote(str(ROOT))} || exit 1\n"
              f"touch {shlex.quote(str(worker['started']))}\n"
              f"{shlex.join(command)} > {shlex.quote(str(worker['log']))} 2>&1\n"
              "rc=$?\n"
              f"printf '%s\\n' \"$rc\" > {shlex.quote(str(worker['status']) + '.tmp')}\n"
              f"mv {shlex.quote(str(worker['status']) + '.tmp')} {shlex.quote(str(worker['status']))}\n"
              'exit "$rc"\n')
    worker["launcher"].write_text(script)
    worker["launcher"].chmod(0o700)


def run(task_file):
    if shutil.which("tmux") is None:
        raise RuntimeError("未找到 tmux，请在评测环境安装 tmux")
    output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
    ids = validate_gpu_ids(os.environ.get("CUDA_VISIBLE_DEVICES", ""), int(os.environ["NUM_GPUS"]))
    density = int(os.environ["MAX_TASKS_PER_GPU"])
    trials = int(os.environ["NUM_TRIALS"])
    wave_size = int(os.environ.get("WORKER_START_WAVE_SIZE", len(ids)))
    interval = float(os.environ.get("WORKER_START_WAVE_INTERVAL", "120"))
    startup_timeout = float(os.environ.get("WORKER_STARTUP_TIMEOUT", "180"))
    poll = float(os.environ.get("MONITORING_INTERVAL", "5"))
    if min(density, trials, wave_size, startup_timeout, poll) <= 0 or interval < 0:
        raise ValueError("worker/trial/wave/timeout/poll 配置非法")
    session = os.environ.get("SESSION_NAME") or f"libero_plus_{uuid.uuid4().hex[:12]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("SESSION_NAME 只能使用字母、数字、下划线和连字符")
    if tmux("has-session", "-t", "=" + session, check=False).returncode == 0:
        raise FileExistsError(f"tmux session 已存在：{session}；不会关闭现有作业")
    workers = prepare_workers(task_file, output_dir, ids, density)
    for worker in workers:
        write_launcher(worker, output_dir)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop_on_signal)
    owned = False
    try:
        tmux("new-session", "-d", "-s", session, "-n", "supervisor", "sleep 2147483647")
        owned = True
        (output_dir / "worker_session.json").write_text(json.dumps({"session": session, "workers": len(workers)}))
        launched, next_launch, next_status = [], 0.0, 0.0
        while len([w for w in workers if w.get("done")]) < len(workers):
            now = time.monotonic()
            # 先检查已启动进程，再启动下一批，避免已知失败后继续冷加载模型。
            pane_result = tmux("list-panes", "-s", "-t", session, "-F", "#{pane_id}", check=False)
            panes = set(pane_result.stdout.splitlines())
            for worker in launched:
                if worker.get("done"):
                    continue
                if worker["status"].is_file():
                    rc = int(worker["status"].read_text().strip())
                    if rc != 0:
                        raise RuntimeError(f"worker {worker['id']} 退出码 {rc}；日志：{worker['log']}")
                    validate_worker_result(worker["result"], worker["tasks"], trials)
                    worker["done"] = True
                elif worker["pane"] not in panes:
                    # 窗口可能在读取 list-panes 后刚刚正常退出，再检查一次原子状态文件。
                    if not worker["status"].exists():
                        raise RuntimeError(f"worker {worker['id']} 窗口消失且没有完成状态；日志：{worker['log']}")
                elif not worker["started"].exists() and now - worker["launched_at"] > startup_timeout:
                    raise TimeoutError(f"worker {worker['id']} 启动超时")
            if len(launched) < len(workers) and now >= next_launch:
                for worker in workers[len(launched):len(launched) + wave_size]:
                    command = shlex.join(["bash", str(worker["launcher"])])
                    pane = tmux("new-window", "-d", "-P", "-F", "#{pane_id}", "-t", session,
                                "-n", f"worker{worker['id']}", command).stdout.strip()
                    worker.update(pane=pane, launched_at=time.monotonic())
                    launched.append(worker)
                next_launch = time.monotonic() + interval
            if now >= next_status:
                print(f"[{session}] launched={len(launched)}/{len(workers)} "
                      f"completed={sum(bool(w.get('done')) for w in workers)}", flush=True)
                next_status = now + 30
            time.sleep(poll)
        subprocess.run([os.environ.get("PYTHON_BIN", sys.executable),
                        str(ROOT / "experiments/libero_plus/summarize_results.py"),
                        "--output_dir", str(output_dir), "--task_file", str(task_file),
                        "--num_trials", str(trials)], check=True)
    finally:
        if owned:
            tmux("kill-session", "-t", "=" + session, check=False)


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve())
