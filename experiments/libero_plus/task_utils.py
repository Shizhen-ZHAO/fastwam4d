"""任务清单及结果完整性检查；不依赖模型或仿真包。"""
import hashlib
import json
from pathlib import Path

SUITES = ("libero_10", "libero_goal", "libero_spatial", "libero_object")
RESULT_FORMAT = "libero_plus_worker_results_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_task_file(path):
    tasks, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(",")
        if len(parts) != 2 or parts[0] not in SUITES or not parts[1].isdigit():
            raise ValueError(f"非法任务：{path}:{number}: {line!r}")
        task = (parts[0], int(parts[1]))
        if task in seen:
            raise ValueError(f"重复任务：{path}:{number}: {task}")
        seen.add(task)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"任务清单为空：{path}")
    return tasks


def validate_gpu_ids(raw, num_gpus):
    if num_gpus < 1:
        raise ValueError("NUM_GPUS 必须大于 0")
    ids = [s.strip() for s in raw.split(",")] if raw else [str(i) for i in range(num_gpus)]
    if len(ids) != num_gpus or len(set(ids)) != len(ids) or any(not s.isdigit() for s in ids):
        raise ValueError("CUDA_VISIBLE_DEVICES 必须包含 NUM_GPUS 个不重复的非负物理卡编号")
    return ids


def validate_worker_result(path, expected_tasks, num_trials):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format_version") != RESULT_FORMAT or payload.get("failed_tasks"):
        raise ValueError(f"worker 格式错误或包含异常：{path}")
    results = payload.get("results", [])
    actual = [(r["task_suite"], int(r["task_id"])) for r in results]
    if actual != list(expected_tasks):
        raise ValueError(f"worker 任务顺序/覆盖不匹配：{path}")
    if payload.get("total_tasks") != len(actual) or payload.get("completed_tasks") != len(actual):
        raise ValueError(f"worker 计数不完整：{path}")
    for result in results:
        success = result.get("success_episodes", [])
        failure = result.get("failure_episodes", [])
        if (result.get("total_episodes") != num_trials
                or result.get("successes") != len(success)
                or sorted(success + failure) != list(range(num_trials))):
            raise ValueError(f"episode 计数/下标错误：{path}: {result.get('task_id')}")
    return results


def _validate_run(output_dir, task_file=None, num_trials=None):
    output_dir = Path(output_dir)
    meta_path = output_dir / "eval_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    task_file = Path(task_file or output_dir / "tasks.txt")
    if num_trials is None:
        num_trials = int(meta["num_trials"])
    if num_trials < 1:
        raise ValueError("num_trials 必须大于 0")
    expected = read_task_file(task_file)
    if meta.get("task_file_sha256") and file_sha256(task_file) != meta["task_file_sha256"]:
        raise ValueError("任务清单与运行时记录的 hash 不一致")
    records, seen, worker_ids = [], set(), set()
    for path in sorted((output_dir / "worker_results").glob("worker*_results.json")):
        payload = json.loads(path.read_text())
        worker_id = str(payload["worker_id"])
        if worker_id in worker_ids or not worker_id.isdigit():
            raise ValueError(f"重复或非法 worker ID：{path}")
        worker_ids.add(worker_id)
        shard = output_dir / "worker_tasks" / f"tasks_worker{worker_id}.txt"
        tasks = read_task_file(shard)
        rows = validate_worker_result(path, tasks, num_trials)
        for row in rows:
            key = (row["task_suite"], int(row["task_id"]))
            if key in seen:
                raise ValueError(f"跨 worker 重复任务：{key}")
            seen.add(key)
        records.extend(rows)
    if seen != set(expected):
        raise ValueError(f"结果覆盖不完整：expected={len(expected)}, actual={len(seen)}, "
                         f"missing={len(set(expected)-seen)}, extra={len(seen-set(expected))}")
    report = {"passed": True, "tasks": len(expected), "episodes": len(expected) * num_trials,
              "workers": len(worker_ids), "task_file_sha256": file_sha256(task_file)}
    (output_dir / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def validate_run(output_dir, task_file=None, num_trials=None):
    try:
        return _validate_run(output_dir, task_file, num_trials)
    except Exception as exc:
        output_dir = Path(output_dir)
        if output_dir.is_dir():
            (output_dir / "coverage.json").write_text(json.dumps({"passed": False, "error": str(exc)}, indent=2) + "\n")
        raise
