import argparse
import json
import os
import sys
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.libero_plus.task_utils import validate_run


KNOWN_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
DEFAULT_PARENT_DIR = Path("evaluate_results/libero_plus")
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})$")


def _new_stats():
    return {
        "total_tasks": 0,
        "total_trials": 0,
        "total_successes": 0,
        "total_time": 0.0,
        "max_time": 0.0,
        "psnr_sum": 0.0,
        "psnr_count": 0,
    }


def format_time(seconds):
    """Format seconds as a human-readable duration string."""
    seconds = round(seconds)

    if seconds < 60:
        return f"{seconds:02d}s"
    if seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}m{remaining_seconds:02d}s"

    hours = seconds // 3600
    remaining = seconds % 3600
    minutes = remaining // 60
    remaining_seconds = remaining % 60
    return f"{hours:02d}h{minutes:02d}m{remaining_seconds:02d}s"


def _load_eval_meta(output_dir: Path) -> dict:
    meta_path = output_dir / "eval_meta.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def pick_latest_run(parent: Path = DEFAULT_PARENT_DIR) -> Path:
    if not parent.is_dir():
        raise FileNotFoundError(f"Default evaluation parent directory does not exist: {parent}")

    candidates: list[tuple[str, Path]] = []
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        match = TIMESTAMP_RE.search(child.name)
        if match:
            candidates.append((match.group(1), child))

    if not candidates:
        raise FileNotFoundError(f"No timestamped evaluation run found under: {parent}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_task_classification_path(output_dir: Path, explicit_path: str | None) -> Path | None:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(explicit_path))))

    for env_name in ("TASK_CLASSIFICATION_PATH", "LIBERO_TASK_CLASSIFICATION_PATH"):
        env_path = os.environ.get(env_name)
        if env_path:
            candidates.append(Path(os.path.expanduser(os.path.expandvars(env_path))))

    meta = _load_eval_meta(output_dir)
    meta_root = meta.get("libero_plus_root")
    if meta_root:
        candidates.append(
            Path(os.path.expanduser(os.path.expandvars(str(meta_root))))
            / "libero"
            / "libero"
            / "benchmark"
            / "task_classification.json"
        )

    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT")
    if libero_plus_root:
        candidates.append(
            Path(os.path.expanduser(os.path.expandvars(libero_plus_root)))
            / "libero"
            / "libero"
            / "benchmark"
            / "task_classification.json"
        )

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    if explicit_path:
        raise FileNotFoundError(f"Task classification file not found: {explicit_path}")
    return None


def _load_task_classification(path: Path | None):
    if path is None:
        return None, None

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"task_classification.json must be a dict keyed by suite, got {type(raw)}")

    index = {}
    for suite, records in raw.items():
        if not isinstance(records, list):
            continue

        by_name = {}
        by_one_based_id = {}
        by_zero_based_id = {}
        for record in records:
            if not isinstance(record, dict):
                continue

            rec = {
                "id": record.get("id"),
                "name": record.get("name"),
                "category": record.get("category"),
                "difficulty_level": record.get("difficulty_level"),
            }
            if rec["name"] is not None:
                by_name[str(rec["name"])] = rec

            try:
                one_based_id = int(rec["id"])
            except (TypeError, ValueError):
                continue
            by_one_based_id[one_based_id] = rec
            by_zero_based_id[one_based_id - 1] = rec

        index[str(suite)] = {
            "by_name": by_name,
            "by_one_based_id": by_one_based_id,
            "by_zero_based_id": by_zero_based_id,
        }

    return path, index


def _lookup_task_classification(classification, suite: str, task_id: int, task_name: str | None):
    if classification is None:
        return None

    suite_index = classification.get(suite)
    if suite_index is None:
        return None

    if task_name:
        record = suite_index["by_name"].get(str(task_name))
        if record is not None:
            return record

    # LIBERO-plus task_classification.json 的 id 从 1 开始；eval task_id 从 0 开始。
    record = suite_index["by_one_based_id"].get(int(task_id) + 1)
    if record is not None:
        return record

    # 兼容后续如果上游改成 0-based id 的情况。
    return suite_index["by_zero_based_id"].get(int(task_id))


def _iter_suite_dirs(output_dir: Path):
    yielded = set()
    for suite in KNOWN_SUITES:
        suite_dir = output_dir / suite
        if suite_dir.is_dir():
            yielded.add(suite)
            yield suite, suite_dir

    for child in sorted(output_dir.iterdir()):
        if child.is_dir() and child.name not in yielded:
            yield child.name, child


def _merge_task_shard_records(records: list[dict]) -> dict:
    """合并同一 (suite, task_id) 的多条 trial 分片记录为一条 task 级记录。

    trial 分片(MULTIRUN.trials_per_shard>0)后,同一 task 会被多个 worker 各跑
    一个 [trial_start, trial_end) 区间,worker_results 里每片一条记录。这里按
    全局下标拼回完整 task:successes/total_episodes 求和,success/failure_episodes
    (已是全局下标)concat,duration 取各片墙钟之和(≈ 该 task 的总 GPU 占用时间,
    片间并行,不是墙钟时长)。
    """
    if len(records) == 1:
        return records[0]

    merged = dict(records[0])
    merged["successes"] = sum(int(r.get("successes", 0)) for r in records)
    merged["total_episodes"] = sum(int(r.get("total_episodes", 0)) for r in records)
    merged["success_episodes"] = sorted(
        ep for r in records for ep in r.get("success_episodes", [])
    )
    merged["failure_episodes"] = sorted(
        ep for r in records for ep in r.get("failure_episodes", [])
    )
    merged["duration"] = sum(float(r.get("duration", 0.0)) for r in records)

    # 逐 episode PSNR 列表直接拼接重算均值(与未分片口径一致)。
    if any("episode_future_video_psnr" in r for r in records):
        episode_psnr = [x for r in records for x in r.get("episode_future_video_psnr", [])]
        merged["episode_future_video_psnr"] = episode_psnr
        valid = [float(x) for x in episode_psnr if x is not None]
        merged["future_video_psnr_mean"] = float(sum(valid) / len(valid)) if valid else None

    # 软校验: 同 task 各片区间不应重叠(缺区间字段的旧记录跳过)。
    intervals = sorted(
        (int(r["trial_start"]), int(r["trial_end"]))
        for r in records
        if r.get("trial_start") is not None and r.get("trial_end") is not None
    )
    for (prev_start, prev_end), (cur_start, cur_end) in zip(intervals, intervals[1:]):
        if cur_start < prev_end:
            print(
                f"[WARN] 分片区间重叠: {records[0].get('task_suite')} task "
                f"{records[0].get('task_id')} [{prev_start},{prev_end}) 与 "
                f"[{cur_start},{cur_end}),请检查是否有重复统计"
            )
            break
    return merged


def _iter_task_result_records(output_dir: Path):
    yielded = set()

    for suite, suite_dir in _iter_suite_dirs(output_dir):
        for filename in os.listdir(suite_dir):
            if not filename.startswith("gpu") or not filename.endswith("_results.json"):
                continue

            result_path = suite_dir / filename
            with result_path.open("r", encoding="utf-8") as f:
                result = json.load(f)

            parts = filename.split("_")
            task_id_raw = result.get("task_id")
            if task_id_raw is None:
                task_id_raw = parts[1].replace("task", "")
            task_id = int(task_id_raw)
            result_suite = str(result.get("task_suite") or suite)
            yielded.add((result_suite, task_id))
            yield result_suite, task_id, result

    worker_result_dir = output_dir / "worker_results"
    if not worker_result_dir.is_dir():
        return

    # trial 分片后同一 (suite, task_id) 有多条记录,先按 task 分组合并再产出;
    # 首次出现的顺序与旧去重逻辑一致(按文件名排序后的记录顺序)。
    grouped: dict[tuple[str, int], list[dict]] = {}
    for result_path in sorted(worker_result_dir.glob("worker*_results.json")):
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for result in payload.get("results", []):
            suite = str(result["task_suite"])
            task_id = int(result["task_id"])
            if (suite, task_id) in yielded:
                continue
            grouped.setdefault((suite, task_id), []).append(result)

    for (suite, task_id), records in grouped.items():
        yielded.add((suite, task_id))
        yield suite, task_id, _merge_task_shard_records(records)


def _update_stats(stats: dict, result: dict):
    stats["total_tasks"] += 1
    stats["total_trials"] += int(result["total_episodes"])
    stats["total_successes"] += int(result["successes"])
    stats["total_time"] += float(result["duration"])
    stats["max_time"] = max(stats["max_time"], float(result["duration"]))

    if "future_video_psnr_mean" in result and result["future_video_psnr_mean"] is not None:
        stats["psnr_sum"] += float(result["future_video_psnr_mean"])
        stats["psnr_count"] += 1


def _success_rate_percent(stats: dict) -> float:
    total_trials = stats["total_trials"]
    return stats["total_successes"] / total_trials * 100 if total_trials else 0.0


def _stats_to_output(stats: dict, has_psnr_metric: bool) -> dict:
    output = {
        "total_tasks": stats["total_tasks"],
        "total_trials": stats["total_trials"],
        "total_successes": stats["total_successes"],
        "total_time": stats["total_time"],
        "max_time": stats["max_time"],
        "success_rate": stats["total_successes"] / stats["total_trials"] if stats["total_trials"] else 0.0,
    }
    if has_psnr_metric:
        output["average_future_video_psnr"] = (
            stats["psnr_sum"] / stats["psnr_count"] if stats["psnr_count"] > 0 else None
        )
    return output


def _write_group_success_csv(output_dir: Path, filename: str, group_name: str, group_stats: dict, has_psnr_metric: bool):
    rows = []
    for group_value in sorted(group_stats.keys(), key=lambda x: str(x)):
        stats = group_stats[group_value]
        row = {
            group_name: group_value,
            "Tasks": stats["total_tasks"],
            "Total Episodes": stats["total_trials"],
            "Successes": stats["total_successes"],
            "Success Rate (%)": f"{_success_rate_percent(stats):.2f}",
        }
        if has_psnr_metric:
            avg_psnr = stats["psnr_sum"] / stats["psnr_count"] if stats["psnr_count"] > 0 else None
            row["Average Future PSNR (dB)"] = f"{avg_psnr:.4f}" if avg_psnr is not None else "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path, df


def _suite_from_task_key(task_key: str) -> str:
    return task_key.rsplit("_", 1)[0]


def summarize_results(output_dir, task_classification_path=None, *, task_file=None, num_trials=None):
    """Summarize all evaluation results."""
    output_dir = Path(output_dir)
    validate_run(output_dir, task_file=task_file, num_trials=num_trials)
    suite_stats = defaultdict(_new_stats)
    category_stats = defaultdict(_new_stats)
    difficulty_stats = defaultdict(_new_stats)

    task_results = {}
    has_psnr_metric = False
    classification_path, classification = _load_task_classification(
        _resolve_task_classification_path(output_dir, task_classification_path)
    )
    classification_matched = 0
    classification_unmatched = 0
    if classification_path is not None:
        print(f"Using task classification: {classification_path}")

    for suite, task_id, result in _iter_task_result_records(output_dir):
        task_key = f"{suite}_{task_id}"

        _update_stats(suite_stats[suite], result)
        if "future_video_psnr_mean" in result:
            has_psnr_metric = True

        task_name = result.get("task_name")
        classification_record = _lookup_task_classification(
            classification,
            suite=suite,
            task_id=task_id,
            task_name=task_name,
        )
        if classification_record is not None:
            classification_matched += 1
            category = classification_record.get("category")
            difficulty_level = classification_record.get("difficulty_level")
            if category is not None:
                _update_stats(category_stats[str(category)], result)
            if difficulty_level is not None:
                _update_stats(difficulty_stats[str(difficulty_level)], result)
        elif classification is not None:
            classification_unmatched += 1

        task_result = {
            "success_rate": result["successes"] / result["total_episodes"] * 100,
            "duration": result["duration"],
            "total_episodes": result["total_episodes"],
            "successes": result["successes"],
            "task_name": task_name or "",
            "task_description": result["task_description"] if "task_description" in result else "",
        }
        if classification_record is not None:
            task_result["classification_id"] = classification_record.get("id")
            task_result["classification_name"] = classification_record.get("name")
            task_result["category"] = classification_record.get("category")
            task_result["difficulty_level"] = classification_record.get("difficulty_level")
        if "future_video_psnr_mean" in result:
            task_result["future_video_psnr_mean"] = (
                float(result["future_video_psnr_mean"])
                if result["future_video_psnr_mean"] is not None
                else None
            )
        task_results[task_key] = task_result

    print("\n=== Evaluation Results Summary ===")
    print("\nStatistics for each task suite:")

    total_success_rate = 0.0
    total_time = 0.0
    total_suites = 0
    overall_psnr_sum = 0.0
    overall_psnr_count = 0

    df_data = {
        "Task Suite": [],
        "Success Rate (%)": [],
        "Average Time (s)": [],
        "Max Time (s)": [],
    }
    if has_psnr_metric:
        df_data["Average Future PSNR (dB)"] = []

    for suite, stats in suite_stats.items():
        if stats["total_trials"] <= 0:
            continue

        success_rate = _success_rate_percent(stats)
        avg_time = stats["total_time"] / stats["total_tasks"]
        max_time = stats["max_time"]
        suite_avg_psnr = None
        if has_psnr_metric:
            suite_avg_psnr = (
                stats["psnr_sum"] / stats["psnr_count"] if stats["psnr_count"] > 0 else None
            )

        print(f"\n{suite}:")
        print(f"- Tasks completed: {stats['total_tasks']}")
        print(f"- Total attempts: {stats['total_trials']}")
        print(f"- Successful attempts: {stats['total_successes']}")
        print(f"- Success rate: {success_rate:.2f}%")
        print(f"- Total time: {format_time(stats['total_time'])}")
        print(f"- Average time per task: {format_time(avg_time)}")
        print(f"- Longest task time: {format_time(max_time)}")
        if has_psnr_metric:
            if suite_avg_psnr is not None:
                print(f"- Average future-video PSNR: {suite_avg_psnr:.4f} dB")
            else:
                print("- Average future-video PSNR: N/A")

        df_data["Task Suite"].append(suite)
        df_data["Success Rate (%)"].append(f"{success_rate:.2f}")
        df_data["Average Time (s)"].append(f"{avg_time:.2f}")
        df_data["Max Time (s)"].append(f"{max_time:.2f}")
        if has_psnr_metric:
            df_data["Average Future PSNR (dB)"].append(
                f"{suite_avg_psnr:.4f}" if suite_avg_psnr is not None else "N/A"
            )

        total_success_rate += success_rate
        total_time += stats["total_time"]
        total_suites += 1
        if has_psnr_metric:
            overall_psnr_sum += stats["psnr_sum"]
            overall_psnr_count += stats["psnr_count"]

    if total_suites > 0:
        print("\nOverall statistics:")
        avg_success_rate = total_success_rate / total_suites
        total_tasks = sum(s["total_tasks"] for s in suite_stats.values())
        avg_task_time = total_time / total_tasks if total_tasks else 0.0
        max_task_time = max(s["max_time"] for s in suite_stats.values())
        overall_avg_psnr = None
        if has_psnr_metric:
            overall_avg_psnr = overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None

        print(f"- Average success rate: {avg_success_rate:.2f}%")
        print(f"- Total time: {format_time(total_time)}")
        print(f"- Average time per task: {format_time(avg_task_time)}")
        print(f"- Longest task time: {format_time(max_task_time)}")
        if has_psnr_metric:
            if overall_avg_psnr is not None:
                print(f"- Average future-video PSNR: {overall_avg_psnr:.4f} dB")
            else:
                print("- Average future-video PSNR: N/A")

        df_data["Task Suite"].append("Overall")
        df_data["Success Rate (%)"].append(f"{avg_success_rate:.2f}")
        df_data["Average Time (s)"].append(f"{avg_task_time:.2f}")
        df_data["Max Time (s)"].append(f"{max_task_time:.2f}")
        if has_psnr_metric:
            df_data["Average Future PSNR (dB)"].append(
                f"{overall_avg_psnr:.4f}" if overall_avg_psnr is not None else "N/A"
            )

    df = pd.DataFrame(df_data)

    ckpt_path = os.environ.get("CKPT", "")
    title = os.path.basename(ckpt_path) if ckpt_path else "Results"

    df = df.set_index("Task Suite").T

    with (output_dir / "summary.csv").open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        df.to_csv(f)

    task_success_data = {
        "Task": [],
        "Task Name": [],
        "Category": [],
        "Difficulty Level": [],
        "Description": [],
        "Success Rate (%)": [],
    }
    if has_psnr_metric:
        task_success_data["Future Video PSNR (dB)"] = []

    suite_tasks = defaultdict(list)
    for task in task_results:
        suite_tasks[_suite_from_task_key(task)].append(task)

    for suite in suite_tasks:
        suite_tasks[suite].sort(key=lambda x: int(x.rsplit("_", 1)[-1]))

    for suite in sorted(suite_tasks.keys()):
        for task in suite_tasks[suite]:
            result = task_results[task]
            task_success_data["Task"].append(task)
            task_success_data["Task Name"].append(result.get("task_name", ""))
            task_success_data["Category"].append(result.get("category", ""))
            task_success_data["Difficulty Level"].append(result.get("difficulty_level", ""))
            task_success_data["Description"].append(result.get("task_description", ""))
            task_success_data["Success Rate (%)"].append(f"{result['success_rate']:.2f}")
            if has_psnr_metric:
                psnr = result["future_video_psnr_mean"] if "future_video_psnr_mean" in result else None
                task_success_data["Future Video PSNR (dB)"].append(
                    f"{psnr:.4f}" if psnr is not None else "N/A"
                )

    suite_stats_output = {
        suite: _stats_to_output(stats, has_psnr_metric)
        for suite, stats in suite_stats.items()
    }

    task_success_df = pd.DataFrame(task_success_data)
    task_success_df.to_csv(output_dir / "task_success_rates.csv", index=False)

    category_csv = None
    difficulty_csv = None
    category_df = None
    difficulty_df = None
    if category_stats:
        category_csv, category_df = _write_group_success_csv(
            output_dir,
            "category_success_rates.csv",
            "Category",
            category_stats,
            has_psnr_metric,
        )
    if difficulty_stats:
        difficulty_csv, difficulty_df = _write_group_success_csv(
            output_dir,
            "difficulty_success_rates.csv",
            "Difficulty Level",
            difficulty_stats,
            has_psnr_metric,
        )

    summary_file = output_dir / "summary.json"
    total_tasks = sum(s["total_tasks"] for s in suite_stats.values())
    overall_stats = {
        "average_success_rate": total_success_rate / total_suites if total_suites > 0 else 0,
        "total_time": total_time,
        "average_task_time": total_time / total_tasks if total_tasks else 0,
    }
    if has_psnr_metric:
        overall_stats["average_future_video_psnr"] = (
            overall_psnr_sum / overall_psnr_count if overall_psnr_count > 0 else None
        )

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": output_dir.name,
                "ckpt": os.environ.get("CKPT", ""),
                "config": os.environ.get("CONFIG", ""),
                "task_classification_path": str(classification_path) if classification_path else "",
                "task_classification_matched": classification_matched,
                "task_classification_unmatched": classification_unmatched,
                "suite_stats": suite_stats_output,
                "category_stats": {
                    key: _stats_to_output(stats, has_psnr_metric)
                    for key, stats in sorted(category_stats.items())
                },
                "difficulty_stats": {
                    key: _stats_to_output(stats, has_psnr_metric)
                    for key, stats in sorted(difficulty_stats.items(), key=lambda item: str(item[0]))
                },
                "task_results": task_results,
                "overall": overall_stats,
            },
            f,
            indent=4,
        )

    print("\n=== Run Information ===")
    print(f"Run ID: {output_dir.name}")
    print(f"Results directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Summary CSV: {output_dir / 'summary.csv'}")
    print(f"Task success rates CSV: {output_dir / 'task_success_rates.csv'}")
    if classification_path is not None:
        print(f"Task classification matched: {classification_matched}, unmatched: {classification_unmatched}")
    if category_csv is not None:
        print(f"Category success rates CSV: {category_csv}")
    if difficulty_csv is not None:
        print(f"Difficulty success rates CSV: {difficulty_csv}")

    print("\n=== Task Success Rates ===")
    print(task_success_df.to_string(index=False))

    print("\n=== Results Table ===")
    print(df.to_string(index=False))
    if category_df is not None:
        print("\n=== Category Success Rates ===")
        print(category_df.to_string(index=False))
    if difficulty_df is not None:
        print("\n=== Difficulty Success Rates ===")
        print(difficulty_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=str,
        nargs="?",
        default=None,
        help=(
            "Root directory containing evaluation results. "
            f"If omitted, use latest run under {DEFAULT_PARENT_DIR}."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory containing evaluation results",
    )
    parser.add_argument(
        "--task_classification_path",
        type=str,
        default=None,
        help=(
            "Optional LIBERO-plus task_classification.json path. Defaults to "
            "TASK_CLASSIFICATION_PATH, eval_meta.json libero_plus_root, or "
            "$LIBERO_PLUS_ROOT/libero/libero/benchmark/task_classification.json."
        ),
    )
    parser.add_argument("--task_file", default=None, help="默认 OUTPUT_DIR/tasks.txt")
    parser.add_argument("--num_trials", type=int, default=None, help="默认 eval_meta.json 中的 num_trials")
    args = parser.parse_args()

    output_dir = args.output_dir or args.root
    if output_dir is None:
        output_dir = pick_latest_run()
        print(f"[INFO] Using latest evaluation directory: {output_dir}")

    summarize_results(output_dir, task_classification_path=args.task_classification_path,
                      task_file=args.task_file, num_trials=args.num_trials)


if __name__ == "__main__":
    main()
