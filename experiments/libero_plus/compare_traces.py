"""小规模两仓库轨迹逐项精确比较；失败时报告首个偏离，不自动放宽容差。"""
import argparse
import json
import re
from pathlib import Path

import numpy as np


def compare(left, right):
    left, right = Path(left), Path(right)
    a = {str(p.relative_to(left)): p for p in left.rglob("*") if p.suffix in {".npz", ".npy"}}
    b = {str(p.relative_to(right)): p for p in right.rglob("*") if p.suffix in {".npz", ".npy"}}
    if not a or set(a) != set(b):
        raise ValueError(f"轨迹文件集合为空或不匹配：left={len(a)}, right={len(b)}")
    episode_files = sorted(left.rglob("episode.json"))
    if not episode_files:
        raise ValueError("没有完整 episode，不能判定通过")
    if {str(p.relative_to(left)) for p in episode_files} != {str(p.relative_to(right)) for p in right.rglob('episode.json')}:
        raise ValueError("episode 覆盖不同")
    arrays = 0
    for key in sorted(a):
        loaded = [np.load(path, allow_pickle=False) for path in (a[key], b[key])]
        try:
            x, y = loaded
            if key.endswith(".npy"):
                fields = [("array", x, y)]
            else:
                if set(x.files) != set(y.files):
                    raise ValueError(f"轨迹字段不同：{key}")
                fields = [(name, x[name], y[name]) for name in sorted(x.files)]
            for name, u, v in fields:
                arrays += 1
                if u.dtype != v.dtype or u.shape != v.shape or not np.array_equal(u, v):
                    diff = None
                    if u.shape == v.shape and u.size and u.dtype.kind in "biuf" and v.dtype.kind in "biuf":
                        diff = float(np.max(np.abs(u.astype(np.float64) - v.astype(np.float64))))
                    raise ValueError(f"首次偏离：{key}:{name}, shape={u.shape}/{v.shape}, max_abs_diff={diff}")
        finally:
            for item in loaded:
                if hasattr(item, "close"):
                    item.close()
    for file in episode_files:
        relative = file.relative_to(left)
        if json.loads(file.read_text()) != json.loads((right / relative).read_text()):
            raise ValueError(f"episode 结果不同：{relative}")
    return {"passed": True, "comparison": "exact", "arrays": arrays, "episodes": len(episode_files)}


def compare_run_metadata(left, right):
    from omegaconf import OmegaConf
    roots = [Path(left).parent, Path(right).parent]
    metadata = [json.loads((root / "eval_meta.json").read_text()) for root in roots]
    for key in ("eval_mode", "ckpt_sha256", "dataset_stats_sha256", "task_file_sha256", "num_trials",
                "libero_revision", "libero_paths", "model_assets", "cuda_visible_devices"):
        if metadata[0].get(key) != metadata[1].get(key):
            raise ValueError(f"运行输入不同：{key}")
        if key.endswith("sha256") and not metadata[0].get(key):
            raise ValueError(f"缺少输入校验值：{key}")
    configs = [OmegaConf.to_container(OmegaConf.load(root / "manager_config.yaml"), resolve=True) for root in roots]
    for cfg in configs:
        cfg["EVALUATION"].pop("output_dir", None)
        cfg["EVALUATION"].pop("dataset_stats_path", None)
        cfg["MULTIRUN"].pop("task_file", None)
        cfg.pop("output_dir", None)  # 未使用的训练输出时间戳
        cfg.pop("ckpt", None)        # checkpoint 已按内容 hash 检查
    if configs[0] != configs[1]:
        raise ValueError("最终 worker/模型配置不同，请比较 manager_config.yaml")
    manifests = [json.loads((root / "task_manifest.json").read_text()) for root in roots]
    if manifests[0] != manifests[1]:
        raise ValueError("benchmark 任务名称/BDDL 清单不同")
    for root, manifest, meta in zip(roots, manifests, metadata):
        coverage = json.loads((root / "coverage.json").read_text())
        if not coverage.get("passed"):
            raise ValueError(f"本次评测没有通过完整性检查：{root}")
        expected = {(row["suite"], row["task_id"], trial) for row in manifest
                    for trial in range(int(meta["num_trials"]))}
        actual = []
        for path in (root / "traces").rglob("episode.json"):
            match = re.fullmatch(r"task(\d+)_trial(\d+)", path.parent.name)
            if match is None:
                raise ValueError(f"非法 episode 路径：{path}")
            actual.append((path.parent.parent.name, int(match[1]), int(match[2])))
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"轨迹未完整覆盖本次请求的 episode：{root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path, help="参考运行的 traces 目录")
    parser.add_argument("right", type=Path, help="目标运行的 traces 目录")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compare_run_metadata(args.left, args.right)
    report = compare(args.left, args.right)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
