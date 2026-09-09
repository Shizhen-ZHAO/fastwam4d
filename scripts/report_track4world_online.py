"""Summarize measured online experiment artifacts; does not run a model."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/libero_track4world_online_60steps")
    parser.add_argument("--rollout", default="outputs/libero_track4world_online_rollout/libero_spatial/gpu0_task2_results.json")
    parser.add_argument("--rollout-log", default="eval_libero_single.log")
    args = parser.parse_args()
    run = Path(args.run)
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    probes = {row["step"]: row for row in rows if row["kind"] == "fixed_probe"}
    probes = [probes[step] for step in sorted(probes)]
    trains = [row for row in rows if row["kind"] == "train"]
    summary = json.loads((run / "summary.json").read_text())
    rollout = json.loads(Path(args.rollout).read_text())
    before, after = probes[0]["mean_action_loss"], probes[-1]["mean_action_loss"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([p["step"] for p in probes], [p["mean_action_loss"] for p in probes], marker="o")
    ax.set(xlabel="Adapter training step", ylabel="Fixed-probe weighted action loss",
           title=f"Online Track4World + FastWAM: {(before-after)/before*100:.2f}% decrease")
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, "3 fixed training-window probes, identical noise seeds; not held-out evaluation.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(run / "fixed_probe_loss.png", dpi=160)
    plt.close(fig)
    package_names = ("torch", "torchvision", "transformers", "h5py", "timm", "moviepy", "e3nn", "pycolmap")
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not found in this report process"
    warm = []
    if Path(args.rollout_log).is_file():
        start = datetime.strptime(rollout["start_time"], "%Y-%m-%d %H:%M:%S")
        for line in Path(args.rollout_log).read_text().splitlines():
            if "Online geometry replan:" not in line:
                continue
            stamp = datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")
            match = re.search(r"total_seconds=([\d.]+) tracker_seconds=([\d.]+)", line)
            if stamp >= start and match:
                warm.append([float(value) for value in match.groups()])
        count = sum(ep["extractor_calls"] for ep in rollout["online_geometry"]["episodes"])
        warm = warm[:count][1:]  # Exclude cold first call, including lazy model load.
    report = {
        "python": platform.python_version(), "packages": versions, "torch_cuda": torch.version.cuda,
        "fastwam_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "train_steps": len(trains), "training_update_seconds_sum": sum(x["seconds"] for x in trains),
        "training_step_seconds_median": statistics.median(x["seconds"] for x in trains),
        "online_extraction_seconds_median": statistics.median(x["extractor_seconds"] for x in trains),
        "fixed_probes": probes, "relative_probe_loss_decrease_percent": (before-after)/before*100,
        "inference": {k: v for k, v in summary.items() if k.startswith("inference_")},
        "rollout": rollout,
        "rollout_warm_replan_seconds_median": statistics.median(x[0] for x in warm) if warm else None,
        "scope": "Small-set optimization and one online simulation episode; no baseline success comparison or held-out generalization claim.",
    }
    (run / "experiment_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
