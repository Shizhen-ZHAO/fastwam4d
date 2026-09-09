"""Plot and summarize measured VAE-entry geometry experiments, without a model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="outputs/libero_vae_geometry_online_60steps")
    parser.add_argument("--rollout", default="outputs/libero_vae_geometry_online_rollout/libero_spatial/gpu0_task2_results.json")
    args = parser.parse_args()
    run = Path(args.run)
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    decrease = {}
    for split, ax in zip(("train", "validation"), axes):
        probes = [x for x in rows if x["kind"] == "fixed_probe" and x["split"] == split]
        before, after = probes[0]["mean_video_loss"], probes[-1]["mean_video_loss"]
        decrease[split] = 100 * (before - after) / before
        ax.plot([x["step"] for x in probes], [x["mean_video_loss"] for x in probes], marker="o")
        ax.axhline(before, color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Training step", ylabel="Fixed-probe weighted video loss",
               title=f"{split}: {decrease[split]:.2f}% loss decrease")
        ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, "3 fixed windows per split; validation episodes excluded from adapter fit, not necessarily base pretraining.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(run / "video_loss_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, key in (("Before", "baseline_generation"), ("After", "final_generation")):
        values = summary[key]["future_psnr_per_frame"]
        ax.plot(range(1, len(values) + 1), values, marker="o", label=label)
    ax.set(xlabel="Future sampled-frame index (current frame excluded)", ylabel="PSNR (dB)",
           title="One validation window, matched generation seed")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(run / "future_psnr_curve.png", dpi=160)
    plt.close(fig)
    trains = [x for x in rows if x["kind"] == "train"]
    versions = {}
    for name in ("torch", "torchvision", "transformers", "h5py", "timm", "moviepy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not in report environment"
    result = {"summary": summary, "python": platform.python_version(), "packages": versions,
              "fastwam_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "relative_video_loss_decrease_percent": decrease,
              "train_update_seconds_sum": sum(x["seconds"] for x in trains),
              "train_step_seconds_median": statistics.median(x["seconds"] for x in trains),
              "extraction_seconds_median": statistics.median(x["extraction_seconds"] for x in trains),
              "final_gates": trains[-1]["gates"], "last_train_latent_fusion": trains[-1]["fusion"],
              "rollout": json.loads(Path(args.rollout).read_text()) if Path(args.rollout).is_file() else None}
    (run / "experiment_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("summary", "packages")}, indent=2))


if __name__ == "__main__":
    main()
