"""Real LIBERO: one online three-bank residual on the current VAE latent.

Train VIDEO loss only. Future targets, solver state and VAE decoder are untouched.
Baseline/validation generation uses matching seeds, no geometry feature files.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torch
from torch.utils.data import default_collate

from experiment_track4world_online import load_policy, prepare_text, seed_all
from fastwam.datasets.libero_geometry import LiberoHDF5HistoryDataset
from fastwam.utils.video_io import save_mp4


def choose_samples(dataset, count):
    candidates = [i for i, (_, _, t) in enumerate(dataset.samples)
                  if t >= (dataset.history_length - 1) * dataset.history_stride]
    if count < 1 or count > len(candidates):
        raise ValueError(f"Requested {count} samples from {len(candidates)} full-history windows")
    indices = [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]
    return indices, [default_collate([dataset[i]]) for i in indices]


def inference_kwargs(sample, cfg):
    return dict(prompt=None, input_image=sample["video"][:, :, 0], action_horizon=int(cfg.data.horizon),
                proprio=sample["proprio"][:, 0], context=sample["context"], context_mask=sample["context_mask"],
                history_images=sample["history_images"], history_timestamps=sample["history_timestamps"],
                history_valid=sample["history_valid"], num_inference_steps=int(cfg.inference_steps), seed=123)


def video_metrics(prediction, sample):
    predicted = np.stack([np.asarray(frame) for frame in prediction["video"]]).astype(np.float32)
    ground_truth = ((sample["video"][0].permute(1, 2, 3, 0).float().numpy() + 1) * 127.5).clip(0, 255)
    if predicted.shape != ground_truth.shape or not np.isfinite(predicted).all():
        raise ValueError("Generated video shape/finite check failed")
    # Exclude the conditioned first frame from prediction quality.
    mse = ((predicted[1:] - ground_truth[1:]) ** 2).mean(axis=(1, 2, 3))
    return {"frames": len(predicted), "frame_shape": list(predicted.shape[1:]),
            "future_psnr_per_frame": (10 * np.log10(255 ** 2 / np.maximum(mse, 1e-8))).tolist(),
            "future_mean_psnr": float((10 * np.log10(255 ** 2 / np.maximum(mse, 1e-8))).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/libero_vae_geometry_online.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--adapter", help="Warm-start weights only, not optimizer/step resume")
    args = parser.parse_args()
    specific = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(OmegaConf.load(specific.base_experiment), specific)
    for argument, field in ((args.steps, "steps"), (args.inference_steps, "inference_steps"), (args.output_dir, "output_dir")):
        if argument is not None:
            cfg[field] = argument
    if cfg.geometry.target != "vae_latent" or cfg.geometry.train_mode != "adapters":
        raise ValueError("This experiment is exclusively VAE-entry adapter training")
    if float(cfg.training_loss_weights.video) != 1 or float(cfg.training_loss_weights.action) != 0:
        raise ValueError("V1 optimizes original video loss only (video=1, action=0)")
    if int(cfg.steps) < 1 or int(cfg.inference_steps) < 1:
        raise ValueError("steps/inference_steps must be positive")
    out = Path(cfg.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Choose an empty output directory; refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "resolved_config.yaml")
    torch.set_num_threads(4)
    seed_all(int(cfg.seed))
    data_args = OmegaConf.to_container(cfg.data, resolve=True)
    dataset = LiberoHDF5HistoryDataset(**data_args)
    validation_args = dict(data_args, episode_ids=list(cfg.validation_episode_ids))
    validation = LiberoHDF5HistoryDataset(**validation_args)
    train_episodes = {(path, ep) for path, ep, _ in dataset.samples}
    validation_episodes = {(path, ep) for path, ep, _ in validation.samples}
    if train_episodes & validation_episodes:
        raise ValueError("Training/validation episodes overlap")
    for ds in (dataset, validation):
        if (ds.history_length != int(cfg.geometry.history_length) or ds.history_stride != int(cfg.geometry.history_stride)
                or ds.history_image_size != int(cfg.geometry.extractor.image_size)
                or any(fps != float(cfg.geometry.history_fps) for fps in ds.fps.values())):
            raise ValueError("Training/inference history configuration mismatch")
        prepare_text(ds, cfg)
    indices, samples = choose_samples(dataset, int(cfg.train_samples))
    validation_indices, validation_samples = choose_samples(validation, int(cfg.validation_samples))
    model = load_policy(cfg)
    if args.adapter:
        print("Warm-start weights only; optimizer/steps start fresh", flush=True)
        model.load_geometry_adapter(args.adapter)
    if model.geometry_layers or any(hasattr(block, "geometry_adapter") for block in model.action_expert.blocks):
        raise RuntimeError("Direct ActionDiT geometry injection must be disabled")
    params = [p for p in model.parameters() if p.requires_grad]
    adapter = model.mot.geometry_latent_adapter
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if p is not adapter.gates]},
        {"params": [adapter.gates], "lr": float(cfg.gate_learning_rate), "weight_decay": 0.0},
    ], lr=float(cfg.learning_rate), weight_decay=0.01)
    metrics_path = out / "metrics.jsonl"

    def record(row):
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def fixed_probe(step, split, probe_samples):
        losses = []
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for j, sample in enumerate(probe_samples):
                seed_all((10000 if split == "train" else 20000) + j)
                _, parts = model.training_loss(sample)
                losses.append(parts["loss_video"])
        row = {"kind": "fixed_probe", "step": step, "split": split,
               "video_losses": losses, "mean_video_loss": float(np.mean(losses))}
        record(row)
        model.configure_geometry_train_mode()
        return row

    def generate(label, sample):
        before_calls = getattr(model._geometry_extractor, "calls", 0)
        before_fusions = adapter.calls
        start = time.monotonic()
        prediction = model.infer_joint(**inference_kwargs(sample, cfg), num_video_frames=sample["video"].shape[2])
        elapsed = time.monotonic() - start
        if model._geometry_extractor.calls - before_calls != 1 or adapter.calls - before_fusions != 1:
            raise RuntimeError("Expected exactly ONE online extraction and ONE latent fusion per generation")
        if not torch.isfinite(prediction["action"]).all():
            raise FloatingPointError("Non-finite joint action prediction")
        report = {"kind": "generation", "label": label, **video_metrics(prediction, sample), "seconds": elapsed,
                  "online_extractions": 1, "latent_fusions": 1, "action_shape": list(prediction["action"].shape)}
        save_mp4(prediction["video"], str(out / f"{label}_prediction.mp4"), fps=5)
        torch.save(prediction["action"], out / f"{label}_joint_action.pt")
        record(report)
        model.configure_geometry_train_mode()
        return report, prediction

    initial_train = fixed_probe(0, "train", samples[:int(cfg.probe_samples)])
    initial_validation = fixed_probe(0, "validation", validation_samples)
    baseline_generation, baseline_prediction = generate("before", validation_samples[0])
    seed_all(int(cfg.seed) + 1)
    gradient_seen = {name: False for name in ("scene", "camera", "track")}
    record({"kind": "setup", "trainable_parameters": sum(p.numel() for p in params),
            "train_indices": indices, "validation_indices": validation_indices,
            "train_windows": [(dataset.samples[i][1], dataset.samples[i][2]) for i in indices],
            "validation_windows": [(validation.samples[i][1], validation.samples[i][2]) for i in validation_indices],
            "direct_action_geometry": False, "loss_video_weight": 1.0, "loss_action_weight": 0.0})
    final_train, final_validation = initial_train, initial_validation
    for step in range(1, int(cfg.steps) + 1):
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.training_loss(samples[(step - 1) % len(samples)])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        branch_grad = {}
        for name, branch in adapter.attention.branches.items():
            grad = branch.to_kv.weight.grad
            branch_grad[name] = 0.0 if grad is None else float(grad.norm())
            gradient_seen[name] |= branch_grad[name] > 0
        if any(p.grad is not None for p in model.video_expert.parameters()):
            raise RuntimeError("Frozen world backbone unexpectedly received parameter gradients")
        if any(p.grad is not None for p in model.action_expert.parameters()):
            raise RuntimeError("Frozen action backbone unexpectedly received parameter gradients")
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        record({"kind": "train", "step": step, **parts, "loss_total": float(loss.detach()),
                "grad_norm": float(grad_norm), "branch_kv_grad_norm": branch_grad,
                "gates": adapter.gates.detach().cpu().tolist(), "seconds": time.monotonic() - start,
                "extraction_seconds": model._geometry_extractor.last_seconds,
                "extraction_calls": model._geometry_extractor.calls, "fusion": adapter.last_metrics})
        if step % int(cfg.eval_every) == 0 or step == int(cfg.steps):
            final_train = fixed_probe(step, "train", samples[:int(cfg.probe_samples)])
            final_validation = fixed_probe(step, "validation", validation_samples)
    checkpoint = out / "geometry_adapter.pt"
    model.save_geometry_adapter(checkpoint, base_checkpoint=cfg.base_checkpoint, steps=int(cfg.steps),
                                objective="video_only", train_indices=indices, validation_indices=validation_indices)
    # Read back serialized weights before generation. Fresh-process reload is also
    # supported by the official LIBERO rollout entrypoint, using this checkpoint.
    reloaded = model.load_geometry_adapter(checkpoint)
    for key, value in reloaded["geometry_adapter"].items():
        torch.testing.assert_close(model.mot.state_dict()[key].detach().cpu(), value, rtol=0, atol=0)
    del reloaded
    final_generation, final_prediction = generate("after", validation_samples[0])
    sample = validation_samples[0]
    before_calls, before_fusions = model._geometry_extractor.calls, adapter.calls
    start = time.monotonic()
    cached = model.infer_action(**inference_kwargs(sample, cfg))
    cache_seconds = time.monotonic() - start
    if model._geometry_extractor.calls - before_calls != 1 or adapter.calls - before_fusions != 1:
        raise RuntimeError("Cached action inference must extract/fuse exactly once")
    torch.testing.assert_close(cached["action"], final_prediction["action"], atol=1e-2, rtol=1e-2)
    if not torch.isfinite(cached["action"]).all():
        raise FloatingPointError("Non-finite cached action")
    torch.save(cached, out / "online_action.pt")
    ground_truth = ((sample["video"][0].permute(1, 2, 3, 0).float().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    comparison = [Image.fromarray(np.concatenate([gt, np.asarray(old), np.asarray(new)], axis=0))
                  for gt, old, new in zip(ground_truth, baseline_prediction["video"], final_prediction["video"])]
    save_mp4(comparison, str(out / "comparison_gt_before_after.mp4"), fps=5)
    comparison[-1].save(out / "comparison_last_frame.png")
    report = {"experiment": "current-VAE-latent three-way residual online geometry, video-only objective",
              "steps": int(cfg.steps), "trainable_parameters": sum(p.numel() for p in params),
              "initial_train_probe": initial_train, "final_train_probe": final_train,
              "initial_validation_probe": initial_validation, "final_validation_probe": final_validation,
              "train_probe_decreased": final_train["mean_video_loss"] < initial_train["mean_video_loss"],
              "validation_probe_decreased": final_validation["mean_video_loss"] < initial_validation["mean_video_loss"],
              "branch_gradients_seen": gradient_seen, "baseline_generation": baseline_generation,
              "final_generation": final_generation, "checkpoint_reload_exact": True,
              "cached_action_matches_joint": True, "cached_action_max_abs_difference": float((cached["action"] - final_prediction["action"]).abs().max()),
              "cached_action_seconds": cache_seconds, "cached_action_shape": list(cached["action"].shape),
              "direct_action_geometry": False, "geometry_features_precomputed": False,
              "checkpoint": str(checkpoint), "caveat": "Small-set experiment. Validation episodes held out only from this adapter fit; no claim that the base checkpoint never saw them or that full LIBERO success improves."}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
