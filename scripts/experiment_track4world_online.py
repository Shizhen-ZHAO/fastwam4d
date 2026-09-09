"""Online Track4World + real, full-size FastWAM LIBERO adapter experiment.

Fixed-seed probes make before/after losses comparable. Training draws fresh noise.
Every training/probe/inference call recomputes tracker features from RGB history.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from fastwam.datasets.libero_geometry import LiberoHDF5HistoryDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_text(dataset, cfg):
    missing = [p for p in dataset.prompts if not dataset.context_path(p).is_file()]
    if not missing:
        return
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    print("Loading local T5 for text context only", flush=True)
    tokenizer = HuggingfaceTokenizer(cfg.tokenizer_path, seq_len=dataset.context_len, clean="whitespace")
    encoder = _load_registered_model(cfg.text_weights, "wan_video_text_encoder",
                                    torch.bfloat16, cfg.model_device).eval().requires_grad_(False)
    dataset.text_cache_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for prompt in missing:
            ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
            context = encoder(ids.to(cfg.model_device), mask.to(cfg.model_device))
            path = dataset.context_path(prompt)
            torch.save({"context": context[0].cpu(), "mask": mask[0].cpu()}, path)
            print(f"Prepared genuine T5 context: {path}", flush=True)
    del encoder, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def load_policy(cfg):
    from fastwam.runtime import create_fastwam
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = cfg.model_base_path
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    original = OmegaConf.load("configs/model/fastwam.yaml")
    root = OmegaConf.create({"data": {"train": {"processor": {"action_output_dim": 7, "proprio_output_dim": 8}}},
                            "model": original})
    root.model.mot_checkpoint_mixed_attn = True
    params = OmegaConf.to_container(root.model, resolve=True)
    params.pop("_target_")
    params.update(skip_dit_load_from_pretrain=True, load_text_encoder=False)
    params["redirect_common_files"] = bool(cfg.get("redirect_common_files", True))
    print("Loading original full-size FastWAM LIBERO policy", flush=True)
    model = create_fastwam(**params, model_dtype=torch.bfloat16, device=cfg.model_device)
    model.load_checkpoint(cfg.base_checkpoint)
    if cfg.get("training_loss_weights") is not None:
        model.loss_lambda_video = float(cfg.training_loss_weights.video)
        model.loss_lambda_action = float(cfg.training_loss_weights.action)
    model.enable_geometry(OmegaConf.to_container(cfg.geometry, resolve=True))
    model.configure_geometry_train_mode()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/libero_track4world_online.yaml")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--prepare-text", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--adapter")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.output_dir:
        cfg.output_dir = args.output_dir
    torch.set_num_threads(4)
    seed_all(int(cfg.seed))
    out = Path(cfg.output_dir)
    if not args.prepare_text and not args.extract_only and (out / "metrics.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}; choose a fresh --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "resolved_config.yaml")
    dataset = LiberoHDF5HistoryDataset(**OmegaConf.to_container(cfg.data, resolve=True))
    if (dataset.history_length != int(cfg.geometry.history_length)
            or dataset.history_stride != int(cfg.geometry.history_stride)
            or dataset.history_image_size != int(cfg.geometry.extractor.image_size)
            or any(fps != float(cfg.geometry.history_fps) for fps in dataset.fps.values())):
        raise ValueError("Training data history length/stride/size/fps must match rollout geometry configuration")
    prepare_text(dataset, cfg)
    if args.prepare_text:
        return
    # A small fixed training set across multiple decision times; not a benchmark.
    candidates = [i for i, (_, _, t) in enumerate(dataset.samples)
                  if t >= (dataset.history_length - 1) * dataset.history_stride]
    indices = [candidates[i] for i in np.linspace(0, len(candidates) - 1, int(cfg.train_samples), dtype=int)]
    samples = [default_collate([dataset[i]]) for i in indices]
    if args.extract_only:
        from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor
        extractor = OnlineTrack4WorldExtractor(**OmegaConf.to_container(cfg.geometry.extractor, resolve=True))
        sample = samples[0]
        raw = extractor(sample["history_images"], sample["history_timestamps"], sample["history_valid"])
        report = {"shapes": {k: list(v.shape) for k, v in raw.items()},
                  "finite": {k: bool(torch.isfinite(v).all()) for k, v in raw.items()},
                  "seconds": extractor.last_seconds, "quality": extractor.last_quality}
        (out / "extractor_smoke.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    model = load_policy(cfg)
    if args.adapter:
        print("Warm-starting adapter weights; optimizer state and step count are NOT resumed", flush=True)
        model.load_geometry_adapter(args.adapter)
    params = [p for p in model.parameters() if p.requires_grad]
    gates = [model.action_expert.blocks[i].geometry_adapter.gates for i in model.geometry_layers]
    gate_ids = {id(p) for p in gates}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if id(p) not in gate_ids]},
        {"params": gates, "lr": float(cfg.get("gate_learning_rate", cfg.learning_rate)), "weight_decay": 0.0},
    ], lr=float(cfg.learning_rate), weight_decay=0.01)
    parameter_count = sum(p.numel() for p in params)
    print(f"Trainable adapter parameters: {parameter_count:,}", flush=True)
    metrics = out / "metrics.jsonl"
    if metrics.exists():
        raise FileExistsError(f"Refusing to overwrite experiment log: {metrics}; choose a fresh --output-dir")

    def record(row):
        with metrics.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def probes(step):
        results = []
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for j, sample in enumerate(samples[:int(cfg.probe_samples)]):
                seed_all(10000 + j)
                _, losses = model.training_loss(sample)
                results.append(losses["loss_action"])
        row = {"kind": "fixed_probe", "step": step, "action_losses": results,
               "mean_action_loss": float(np.mean(results))}
        record(row)
        model.configure_geometry_train_mode()
        return row

    initial = probes(0)
    seed_all(int(cfg.seed) + 1)
    for step in range(1, int(cfg.steps) + 1):
        start = time.monotonic()
        sample = samples[(step - 1) % len(samples)]
        optimizer.zero_grad(set_to_none=True)
        loss, parts = model.training_loss(sample)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        gates = {str(i): model.action_expert.blocks[i].geometry_adapter.gates.detach().float().cpu().tolist()
                 for i in model.geometry_layers}
        record({"kind": "train", "step": step, **parts, "loss_total": float(loss.detach()),
                "grad_norm": float(grad), "seconds": time.monotonic() - start,
                "extractor_seconds": model._geometry_extractor.last_seconds,
                "extractor_calls": model._geometry_extractor.calls,
                "geometry_quality": model._geometry_extractor.last_quality, "gates": gates})
        if step % int(cfg.eval_every) == 0:
            probes(step)
    final = probes(int(cfg.steps))
    adapter_path = out / "geometry_adapter.pt"
    model.save_geometry_adapter(adapter_path, base_checkpoint=cfg.base_checkpoint,
                                steps=int(cfg.steps), dataset_indices=indices)
    # Inference must extract from RGB again, exactly once for all denoising steps.
    sample = samples[0]
    calls = model._geometry_extractor.calls
    start = time.monotonic()
    prediction = model.infer_action(
        prompt=None, input_image=sample["video"][:, :, 0], action_horizon=cfg.data.horizon,
        proprio=sample["proprio"][:, 0], context=sample["context"], context_mask=sample["context_mask"],
        history_images=sample["history_images"], history_timestamps=sample["history_timestamps"],
        history_valid=sample["history_valid"], num_inference_steps=int(cfg.inference_steps), seed=123,
    )
    inference_calls = model._geometry_extractor.calls - calls
    action = prediction["action"]
    if inference_calls != 1 or not torch.isfinite(action).all():
        raise RuntimeError("Online action inference failed extraction-count/finite checks")
    torch.save(prediction, out / "online_inference.pt")
    report = {"experiment": "real LIBERO HDF5 small-set online adapter training",
              "steps": int(cfg.steps), "trainable_parameters": parameter_count,
              "initial_probe": initial, "final_probe": final,
              "probe_loss_decreased": final["mean_action_loss"] < initial["mean_action_loss"],
              "inference_shape": list(action.shape), "inference_finite": bool(torch.isfinite(action).all()),
              "inference_seconds": time.monotonic() - start, "inference_extractor_calls": inference_calls,
              "geometry_features_precomputed": False, "adapter_checkpoint": str(adapter_path),
              "caveat": "Small-set optimization/inference smoke, not held-out success or full LIBERO benchmark."}
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
