"""Extract frozen geometry, train from disk or online RGB, and test online.

Default train has no tracker construction and no historical RGB IO. train-online
extracts once per forward from lazily loaded history, without a geometry cache.
Testing requires only the adapter + original weights + causal RGB.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader, Subset, default_collate

from experiment_track4world_online import load_policy, prepare_text, seed_all
from experiment_vae_geometry_online import inference_kwargs, video_metrics
from fastwam.datasets.libero_geometry import LiberoHDF5HistoryDataset
from fastwam.datasets.geometry_cache import GeometryFeatureCache, CachedLiberoGeometryDataset, atomic_create
from fastwam.datasets.geometry_cache_hdf5 import open_geometry_cache
from fastwam.utils.video_io import save_mp4


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError("Cyclic base_experiment configuration")
    cfg = OmegaConf.load(path)
    if cfg.get("base_experiment"):
        # Existing experiment configs specify repo-relative paths.
        cfg = OmegaConf.merge(load_config(cfg.base_experiment, (*seen, path)), cfg)
    return cfg


def select_indices(dataset, count):
    if count == 0:
        return list(range(len(dataset)))
    candidates = [i for i, (_, _, t) in enumerate(dataset.samples)
                  if t >= (dataset.history_length - 1) * dataset.history_stride]
    if count < 0 or count > len(candidates):
        raise ValueError(f"Requested {count} samples from {len(candidates)} full-history windows")
    return [candidates[i] for i in np.linspace(0, len(candidates) - 1, count, dtype=int)]


def make_dataset(cfg, *, validation=False, online=False):
    args = OmegaConf.to_container(cfg.data, resolve=True)
    args["load_history"] = online
    args["history_only"] = False
    if validation:
        args["episode_ids"] = list(cfg.validation_episode_ids)
    ds = LiberoHDF5HistoryDataset(**args)
    if (ds.history_length != int(cfg.geometry.history_length)
            or ds.history_stride != int(cfg.geometry.history_stride)
            or ds.history_image_size != int(cfg.geometry.extractor.image_size)
            or any(fps != float(cfg.geometry.history_fps) for fps in ds.fps.values())):
        raise ValueError("Dataset and online inference history settings differ")
    return ds


def open_output(path, cfg):
    out = Path(path)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out / "resolved_config.yaml")
    return out


def record(out, row):
    with (out / "metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


def ensure_no_tracker(model):
    if model._geometry_extractor is not None or "track4world.nets.model" in sys.modules:
        raise RuntimeError("Cached training must never instantiate/import the Track4World model")


def prepare_selected_text(dataset, cfg, indices):
    selected = copy.copy(dataset)
    paths = {dataset.samples[i][0] for i in indices}
    selected.instructions = {p: dataset.instructions[p] for p in paths}
    prepare_text(selected, cfg)


def compare_raw(first, second):
    report = {}
    for key in first:
        a, b = first[key].detach().cpu(), second[key].detach().cpu()
        report[key] = (bool(torch.equal(a, b)) if a.dtype == torch.bool
                       else {"max_abs": float(torch.nan_to_num((a - b).abs()).max()),
                             "rms": float(torch.nan_to_num(a - b).square().mean().sqrt()),
                             "reference_rms": float(torch.nan_to_num(a).square().mean().sqrt())})
    print(json.dumps({"kind": "raw_parity_errors", "errors": report}), flush=True)
    for key in first:
        # FP16 iterative tracking and 20Hz finite-difference velocities amplify
        # small CUDA rounding differences, especially around zero. Check both
        # pointwise outliers and aggregate error; never relax bool masks. This
        # is NOT the lossless serialization test (that uses zero tolerance).
        torch.testing.assert_close(first[key].cpu(), second[key].cpu(),
                                   rtol=1e-3, atol=1e-2, equal_nan=True)
        if isinstance(report[key], dict):
            if report[key]["rms"] > 1e-3 * max(report[key]["reference_rms"], 1.0):
                raise AssertionError(f"Online/cache aggregate numerical error too large: {key}: {report[key]}")
    return report


def extract(cfg, args):
    dataset = make_dataset(cfg)
    geometry = OmegaConf.to_container(cfg.geometry, resolve=True)
    cache = open_geometry_cache(cfg, dataset, create=True)
    indices = select_indices(dataset, int(cfg.train_samples))
    if args.include_episode_starts:
        indices = sorted(set(indices) | {i for i, (_, _, t) in enumerate(dataset.samples) if t == 0})
    missing = sum(not cache.contains(i) for i in indices)
    print(json.dumps({"kind": "extraction_setup", "windows": len(indices), "missing": missing,
                      "cache_dir": str(cache.root), "contract_id": cache.contract_id,
                      "reads_future_frames": False, "dtype": "float32+bool"}), flush=True)
    extractor = None
    created, resumed, seconds = 0, 0, []
    started = time.monotonic()
    for n, index in enumerate(indices):
        if cache.contains(index):
            cache.read(index)  # Resume validates existing identities, checksums and shapes.
            resumed += 1
        else:
            if extractor is None:
                from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor
                extractor = OnlineTrack4WorldExtractor(**geometry["extractor"])
            history = default_collate([dataset.history_item(index)])
            start = time.monotonic()
            raw = extractor(history["history_images"], history["history_timestamps"], history["history_valid"])
            raw = {k: v[0] for k, v in raw.items()}
            try:
                cache.write(index, raw)
                created += 1
            except FileExistsError:
                resumed += 1
            # Serialization must not quantize even one FP32 bit (NaN allowed).
            stored = cache.read(index)
            for key in raw:
                torch.testing.assert_close(stored[key], raw[key], rtol=0, atol=0, equal_nan=True)
            seconds.append(time.monotonic() - start)
        print(json.dumps({"kind": "extraction_progress", "done": n + 1, "total": len(indices),
                          "episode": dataset.samples[index][1], "frame": dataset.samples[index][2],
                          "created": created, "resumed": resumed}), flush=True)
    parity = None
    if args.verify_online:
        if extractor is None:
            from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor
            extractor = OnlineTrack4WorldExtractor(**geometry["extractor"])
        # Recompute a non-padded full-history window, independently of disk.
        index = select_indices(dataset, int(cfg.train_samples))[0]
        history = default_collate([dataset.history_item(index)])
        online = extractor(history["history_images"], history["history_timestamps"], history["history_valid"])
        parity = compare_raw(cache.read(index), {k: v[0] for k, v in online.items()})
    report = {"kind": "extraction_summary", "windows": len(indices), "indices": indices,
              "created": created, "resumed": resumed, "seconds": time.monotonic() - started,
              "median_extract_write_seconds": float(np.median(seconds)) if seconds else None,
              "window_bytes": sum(p.stat().st_size for p in {cache.entry_path(i) for i in indices}),
              "extractor_calls": getattr(extractor, "calls", 0), "roundtrip_lossless": True,
              "recomputed_online_parity": parity, "contract_id": cache.contract_id}
    atomic_create(cache.root / f"extraction_report_{time.time_ns()}.json", json_value=report)
    print(json.dumps(report, indent=2), flush=True)
    if extractor is not None:
        extractor.close()


def train(cfg, args):
    online = getattr(args, "mode", "train") == "train-online"
    geometry_source = "online_rgb" if online else "disk"
    dataset = make_dataset(cfg, online=online)
    indices = (list(cfg.train_indices) if cfg.get("train_indices") is not None
               else select_indices(dataset, int(cfg.train_samples)))
    cache_metadata = {}
    if online:
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Empty or duplicate online dataset indices")
        if any(not isinstance(i, int) or not 0 <= i < len(dataset) for i in indices):
            raise ValueError("Online dataset index out of range")
        training = Subset(dataset, indices)
    else:
        cache = open_geometry_cache(cfg, dataset)
        training = CachedLiberoGeometryDataset(dataset, cache, indices)
        cache_metadata = {"cache_contract_id": str(cache.contract_id)}
        # Coverage/identity checks are always performed above. Do not read a TB
        # at startup; each disk training batch ALWAYS verifies SHA256.
        if cfg.get("verify_cache_before_training", False):
            for i in indices:
                cache.read(i)
    out = open_output(cfg.output_dir, cfg)
    np.save(out / "train_indices.npy", np.asarray(indices, dtype=np.int64))
    prepare_selected_text(dataset, cfg, indices)
    loader = DataLoader(training, batch_size=int(cfg.batch_size), shuffle=bool(cfg.shuffle),
                        num_workers=int(cfg.num_workers), pin_memory=False,
                        persistent_workers=int(cfg.num_workers) > 0,
                        generator=torch.Generator().manual_seed(int(cfg.seed)),
                        **({"multiprocessing_context": "spawn"} if int(cfg.num_workers) else {}))
    model = load_policy(cfg)
    if args.adapter:
        model.load_geometry_adapter(args.adapter)
        print("Warm start: adapter weights only, not optimizer resume", flush=True)
    if not online:
        ensure_no_tracker(model)
    if model.geometry_layers or any(hasattr(b, "geometry_adapter") for b in model.action_expert.blocks):
        raise RuntimeError("Direct ActionDiT geometry must be disabled")
    params = [p for p in model.parameters() if p.requires_grad]
    adapter = model.mot.geometry_latent_adapter
    optimizer = torch.optim.AdamW([
        {"params": [p for p in params if p is not adapter.gates]},
        {"params": [adapter.gates], "lr": float(cfg.gate_learning_rate), "weight_decay": 0.0},
    ], lr=float(cfg.learning_rate), weight_decay=0.01)

    extraction_calls = {"train": 0, "probe": 0}

    def forward(sample, split):
        if online:
            if "geometry_raw" in sample or not all(
                    key in sample for key in ("history_images", "history_timestamps", "history_valid")):
                raise RuntimeError("Online training requires RGB history, never cached raw features")
            before = getattr(model._geometry_extractor, "calls", 0)
        else:
            if any(key.startswith("history_") for key in sample) or "geometry_raw" not in sample:
                raise RuntimeError("Cached training must use disk raw features, never RGB history")
            ensure_no_tracker(model)
        result = model.training_loss(sample)
        if online:
            calls = getattr(model._geometry_extractor, "calls", 0) - before
            if calls != 1:
                raise RuntimeError("Online training/probes must extract geometry exactly once per forward")
            extraction_calls[split] += calls
        else:
            ensure_no_tracker(model)
        return result

    def probe(step):
        losses = []
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for j in range(min(int(cfg.probe_samples), len(training))):
                seed_all(10000 + j)
                _, parts = forward(default_collate([training[j]]), "probe")
                losses.append(parts["loss_video"])
        model.configure_geometry_train_mode()
        return record(out, {"kind": "fixed_probe", "step": step, "split": "train",
                            "geometry_source": geometry_source, "video_losses": losses,
                            "extraction_calls": len(losses) if online else 0,
                            "mean_video_loss": float(np.mean(losses))})

    record(out, {"kind": "setup", "geometry_source": geometry_source,
                 **({"cache_backend": cfg.get("cache_backend", "pt")} if not online else {}),
                 "train_indices_preview": indices[:16], "training_windows": len(indices),
                 "train_index_file": str(out / "train_indices.npy"),
                 "train_windows_preview": [list(dataset.samples[i]) for i in indices[:16]],
                 "trainable_parameters": sum(p.numel() for p in params),
                 "batch_size": int(cfg.batch_size), "num_workers": int(cfg.num_workers),
                 **cache_metadata, "direct_action_geometry": False})
    train_start = time.monotonic()
    initial = probe(0)
    seed_all(int(cfg.seed) + 1)
    gradient_seen = {name: False for name in ("scene", "camera", "track")}
    tokenizer_gradient_seen = dict(gradient_seen)
    iterator = iter(loader)
    timings = []
    for step in range(1, int(cfg.steps) + 1):
        start = time.monotonic()
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            sample = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = forward(sample, "train")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        branch_grads, tokenizer_grads = {}, {}
        for name, branch in adapter.attention.branches.items():
            grad = branch.to_kv.weight.grad
            branch_grads[name] = 0.0 if grad is None else float(grad.norm())
            gradient_seen[name] |= branch_grads[name] > 0
            grad = getattr(model.mot.geometry_tokenizer, name)[1].weight.grad
            tokenizer_grads[name] = 0.0 if grad is None else float(grad.norm())
            tokenizer_gradient_seen[name] |= tokenizer_grads[name] > 0
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen base parameter unexpectedly received a gradient")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        if not online:
            ensure_no_tracker(model)
        timings.append(time.monotonic() - start)
        record(out, {"kind": "train", "step": step, **parts, "loss_total": float(loss.detach()),
                     "grad_norm": float(norm), "branch_kv_grad_norm": branch_grads,
                     "tokenizer_grad_norm": tokenizer_grads, "gates": adapter.gates.detach().cpu().tolist(),
                     "seconds": timings[-1], "extraction_calls": 1 if online else 0,
                     "geometry_source": geometry_source,
                     "fusion": adapter.last_metrics})
        if step % int(cfg.eval_every) == 0 or step == int(cfg.steps):
            final = probe(step)
    checkpoint = out / "geometry_adapter.pt"
    model.save_geometry_adapter(checkpoint, base_checkpoint=str(cfg.base_checkpoint),
                                steps=int(cfg.steps), objective="video_only", training_geometry_source=geometry_source,
                                **cache_metadata, training_windows=len(indices),
                                train_index_file="train_indices.npy",
                                train_episodes=sorted({f"{dataset.samples[i][0]}:{dataset.samples[i][1]}" for i in indices}))
    payload = model.load_geometry_adapter(checkpoint)
    for key, value in payload["geometry_adapter"].items():
        torch.testing.assert_close(model.mot.state_dict()[key].detach().cpu(), value, rtol=0, atol=0)
    if not online:
        ensure_no_tracker(model)
    report = {"experiment": ("online RGB frozen geometry, VAE-entry video-loss adapter training" if online
                             else "offline frozen geometry, VAE-entry video-loss adapter training"),
              "training_geometry_source": geometry_source,
              "steps": int(cfg.steps), "training_windows": len(indices),
              "initial_train_probe": initial, "final_train_probe": final,
              "train_probe_decreased": final["mean_video_loss"] < initial["mean_video_loss"],
              "branch_gradients_seen": gradient_seen, "tokenizer_gradients_seen": tokenizer_gradient_seen,
              "trainable_parameters": sum(p.numel() for p in params),
              "tracker_initialized": model._geometry_extractor is not None,
              "extraction_count_unit": "extractor calls per batch forward; training excludes probes",
              "online_extractions_during_training": extraction_calls["train"],
              "online_extractions_during_probes": extraction_calls["probe"],
              "online_extractor_calls": sum(extraction_calls.values()),
              "training_seconds_including_probes_and_save": time.monotonic() - train_start,
              "median_step_seconds": float(np.median(timings)), "checkpoint_reload_exact": True,
              "checkpoint": str(checkpoint), **cache_metadata,
              "validation": "Run test in a separate process; it always extracts geometry online"}
    atomic_create(out / "summary.json", json_value=report)
    print(json.dumps(report, indent=2), flush=True)


def online_test(cfg, args):
    adapter_path = Path(args.adapter or Path(cfg.output_dir) / "geometry_adapter.pt")
    payload = torch.load(adapter_path, map_location="cpu", weights_only=True)
    if payload["geometry_config"].get("target") != "vae_latent":
        raise ValueError("Online VAE test requires a VAE-entry geometry adapter checkpoint")
    # Inference uses the checkpoint's geometry architecture / online extractor,
    # never a cache-dir from training metadata.
    cfg.geometry = OmegaConf.create(payload["geometry_config"])
    dataset = make_dataset(cfg, validation=True, online=True)
    used_episodes = set(payload.get("metadata", {}).get("train_episodes", []))
    if used_episodes & {f"{p}:{e}" for p, e, _ in dataset.samples}:
        raise ValueError("Online test episodes overlap the adapter training episodes")
    indices = select_indices(dataset, int(cfg.validation_samples))
    out = open_output(args.output_dir or cfg.online_test_output_dir, cfg)
    prepare_selected_text(dataset, cfg, indices)
    samples = [default_collate([dataset[i]]) for i in indices]
    model = load_policy(cfg)
    model.load_geometry_adapter(adapter_path)
    model.eval()
    latent_adapter = model.mot.geometry_latent_adapter
    trained_gates = latent_adapter.gates.detach().clone()
    probes = {}
    with torch.no_grad():
        for label in ("zero_gate_baseline", "trained"):
            latent_adapter.gates.copy_(torch.zeros_like(trained_gates) if label == "zero_gate_baseline" else trained_gates)
            losses = []
            for j, sample in enumerate(samples):
                before = getattr(model._geometry_extractor, "calls", 0)
                seed_all(20000 + j)
                _, parts = model.training_loss(sample)
                if model._geometry_extractor.calls - before != 1:
                    raise RuntimeError("Online test must recompute geometry from RGB on each forward")
                losses.append(parts["loss_video"])
            probes[label] = record(out, {"kind": "online_validation", "label": label,
                                        "video_losses": losses, "mean_video_loss": float(np.mean(losses)),
                                        "online_extractions": len(samples), "geometry_source": "online_rgb"})
    # Optional diagnostic only. Held-out loss and predictions above/below never
    # consume cached features; normal test does not open a cache at all.
    parity = None
    if args.verify_online:
        train_dataset = make_dataset(cfg, online=False)
        cache = open_geometry_cache(cfg, train_dataset)
        index = (int(cfg.train_indices[0]) if cfg.get("train_indices") is not None
                 else select_indices(train_dataset, int(cfg.train_samples))[0])
        rgb_sample = default_collate([train_dataset[index]])
        rgb_sample.update(default_collate([train_dataset.history_item(index)]))
        raw = model._geometry_extractor(rgb_sample["history_images"], rgb_sample["history_timestamps"], rgb_sample["history_valid"])
        cached_raw = default_collate([cache.read(index)])
        raw_parity = compare_raw(cached_raw, raw)
        repeated = model._geometry_extractor(rgb_sample["history_images"], rgb_sample["history_timestamps"], rgb_sample["history_valid"])
        online_repeat_parity = compare_raw(raw, repeated)
        disk_sample = {k: v for k, v in rgb_sample.items() if not k.startswith("history_")}
        disk_sample["geometry_raw"] = cached_raw
        with torch.no_grad():
            seed_all(30000)
            disk_loss, _ = model.training_loss(disk_sample)
            seed_all(30000)
            disk_repeat_loss, _ = model.training_loss(disk_sample)
            seed_all(30000)
            rgb_loss, _ = model.training_loss(rgb_sample)
            seed_all(30000)
            rgb_repeat_loss, _ = model.training_loss(rgb_sample)
        # Separate exact disk-path repeatability from independent FP16 tracker
        # runs followed by a BF16 world model. Sub-ULP condition changes may
        # flip BF16 rounding at the world-model input; log all control values.
        loss_rtol = max(1e-4, float(torch.finfo(model.torch_dtype).eps))
        parity = {"raw": raw_parity, "online_vs_online_control": online_repeat_parity,
                  "cached_loss": float(disk_loss), "online_loss": float(rgb_loss),
                  "cached_repeat_loss": float(disk_repeat_loss), "online_repeat_loss": float(rgb_repeat_loss),
                  "loss_rtol": loss_rtol, "loss_atol": 1e-5,
                  "absolute_loss_difference": float((disk_loss - rgb_loss).abs())}
        record(out, {"kind": "optional_cache_parity_diagnostic", **parity})
        torch.testing.assert_close(disk_loss, disk_repeat_loss, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(disk_loss, rgb_loss, rtol=loss_rtol, atol=1e-5)
        torch.testing.assert_close(rgb_loss, rgb_repeat_loss, rtol=loss_rtol, atol=1e-5)
    predictions, reports = {}, {}
    sample = samples[0]
    for label in ("zero_gate_baseline", "trained"):
        with torch.no_grad():
            latent_adapter.gates.copy_(torch.zeros_like(trained_gates) if label == "zero_gate_baseline" else trained_gates)
        before, before_fusions = model._geometry_extractor.calls, latent_adapter.calls
        start = time.monotonic()
        prediction = model.infer_joint(**inference_kwargs(sample, cfg), num_video_frames=sample["video"].shape[2])
        if model._geometry_extractor.calls - before != 1 or latent_adapter.calls - before_fusions != 1:
            raise RuntimeError("Expected one online extraction and one current-latent fusion per joint generation")
        if not torch.isfinite(prediction["action"]).all():
            raise FloatingPointError("Non-finite online joint action")
        predictions[label] = prediction
        reports[label] = record(out, {"kind": "generation", "label": label, **video_metrics(prediction, sample),
                                     "seconds": time.monotonic() - start, "online_extractions": 1,
                                     "latent_fusions": 1, "action_shape": list(prediction["action"].shape)})
        save_mp4(prediction["video"], str(out / f"{label}_prediction.mp4"), fps=5)
        torch.save(prediction["action"], out / f"{label}_joint_action.pt")
    before, before_fusions = model._geometry_extractor.calls, latent_adapter.calls
    action = model.infer_action(**inference_kwargs(sample, cfg))
    if model._geometry_extractor.calls - before != 1 or latent_adapter.calls - before_fusions != 1:
        raise RuntimeError("Action inference must extract/fuse online once, even with visual KV caching")
    torch.testing.assert_close(action["action"], predictions["trained"]["action"], rtol=1e-2, atol=1e-2)
    torch.save(action, out / "online_action.pt")
    report = {"geometry_source": "online_rgb", "cache_required_for_test": False,
              "adapter": str(adapter_path), "validation_windows": [list(dataset.samples[i]) for i in indices],
              "validation": probes, "generation": reports, "online_extractor_calls": model._geometry_extractor.calls,
              "cached_action_matches_joint": True,
              "action_max_abs_difference": float((action["action"] - predictions["trained"]["action"]).abs().max()),
              "optional_cache_parity_diagnostic": parity,
              "caveat": "Small adapter-held-out check, not full LIBERO success-rate evaluation; base pretraining may include these episodes"}
    atomic_create(out / "summary.json", json_value=report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("extract", "train", "train-online", "test"))
    parser.add_argument("--config", default="configs/experiments/libero_vae_geometry_cached.yaml")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--train-samples", type=int, help="0 = all windows, including padded starts")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--adapter", help="train/train-online: weight-only warm start. Test: adapter to evaluate")
    parser.add_argument("--include-episode-starts", action="store_true", help="Extract extra padded-start windows")
    parser.add_argument("--verify-online", action="store_true", help="Extract/test only: compare cache against online recomputation")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ("cache_dir", "train_samples", "steps", "batch_size", "num_workers", "inference_steps"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.output_dir and args.mode != "test":
        cfg.output_dir = args.output_dir
    if (cfg.geometry.target != "vae_latent" or cfg.geometry.train_mode != "adapters"
            or float(cfg.training_loss_weights.video) != 1 or float(cfg.training_loss_weights.action) != 0):
        raise ValueError("This runner requires VAE-entry adapters and video-only loss")
    if min(int(cfg[k]) for k in ("steps", "batch_size", "probe_samples", "validation_samples", "eval_every", "inference_steps")) < 1:
        raise ValueError("Steps, batch size, probes and inference steps must be positive")
    if int(cfg.num_workers) < 0 or int(cfg.train_samples) < 0:
        raise ValueError("num_workers/train_samples must be nonnegative")
    if args.mode in ("train", "train-online") and (args.verify_online or args.include_episode_starts):
        raise ValueError("Cache verification/extraction flags are not allowed during training")
    if args.include_episode_starts and args.mode != "extract":
        raise ValueError("--include-episode-starts applies only to extraction")
    torch.set_num_threads(4)
    seed_all(int(cfg.seed))
    {"extract": extract, "train": train, "train-online": train, "test": online_test}[args.mode](cfg, args)


if __name__ == "__main__":
    main()
