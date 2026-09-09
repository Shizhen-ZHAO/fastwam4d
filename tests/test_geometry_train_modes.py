"""Exercise the real runner with CPU FastWAM and tiny HDF5/tracker fixtures."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from fastwam.datasets.geometry_cache import CachedLiberoGeometryDataset, GeometryFeatureCache
from fastwam.datasets.geometry_cache_hdf5 import HDF5GeometryFeatureCache
from fastwam.datasets.libero_geometry import LiberoHDF5HistoryDataset
from fastwam.models.wan22 import track4world_online
from test_geometry_cache import fixture, single_raw
from test_vae_geometry import tiny_policy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import experiment_vae_geometry_cached as runner


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden geometry cache/tracker/history access")


class BatchFixtureExtractor:
    """One public call per batch, with the sample loop inside the extractor."""

    def __init__(self, raw):
        self.raw, self.calls, self.batch_sizes = raw, 0, []

    def __call__(self, images, timestamps, valid, output_device="cpu"):
        self.calls += 1
        self.batch_sizes.append(len(images))
        assert images.device.type == "cpu"
        assert images.shape[1:4] == (2, 8, 3)
        assert (timestamps <= 0).all() and valid[:, -1].all()
        batch = [{key: value.to(output_device) for key, value in self.raw.items()}
                 for _ in images]
        return {key: torch.stack([item[key] for item in batch]) for key in self.raw}


@pytest.fixture
def training_case(fixture, tmp_path, monkeypatch):
    dataset, geometry, root = fixture
    geometry.update(target="vae_latent", latent_channels=4, memory_dim=32,
                    inner_dim=32, heads=2, temporal_layers=1, train_mode="adapters")
    geometry["extractor"]["device"] = "cpu"
    cfg = OmegaConf.create(dict(
        data=dict(files=dataset.files, stats_path=dataset.stats_path,
                  text_cache_dir=str(dataset.text_cache_dir), image_size=16,
                  history_image_size=8, context_len=3, load_history=False),
        geometry=geometry, output_dir=str(tmp_path / "run"), base_checkpoint="fixture-base.pt",
        train_samples=0, steps=3, batch_size=2, num_workers=0, shuffle=False,
        seed=123, probe_samples=2, eval_every=3, learning_rate=1e-3,
        gate_learning_rate=1e-2, training_loss_weights=dict(video=1., action=0.),
        validation_samples=1, inference_steps=1, verify_cache_before_training=True,
    ))
    context_dataset = runner.make_dataset(cfg)
    torch.save({"context": torch.zeros(3, 48), "mask": torch.ones(3, dtype=torch.bool)},
               context_dataset.context_path(context_dataset.prompts[0]))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(567)
        raw = single_raw()
    case = SimpleNamespace(cfg=cfg, dataset=dataset, root=root, raw=raw,
                           models=[], extractors=[], forwards=[], reads=[], loaders=[], base={})
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    getitem = LiberoHDF5HistoryDataset.__getitem__

    def read_item(ds, index):
        case.reads.append(index)
        return getitem(ds, index)

    monkeypatch.setattr(LiberoHDF5HistoryDataset, "__getitem__", read_item)

    def load_tiny(config):
        # No selected RGB is materialized before model construction.
        assert case.reads == []
        model = tiny_policy(checkpoint=True)
        model.enable_geometry(OmegaConf.to_container(config.geometry, resolve=True))
        model.configure_geometry_train_mode()
        case.base = {name: value.detach().clone() for name, value in model.named_parameters()
                     if not value.requires_grad}
        original = model.training_loss

        def forward(sample):
            case.forwards.append((torch.is_grad_enabled(), sample["decision_frame"].tolist()))
            assert ("geometry_raw" in sample) != ("history_images" in sample)
            assert model.loss_lambda_video == 1 and model.loss_lambda_action == 0
            assert model.geometry_layers == []
            return original(sample)

        monkeypatch.setattr(model, "training_loss", forward)
        case.models.append(model)
        return model

    def make_extractor(**kwargs):
        extractor = BatchFixtureExtractor(raw)
        case.extractors.append(extractor)
        return extractor

    def loader_spy(training, **kwargs):
        assert case.reads == []
        loader = DataLoader(training, **kwargs)
        case.loaders.append(loader)
        return loader

    monkeypatch.setattr(runner, "load_policy", load_tiny)
    monkeypatch.setattr(runner, "DataLoader", loader_spy)
    monkeypatch.setattr(track4world_online, "OnlineTrack4WorldExtractor", make_extractor)
    yield case
    # The spy holds the loader; close only these fixture workers after assertions.
    for loader in case.loaders:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()
    torch.set_num_threads(previous_threads)


def configure_source(case, monkeypatch, online):
    if online:
        # Deliberately omit cache_dir/backend, while verify_cache_before_training
        # remains true: the online branch must not even attempt to open a cache.
        monkeypatch.setattr(runner, "open_geometry_cache", forbidden)
        monkeypatch.setattr(runner, "CachedLiberoGeometryDataset", forbidden)
        monkeypatch.setattr(GeometryFeatureCache, "__init__", forbidden)
        monkeypatch.setattr(HDF5GeometryFeatureCache, "__init__", forbidden)
    else:
        case.cfg.cache_dir = str(case.root)
        cache = GeometryFeatureCache(case.root, case.dataset,
                                     OmegaConf.to_container(case.cfg.geometry, resolve=True), create=True)
        for index in range(len(case.dataset)):
            cache.write(index, case.raw)
        monkeypatch.setattr(track4world_online, "OnlineTrack4WorldExtractor", forbidden)
        make_dataset = runner.make_dataset

        def no_history_dataset(*args, **kwargs):
            dataset = make_dataset(*args, **kwargs)
            # Patch only this instance: class source participates in the cache
            # preprocessing fingerprint and must stay unchanged, even in tests.
            monkeypatch.setattr(dataset, "history_item", forbidden)
            return dataset

        monkeypatch.setattr(runner, "make_dataset", no_history_dataset)


@pytest.mark.parametrize("online", [False, True], ids=["disk", "online"])
@pytest.mark.parametrize("workers", [0, 1], ids=["main-process", "spawn"])
@pytest.mark.parametrize("selection", [None, [3, 0, 2]], ids=["all", "explicit-order"])
def test_runner_modes_stream_batches_and_save_safe_metadata(training_case, monkeypatch,
                                                           online, workers, selection):
    case, cfg = training_case, training_case.cfg
    cfg.num_workers = workers
    if selection is not None:
        cfg.train_indices = selection
        cfg.train_samples = 999  # Explicit indices take precedence over count.
    expected = list(range(len(case.dataset))) if selection is None else selection
    configure_source(case, monkeypatch, online)
    # The old train(cfg, args) interface (no args.mode) must remain offline.
    args = SimpleNamespace(adapter=None)
    if online:
        args.mode = "train-online"
    runner.train(cfg, args)

    assert len(case.models) == len(case.loaders) == 1
    loader, model = case.loaders[0], case.models[0]
    assert isinstance(loader.dataset, Subset if online else CachedLiberoGeometryDataset)
    assert loader.dataset.indices == expected
    assert loader.dataset.dataset.load_history is online
    assert loader.batch_size == 2 and loader.num_workers == workers
    assert loader.persistent_workers is bool(workers)
    assert loader.generator.initial_seed() == cfg.seed
    if workers:
        assert loader.multiprocessing_context.get_start_method() == "spawn"
    else:
        assert loader.multiprocessing_context is None

    batches = [expected[j:j + 2] for j in range(0, len(expected), 2)]
    batches = [batches[j % len(batches)] for j in range(cfg.steps)]
    frames = lambda indices: [case.dataset.samples[i][2] for i in indices]
    assert [f for grad, f in case.forwards if grad] == [frames(b) for b in batches]
    assert [f for grad, f in case.forwards if not grad] == [frames([i]) for i in expected[:2]] * 2
    if not workers:
        # Initial/final probes each read two windows; no eager RGB list exists.
        assert case.reads == expected[:2] + [i for b in batches for i in b] + expected[:2]

    out = Path(cfg.output_dir)
    np.testing.assert_array_equal(np.load(out / "train_indices.npy"), expected)
    payload = torch.load(out / "geometry_adapter.pt", map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    source = "online_rgb" if online else "disk"
    assert metadata["training_geometry_source"] == source
    assert metadata["objective"] == "video_only" and metadata["steps"] == cfg.steps
    assert metadata["training_windows"] == len(expected)
    assert metadata["train_index_file"] == "train_indices.npy"
    assert metadata["train_episodes"] == sorted({f"{case.dataset.samples[i][0]}:{case.dataset.samples[i][1]}"
                                                for i in expected})
    assert ("cache_contract_id" in metadata) is (not online)
    assert json.loads(json.dumps(metadata)) == metadata
    assert not any("optimizer" in key for key in payload)
    for key, value in payload["geometry_adapter"].items():
        torch.testing.assert_close(model.mot.state_dict()[key], value, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        if name in case.base:
            assert parameter.grad is None and not parameter.requires_grad
            torch.testing.assert_close(parameter, case.base[name], rtol=0, atol=0)
    for name in ("scene", "camera", "track"):
        assert model.mot.geometry_latent_adapter.attention.branches[name].to_kv.weight.grad.norm() > 0
        assert getattr(model.mot.geometry_tokenizer, name)[1].weight.grad.norm() > 0

    report = json.loads((out / "summary.json").read_text())
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all(row["geometry_source"] == source for row in rows)
    assert report["training_geometry_source"] == source
    assert report["tracker_initialized"] is online
    assert "training excludes probes" in report["extraction_count_unit"]
    assert report["online_extractions_during_training"] == (cfg.steps if online else 0)
    assert report["online_extractions_during_probes"] == (4 if online else 0)
    assert report["online_extractor_calls"] == (cfg.steps + 4 if online else 0)
    assert report["checkpoint_reload_exact"]
    assert all(report["branch_gradients_seen"].values())
    assert all(report["tokenizer_gradients_seen"].values())
    assert ("cache_contract_id" in report) is (not online)
    assert ("cache_contract_id" in rows[0]) is (not online)
    assert ("cache_backend" in rows[0]) is (not online)
    assert ("online RGB" if online else "offline") in report["experiment"]
    assert "track4world.nets.model" not in sys.modules
    if online:
        assert len(case.extractors) == 1
        extractor = case.extractors[0]
        assert extractor.calls == cfg.steps + 4
        assert extractor.batch_sizes == [1, 1] + [len(b) for b in batches] + [1, 1]
        assert not case.root.exists()
    else:
        assert case.extractors == [] and model._geometry_extractor is None


@pytest.mark.parametrize("stage", ["probe", "train"])
@pytest.mark.parametrize("call_delta", [0, 2])
def test_online_rejects_skipped_or_repeated_extraction(training_case, monkeypatch, stage, call_delta):
    case = training_case
    configure_source(case, monkeypatch, True)
    original = BatchFixtureExtractor.__call__

    def bad_count(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if torch.is_grad_enabled() == (stage == "train"):
            self.calls += call_delta - 1
        return result

    monkeypatch.setattr(BatchFixtureExtractor, "__call__", bad_count)
    with pytest.raises(RuntimeError, match="exactly once per forward"):
        runner.train(case.cfg, SimpleNamespace(mode="train-online", adapter=None))
    assert not (Path(case.cfg.output_dir) / "geometry_adapter.pt").exists()


@pytest.mark.parametrize("online", [False, True])
def test_warm_start_keeps_adapter_only_and_positive_sample_selection(training_case, monkeypatch, online):
    case, cfg = training_case, training_case.cfg
    cfg.train_samples = 1
    cfg.steps = 1
    configure_source(case, monkeypatch, online)
    initial = tiny_policy()
    initial.enable_geometry(OmegaConf.to_container(cfg.geometry, resolve=True))
    with torch.no_grad():
        initial.mot.geometry_latent_adapter.gates.fill_(0.125)
    checkpoint = Path(cfg.output_dir).with_name("warm.pt")
    initial.save_geometry_adapter(checkpoint, steps=999)
    original = runner.load_policy

    def load_and_check(config):
        model = original(config)
        forward = model.training_loss

        def check_warm(sample):
            if not case.forwards:
                torch.testing.assert_close(model.mot.geometry_latent_adapter.gates,
                                           initial.mot.geometry_latent_adapter.gates, rtol=0, atol=0)
            return forward(sample)

        monkeypatch.setattr(model, "training_loss", check_warm)
        return model

    monkeypatch.setattr(runner, "load_policy", load_and_check)
    runner.train(cfg, SimpleNamespace(mode="train-online" if online else "train", adapter=str(checkpoint)))
    out = Path(cfg.output_dir)
    payload = torch.load(out / "geometry_adapter.pt", weights_only=True)
    assert payload["metadata"]["steps"] == 1
    np.testing.assert_array_equal(np.load(out / "train_indices.npy"), [2])


@pytest.mark.parametrize("mode", ["train", "train-online"])
def test_main_dispatches_training_overrides_from_resolved_config(training_case, monkeypatch, tmp_path, mode):
    cfg = training_case.cfg
    config = tmp_path / "portable.yaml"
    OmegaConf.save(cfg, config)
    calls = []
    monkeypatch.setattr(runner, "train", lambda resolved, args: calls.append((resolved, args)))
    monkeypatch.chdir(tmp_path)  # No repo-relative config or prior experiment path.
    monkeypatch.setattr(sys, "argv", ["runner", mode, "--config", str(config), "--train-samples", "0",
                                    "--steps", "5", "--batch-size", "3", "--num-workers", "2",
                                    "--output-dir", str(tmp_path / "other"), "--adapter", "warm.pt"])
    runner.main()
    assert len(calls) == 1
    resolved, args = calls[0]
    assert args.mode == mode and args.adapter == "warm.pt"
    assert (resolved.train_samples, resolved.steps, resolved.batch_size, resolved.num_workers) == (0, 5, 3, 2)
    assert resolved.output_dir == str(tmp_path / "other")


@pytest.mark.parametrize("mode", ["train", "train-online"])
@pytest.mark.parametrize("flag", ["--verify-online", "--include-episode-starts"])
def test_training_cli_rejects_cache_diagnostics(training_case, monkeypatch, mode, flag):
    monkeypatch.setattr(runner, "load_config", lambda path: training_case.cfg)
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(sys, "argv", ["runner", mode, flag])
    with pytest.raises(ValueError, match="flags are not allowed during training"):
        runner.main()
