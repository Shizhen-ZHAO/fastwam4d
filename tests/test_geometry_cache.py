"""Cache safety and training/inference routing. Tiny fixtures are unit tests only."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from fastwam.datasets.libero_geometry import LiberoHDF5HistoryDataset
from fastwam.datasets.geometry_cache import GeometryFeatureCache, CachedLiberoGeometryDataset
from fastwam.models.wan22.geometry_features import validate_raw_geometry
from test_geometry_online import raw_features
from test_vae_geometry import tiny_policy, enable, sample, inference_inputs, FixtureExtractor


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "demo.hdf5"
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({kind: {"default": {"global_min": [-1.] * dim, "global_max": [1.] * dim}}
                                for kind, dim in (("action", 7), ("state", 8))}))
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "cache fixture"})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((48, 7), np.float32))
        obs = demo.create_group("obs")
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            obs.create_dataset(key, data=np.broadcast_to(np.arange(48, dtype=np.uint8)[:, None, None, None], (48, 8, 8, 3)))
        for key, dim in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
            obs.create_dataset(key, data=np.zeros((48, dim), np.float32))
    dataset = LiberoHDF5HistoryDataset([source], stats, tmp_path, image_size=16,
                                     history_image_size=8, load_history=False)
    repo = tmp_path / "tracker"
    (repo / "track4world/nets").mkdir(parents=True)
    (repo / "track4world/nets/model.py").write_text("# Unit fixture, never loaded\n")
    weights = repo / "tracker.pt"
    weights.write_bytes(b"unit fixture")
    da3 = repo / "da3"
    da3.mkdir()
    (da3 / "config.json").write_text("{}")
    (da3 / "model.safetensors").write_bytes(b"unit fixture")
    geometry = dict(num_views=2, history_length=8, history_stride=1, history_fps=20,
                    extractor=dict(repo_path=str(repo), checkpoint_path=str(weights), da3_path=str(da3),
                                   image_size=8, grid_size=2, iters=4, device="cuda:1"))
    return dataset, geometry, tmp_path / "cache"


def single_raw():
    return {k: v[0] for k, v in raw_features(b=1, length=8).items()}


def add_context(dataset):
    torch.save({"context": torch.zeros(128, 48), "mask": torch.ones(128, dtype=torch.bool)},
               dataset.context_path(dataset.prompts[0]))


def test_history_only_needs_no_text_and_never_reads_future(fixture):
    dataset, _, _ = fixture
    first = dataset.history_item(0)
    assert first["history_valid"].tolist() == [False] * 7 + [True]
    before = dataset.history_item(2)
    with h5py.File(dataset.files[0], "a") as handle:
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            handle[f"data/demo_0/obs/{key}"][9:] = 255
    after = dataset.history_item(2)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert before["history_timestamps"].max() == 0
    assert not dataset.context_path(dataset.prompts[0]).exists()


def test_lossless_roundtrip_resume_and_disk_dataloader_no_history(fixture, monkeypatch):
    dataset, geometry, root = fixture
    cache = GeometryFeatureCache(root, dataset, geometry, create=True)
    raw = single_raw()
    cache.write(0, raw)
    cache.write(2, raw)
    with pytest.raises(FileExistsError):
        cache.write(0, raw)
    again = GeometryFeatureCache(root, dataset, geometry, create=True)
    for key in raw:
        torch.testing.assert_close(raw[key], again.read(2)[key], rtol=0, atol=0)
    add_context(dataset)
    def no_history(*args):
        raise AssertionError("Cached training read historical RGB")
    monkeypatch.setattr(dataset, "history_item", no_history)
    cached = CachedLiberoGeometryDataset(dataset, again, [0, 2])
    for workers in (0, 2):
        batch = next(iter(DataLoader(cached, batch_size=2, num_workers=workers)))
        assert "history_images" not in batch and "history_valid" not in batch
        assert batch["decision_frame"].tolist() == [0, 8]
        assert batch["geometry_raw"]["track"].shape == (2, 2, 8, 4, 256)
        assert not batch["geometry_raw"]["scene"].requires_grad


def test_missing_cache_and_missing_window_fail_closed(fixture):
    dataset, geometry, root = fixture
    with pytest.raises(FileNotFoundError, match="extract first"):
        GeometryFeatureCache(root, dataset, geometry)
    cache = GeometryFeatureCache(root, dataset, geometry, create=True)
    with pytest.raises(FileNotFoundError, match="no online fallback"):
        cache.read(0)
    with pytest.raises(FileNotFoundError, match="extract first"):
        CachedLiberoGeometryDataset(dataset, cache, [0])


@pytest.mark.parametrize("kind", ["rotate", "grid", "source", "weights", "code"])
def test_stale_cache_rejected(fixture, kind):
    dataset, geometry, root = fixture
    GeometryFeatureCache(root, dataset, geometry, create=True)
    if kind == "rotate":
        dataset.rotate_180 = not dataset.rotate_180
    elif kind == "grid":
        geometry["extractor"]["grid_size"] = 4
    elif kind == "source":
        with h5py.File(dataset.files[0], "a") as handle:
            handle.attrs["changed"] = True
    elif kind == "weights":
        Path(geometry["extractor"]["checkpoint_path"]).write_bytes(b"updated fixture")
    else:
        (Path(geometry["extractor"]["repo_path"]) / "track4world/nets/model.py").write_text("# changed fixture")
    with pytest.raises(ValueError, match="mismatch|changed"):
        GeometryFeatureCache(root, dataset, geometry)


@pytest.mark.parametrize("kind", ["identity", "checksum", "shape", "mask_dtype"])
def test_corrupt_or_swapped_entry_rejected(fixture, kind):
    dataset, geometry, root = fixture
    cache = GeometryFeatureCache(root, dataset, geometry, create=True)
    cache.write(0, single_raw())
    path = cache.entry_path(0)
    payload = torch.load(path, weights_only=True)
    if kind == "identity":
        payload["window"]["decision_frame"] = 8
    elif kind == "checksum":
        payload["raw"]["scene"][0, 0, 0] += 1
    elif kind == "shape":
        payload["raw"]["track"] = payload["raw"]["track"][:, :-1]
    else:
        payload["raw"]["camera_valid"] = payload["raw"]["camera_valid"].float()
    torch.save(payload, path)
    with pytest.raises(ValueError):
        cache.read(0)


def test_cache_identity_independent_of_trainable_tokenizer_and_device(fixture):
    dataset, geometry, root = fixture
    cache = GeometryFeatureCache(root, dataset, geometry, create=True)
    cache.write(0, single_raw())
    geometry.update(memory_dim=256, heads=4, target="vae_latent")
    geometry["extractor"]["device"] = "cuda:0"
    other = GeometryFeatureCache(root, dataset, geometry)
    assert cache.entry_path(0) == other.entry_path(0)
    other.read(0)


def test_cache_read_does_not_require_tracker_assets(fixture):
    dataset, geometry, root = fixture
    cache = GeometryFeatureCache(root, dataset, geometry, create=True)
    cache.write(0, single_raw())
    repo = Path(geometry["extractor"]["repo_path"])
    repo.rename(repo.with_name("tracker_temporarily_unavailable"))
    GeometryFeatureCache(root, dataset, geometry).read(0)


def test_spawn_workers_can_read_cache(fixture):
    dataset, geometry, root = fixture
    cache = GeometryFeatureCache(root, dataset, geometry, create=True)
    cache.write(0, single_raw())
    cache.write(2, single_raw())
    add_context(dataset)
    cached = CachedLiberoGeometryDataset(dataset, cache, [0, 2])
    loader = DataLoader(cached, batch_size=2, num_workers=2,
                        multiprocessing_context="spawn", persistent_workers=True)
    for _ in range(2):
        batch = next(iter(loader))
        assert batch["decision_frame"].tolist() == [0, 8]
        assert batch["geometry_raw"]["scene"].shape == (2, 2, 4, 1024)
        assert "history_images" not in batch


@pytest.mark.parametrize("checkpoint", [False, True])
def test_cached_training_matches_online_and_trains_tokenizer_without_extractor(checkpoint):
    model, rgb = tiny_policy(checkpoint), sample()
    enable(model)
    model.mot.geometry_latent_adapter.gates.data.fill_(0.1)
    extractor = model._geometry_extractor
    raw = {k: v.detach().clone().requires_grad_(v.is_floating_point()) for k, v in extractor.raw.items()}
    disk = {k: v for k, v in rgb.items() if not k.startswith("history_")}
    disk["geometry_raw"] = raw
    torch.manual_seed(987)
    online_loss, _ = model.training_loss(rgb)
    assert extractor.calls == 1
    model._geometry_extractor = None
    torch.manual_seed(987)
    disk_loss, _ = model.training_loss(disk)
    assert model._geometry_extractor is None
    torch.testing.assert_close(disk_loss, online_loss, rtol=0, atol=0)
    disk_loss.backward()
    for name in ("scene", "camera", "track"):
        assert getattr(model.mot.geometry_tokenizer, name)[1].weight.grad.abs().sum() > 0
        assert model.mot.geometry_latent_adapter.attention.branches[name].to_kv.weight.grad.abs().sum() > 0
    assert all(v.grad is None for v in raw.values())
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    with pytest.raises(ValueError, match="Ambiguous"):
        model.build_inputs(dict(rgb, geometry_raw=raw))
    model.eval()
    # eval() does not select the route; disk training probes are still disk-only.
    with torch.no_grad():
        model.training_loss(disk)
    assert model._geometry_extractor is None
    inputs = inference_inputs(rgb)
    inputs.pop("history_images")
    with pytest.raises(ValueError, match="Online geometry requires"):
        model.infer_action(**inputs)
    model._geometry_extractor = FixtureExtractor()
    model.infer_action(**inference_inputs(rgb))
    assert model._geometry_extractor.calls == 1


def test_geometry_disabled_cannot_silently_ignore_cache():
    model, s = tiny_policy(), sample()
    s = {k: v for k, v in s.items() if not k.startswith("history_")}
    s["geometry_raw"] = raw_features(b=1, length=8)
    with pytest.raises(ValueError, match="not enabled"):
        model.build_inputs(s)


def test_raw_schema_rejects_wrong_batch_and_dims():
    model, s = tiny_policy(), sample()
    enable(model)
    s = {k: v for k, v in s.items() if not k.startswith("history_")}
    s["geometry_raw"] = raw_features(b=2, length=8)
    with pytest.raises(ValueError, match="batch size mismatch"):
        model.build_inputs(s)
    with pytest.raises(ValueError, match="length/grid mismatch"):
        validate_raw_geometry(raw_features(), length=8)
