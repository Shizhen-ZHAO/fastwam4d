"""Packed cache + direct-training + online-only inference integration tests."""
import copy
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, default_collate

from fastwam.datasets.geometry_cache import CachedLiberoGeometryDataset, GeometryFeatureCache
from fastwam.datasets.geometry_cache_hdf5 import HDF5GeometryFeatureCache
from fastwam.datasets.libero_geometry import LiberoHDF5HistoryDataset, LiberoHistoryBuffer
from test_geometry_cache import fixture, single_raw, add_context
from test_vae_geometry import tiny_policy, enable, sample, inference_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_libero_geometry_full import discover_files, selected_work, SUITE_TASKS, run_extraction


def clone_dataset(dataset, **kwargs):
    args = dict(files=dataset.files, stats_path=dataset.stats_path, text_cache_dir=dataset.text_cache_dir,
                image_size=dataset.image_size, history_image_size=dataset.history_image_size,
                history_length=dataset.history_length, history_stride=dataset.history_stride,
                load_history=False)
    args.update(kwargs)
    return LiberoHDF5HistoryDataset(**args)


def test_hdf5_lossless_commit_and_missing_slot(fixture):
    dataset, geometry, root = fixture
    cache = HDF5GeometryFeatureCache(root, dataset, geometry, create=True)
    raw = single_raw()
    cache.write(0, raw)
    for key in raw:
        torch.testing.assert_close(cache.read(0)[key], raw[key], rtol=0, atol=0)
    assert cache.entry_path(0) == cache.entry_path(1)  # Same file != same slot!
    assert cache.contains(0) and not cache.contains(1)
    with pytest.raises(FileExistsError):
        cache.write(0, raw)
    with pytest.raises(FileNotFoundError, match="no online fallback"):
        cache.read(1)
    with pytest.raises(FileNotFoundError, match="Missing 1"):
        CachedLiberoGeometryDataset(dataset, cache, [0, 1])
    with h5py.File(cache.entry_path(0), "r+") as handle:
        handle["written"][0] = 0  # Simulated interruption BEFORE commit.
        handle["raw/scene"][0] = 99
    assert not cache.contains(0)
    with pytest.raises(FileNotFoundError):
        cache.read(0)
    cache.write(0, raw)
    torch.testing.assert_close(cache.read(0)["scene"], raw["scene"], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["tensor", "identity", "episode", "commit"])
def test_hdf5_corruption_rejected(fixture, kind):
    dataset, geometry, root = fixture
    cache = HDF5GeometryFeatureCache(root, dataset, geometry, create=True)
    cache.write(2, single_raw())
    with h5py.File(cache.entry_path(2), "r+") as handle:
        if kind == "tensor":
            handle["raw/scene"][8, 0, 0, 0] += 1
        elif kind == "identity":
            handle["window_sha256"][8] = b"incorrect-frame"
        elif kind == "episode":
            handle.attrs["episode"] = "demo_99"
        else:
            handle["written"][8] = 2
    with pytest.raises((ValueError, FileNotFoundError)):
        cache.read(2)


def test_backend_mismatch_rejected(fixture):
    dataset, geometry, root = fixture
    HDF5GeometryFeatureCache(root, dataset, geometry, create=True)
    with pytest.raises(ValueError, match="layout"):
        GeometryFeatureCache(root, dataset, geometry)


def test_full_cache_can_train_on_source_subset_and_different_decision_stride(fixture):
    dataset, geometry, root = fixture
    second = Path(dataset.files[0]).with_name("second_demo.hdf5")
    shutil.copy2(dataset.files[0], second)
    all_data = clone_dataset(dataset, files=[*dataset.files, second], history_only=True, sample_stride=1)
    cache = HDF5GeometryFeatureCache(root, all_data, geometry, create=True)
    for index, (source, _, t) in enumerate(all_data.samples):
        if t in (0, 8, 47):
            raw = single_raw()
            raw["scene"].fill_(t + (100 if source == str(second) else 0))
            cache.write(index, raw)
    # Train on ONLY the second file, every fourth frame, no tail supervision.
    training = clone_dataset(dataset, files=[second], sample_stride=4)
    training_cache = HDF5GeometryFeatureCache(root, training, geometry)
    add_context(training)
    batch = default_collate([CachedLiberoGeometryDataset(training, training_cache, [2, 0])[i] for i in range(2)])
    assert batch["decision_frame"].tolist() == [8, 0]
    assert batch["geometry_raw"]["scene"][:, 0, 0, 0].tolist() == [108, 100]
    assert "history_images" not in batch
    assert max(t for _, _, t in training.samples) == 12
    # A stored tail window is still accessible through the extraction dataset.
    cache.read(next(i for i, (p, _, t) in enumerate(all_data.samples) if p == str(second) and t == 47))


def test_full_cache_rejects_unregistered_source_and_history_change(fixture):
    dataset, geometry, root = fixture
    HDF5GeometryFeatureCache(root, dataset, geometry, create=True)
    changed = clone_dataset(dataset, history_stride=2)
    geo = copy.deepcopy(geometry)
    geo["history_stride"] = 2
    with pytest.raises(ValueError, match="contract mismatch"):
        HDF5GeometryFeatureCache(root, changed, geo)
    other = Path(dataset.files[0]).with_name("unregistered.hdf5")
    shutil.copy2(dataset.files[0], other)
    with pytest.raises(ValueError, match="contract mismatch"):
        HDF5GeometryFeatureCache(root, clone_dataset(dataset, files=[other]), geometry)


def test_spawn_shuffle_preserves_frame_feature_alignment(fixture):
    dataset, geometry, root = fixture
    add_context(dataset)
    cache = HDF5GeometryFeatureCache(root, dataset, geometry, create=True)
    for index, (_, _, frame) in enumerate(dataset.samples):
        raw = single_raw()
        raw["scene"].fill_(frame)
        cache.write(index, raw)
    loader = DataLoader(CachedLiberoGeometryDataset(dataset, cache), batch_size=2,
                        num_workers=2, multiprocessing_context="spawn", persistent_workers=True,
                        shuffle=True, generator=torch.Generator().manual_seed(11))
    for _ in range(2):
        seen = []
        for batch in loader:
            assert "history_images" not in batch
            assert batch["geometry_raw"]["scene"][:, 0, 0, 0].tolist() == batch["decision_frame"].tolist()
            seen.extend(batch["decision_frame"].tolist())
        assert sorted(seen) == [0, 4, 8, 12]


@pytest.mark.parametrize("stride", [1, 2])
def test_hdf5_history_matches_live_buffer_orientation_views_padding_and_tails(fixture, stride):
    dataset, _, _ = fixture
    with h5py.File(dataset.files[0], "r+") as handle:
        yy, xx = np.mgrid[:8, :8]
        for view, key in enumerate(("agentview_rgb", "eye_in_hand_rgb")):
            frames = np.stack([np.stack([yy + t, xx + view * 80, yy + xx], -1) for t in range(48)]).astype(np.uint8)
            handle[f"data/demo_0/obs/{key}"][:] = frames
    extraction = clone_dataset(dataset, history_only=True, sample_stride=1, history_stride=stride)
    live = LiberoHistoryBuffer(length=8, stride=stride, image_size=8, fps=20)
    with h5py.File(dataset.files[0], "r") as handle:
        for t in range(48):
            # The actual LIBERO get_libero_image returns orientation-corrected
            # images. Buffer itself must NOT rotate them a second time.
            rgb = [handle[f"data/demo_0/obs/{key}"][t][::-1, ::-1].copy()
                   for key in ("agentview_rgb", "eye_in_hand_rgb")]
            live.append(rgb, t)
            if t in (0, 1, 7, 14, 31, 47):
                cached_input = default_collate([extraction.history_item(t)])
                for key, tensor in live.inputs().items():
                    torch.testing.assert_close(cached_input[key], tensor, rtol=0, atol=0)
    live.reset()
    live.append(rgb, 0)
    assert live.inputs()["history_valid"].tolist() == [[False] * 7 + [True]]


def test_packed_batch_runs_real_model_autograd_without_tracker(fixture):
    dataset, geometry, root = fixture
    add_context(dataset)
    cache = HDF5GeometryFeatureCache(root, dataset, geometry, create=True)
    for i in (0, 2):
        cache.write(i, single_raw())
    batch = next(iter(DataLoader(CachedLiberoGeometryDataset(dataset, cache, [2, 0]), batch_size=2)))
    model = tiny_policy(checkpoint=True)
    enable(model)
    model._geometry_extractor = None
    model.mot.geometry_latent_adapter.gates.data.fill_(0.1)
    loss, _ = model.training_loss(batch)
    loss.backward()
    assert torch.isfinite(loss) and model._geometry_extractor is None
    for bank in ("scene", "camera", "track"):
        assert getattr(model.mot.geometry_tokenizer, bank)[1].weight.grad.norm() > 0
        assert model.mot.geometry_latent_adapter.attention.branches[bank].to_kv.weight.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_both_inference_apis_forbid_disk_cache_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Online inference accessed a disk geometry cache")
    monkeypatch.setattr(GeometryFeatureCache, "read", forbidden)
    monkeypatch.setattr(GeometryFeatureCache, "__init__", forbidden)
    monkeypatch.setattr(HDF5GeometryFeatureCache, "read", forbidden)
    model, inputs = tiny_policy(), sample()
    enable(model)
    model.eval()
    model.mot.geometry_latent_adapter.gates.data.fill_(0.1)
    for api in (model.infer_joint, model.infer_action):
        before = model._geometry_extractor.calls
        kwargs = inference_inputs(inputs)
        if api == model.infer_joint:
            kwargs["num_video_frames"] = 9
        prediction = api(**kwargs)
        assert model._geometry_extractor.calls == before + 1
        assert torch.isfinite(prediction["action"]).all()


def test_full_suite_discovery_never_silently_omits_90(tmp_path):
    for suite, count in SUITE_TASKS.items():
        directory = tmp_path / suite
        directory.mkdir()
        for i in range(count):
            (directory / f"task_{i:02d}_demo.hdf5").touch()
    assert len(discover_files(tmp_path, list(SUITE_TASKS))) == 130
    assert len(discover_files(tmp_path, list(SUITE_TASKS), smoke=True)) == 5
    (tmp_path / "libero_90/task_89_demo.hdf5").rename(tmp_path / "omitted.hdf5")
    with pytest.raises(ValueError, match="Incomplete libero_90"):
        discover_files(tmp_path, list(SUITE_TASKS))


def test_shards_disjoint_and_full_tail_coverage(fixture):
    dataset, _, _ = fixture
    extraction = clone_dataset(dataset, history_only=True, sample_stride=1)
    parts = [set(selected_work(extraction, i, 3)) for i in range(3)]
    assert set.union(*parts) == set(range(48))
    assert not any(parts[i] & parts[j] for i in range(3) for j in range(i))
    assert selected_work(extraction, 0, 1, smoke=True) == [0, 7, 47]


def test_extraction_loop_resume_never_initializes_tracker(fixture):
    dataset, geometry, root = fixture
    extraction = clone_dataset(dataset, history_only=True, sample_stride=1)
    cache = HDF5GeometryFeatureCache(root, extraction, geometry, create=True)
    cache.extraction_options = geometry["extractor"]
    class UnitExtractor:
        def __init__(self, **kwargs):
            self.calls, self.last_seconds = 0, 0.001
        def __call__(self, images, timestamps, valid):
            self.calls += 1
            assert timestamps.max() == 0 and images.shape == (1, 2, 8, 3, 8, 8)
            return {k: v[None] for k, v in single_raw().items()}
        def close(self):
            pass
    result = run_extraction(cache, [0, 7, 47], device="cpu", min_free_gib=0, extractor_factory=UnitExtractor)
    assert result["created"] == 3 and result["extractor_calls"] == 3
    def forbidden(**kwargs):
        raise AssertionError("Resume initialized tracker")
    result = run_extraction(cache, [0, 7, 47], device="cpu", min_free_gib=0,
                            verify_existing=True, extractor_factory=forbidden)
    assert result["resumed"] == 3 and not result["tracker_initialized"]
