from pathlib import Path
import copy
import json
import os
import shutil

import numpy as np
import pytest
import torch

from fastwam.datasets.geometry_cache import (
    CachedLeRobotGeometryDataset,
    LeRobotGeometryCache,
)
from fastwam.datasets.libero_geometry import LiberoHistoryBuffer


def test_online_history_matches_offline_boundary_and_camera_order():
    buffer = LiberoHistoryBuffer(length=4, stride=1, image_size=4, fps=20)
    for step in range(4):
        external = np.full((4, 4, 3), step, dtype=np.uint8)
        wrist = np.full((4, 4, 3), 100 + step, dtype=np.uint8)
        buffer.append([external, wrist], step)
    result = buffer.inputs()
    assert result["history_images"].shape == (1, 2, 4, 3, 4, 4)
    assert result["history_images"][0, 0, :, 0, 0, 0].mul(255).round().tolist() == [0, 1, 2, 3]
    assert result["history_images"][0, 1, :, 0, 0, 0].mul(255).round().tolist() == [100, 101, 102, 103]
    torch.testing.assert_close(
        result["history_timestamps"], torch.tensor([[-0.15, -0.10, -0.05, 0.0]])
    )
    assert result["history_valid"].tolist() == [[True, True, True, True]]


def test_online_history_replicates_episode_start_but_marks_it_invalid():
    buffer = LiberoHistoryBuffer(length=4, stride=1, image_size=4, fps=20)
    image = np.full((4, 4, 3), 17, dtype=np.uint8)
    buffer.append([image, image], step=0)
    result = buffer.inputs()
    assert result["history_valid"].tolist() == [[False, False, False, True]]
    assert result["history_timestamps"].tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert torch.equal(result["history_images"][:, :, :1], result["history_images"][:, :, -1:])


class _Meta:
    total_episodes = 1
    episodes = {0: {"length": 2}}


class _InnerDataset:
    def __init__(self, root):
        self.root = root
        self.meta = _Meta()


class _MultiDataset:
    def __init__(self, root):
        self._datasets = [_InnerDataset(root)]


class _FakeHistory:
    def __init__(self, root):
        self.dataset_dirs = [str(root)]
        self.camera_keys = ("observation.images.image", "observation.images.wrist_image")
        self.history_length = 3
        self.history_stride = 1
        self.fps = 20.0
        self.image_size = 256
        self.dataset = _MultiDataset(root)

    def __len__(self):
        return 2

    def metadata(self, index):
        index = int(index)
        return {
            "dataset_id": 0,
            "dataset_root": str(Path(self.dataset_dirs[0]).resolve()),
            "local_index": index,
            "episode_index": 0,
            "frame_index": index,
            "history_frame_indices": [max(0, index - 2), max(0, index - 1), index],
            "history_valid": [index >= 2, index >= 1, True],
            "history_timestamps": [-0.05 if index else 0.0, -0.05 if index else 0.0, 0.0],
        }


def _raw(value):
    views, length, points = 2, 3, 4
    floating = lambda *shape: torch.full(shape, float(value), dtype=torch.float32)
    return {
        "scene": floating(views, points, 1024),
        "scene_aux": floating(views, points, 6),
        "scene_valid": torch.ones(views, points, dtype=torch.bool),
        "camera": floating(views, 3072),
        "camera_aux": floating(views, 13),
        "camera_valid": torch.ones(views, dtype=torch.bool),
        "track": floating(views, length, points, 256),
        "track_aux": floating(views, length, points, 11),
        "track_valid": torch.ones(views, length, points, dtype=torch.bool),
    }


def _cache(tmp_path):
    dataset_root = tmp_path / "lerobot"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text('{"fps": 20}')
    (dataset_root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n')
    history = _FakeHistory(dataset_root)
    producer = tmp_path / "producer"
    (producer / "track4world/nets").mkdir(parents=True)
    (producer / "track4world/nets/model.py").write_text("# fixture tracker version 1\n")
    (producer / "weights.pt").write_bytes(b"fixture tracker weights")
    (producer / "da3").mkdir()
    (producer / "da3/config.json").write_text('{}')
    (producer / "da3/model.safetensors").write_bytes(b"fixture DA3 weights")
    geometry = {"extractor": {
        "repo_path": str(producer), "checkpoint_path": str(producer / "weights.pt"),
        "da3_path": str(producer / "da3"), "grid_size": 2, "image_size": 256, "iters": 4,
    }}
    return history, LeRobotGeometryCache(tmp_path / "cache", history, geometry, create=True)


def test_cache_roundtrip_coverage_and_actual_sample_binding(tmp_path):
    history, cache = _cache(tmp_path)
    assert cache.coverage()["written_frames"] == 0
    expected = _raw(2)
    cache.write(1, expected)
    actual = cache.read(1)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])
    assert cache.coverage() == {
        "expected_frames": 2,
        "written_frames": 1,
        "missing_frames": 1,
        "missing_episodes": 0,
    }
    with pytest.raises(FileNotFoundError, match="incomplete"):
        cache.assert_complete()
    with pytest.raises(FileExistsError):
        cache.write(1, expected)

    class Supervision:
        def __len__(self):
            return 2

        def __getitem__(self, requested_index):
            return {"sample_index": 1, "requested_index": requested_index}

    sample = CachedLeRobotGeometryDataset(Supervision(), cache)[0]
    assert sample["sample_index"] == 1
    assert sample["requested_index"] == 0
    torch.testing.assert_close(sample["geometry_raw"]["track"], expected["track"])


def test_cache_and_all_inputs_can_move_without_reextraction(tmp_path):
    history, cache = _cache(tmp_path / "source")
    cache.write(1, _raw(7))
    target = tmp_path / "different_machine"
    shutil.copytree(tmp_path / "source", target)
    geometry = copy.deepcopy(cache.geometry)
    geometry["extractor"].update(
        repo_path=str(target / "producer"),
        checkpoint_path=str(target / "producer/weights.pt"),
        da3_path=str(target / "producer/da3"), device="cuda:15",
    )
    moved = LeRobotGeometryCache(target / "cache", _FakeHistory(target / "lerobot"), geometry)
    assert moved.contract_id == cache.contract_id
    assert moved.entry_path(1).name == cache.entry_path(1).name
    assert str(tmp_path) not in json.dumps(moved.contract)
    for name, value in moved.read(1).items():
        torch.testing.assert_close(value, _raw(7)[name])
    assert moved.coverage() == cache.coverage()


@pytest.mark.parametrize("relative", [
    "producer/weights.pt", "producer/da3/model.safetensors",
    "producer/da3/config.json", "producer/track4world/nets/model.py",
    "lerobot/meta/episodes.jsonl",
])
def test_cache_rejects_changed_contents_at_same_path(tmp_path, relative):
    history, cache = _cache(tmp_path)
    path = tmp_path / relative
    previous = path.stat()
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    # Same length AND preserved mtime must still invalidate in-process hashing.
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    with pytest.raises(ValueError, match="contract mismatch"):
        LeRobotGeometryCache(cache.root, history, cache.geometry)


def test_cache_binds_video_contents_not_just_metadata(tmp_path):
    history, fixture = _cache(tmp_path)
    video = Path(history.dataset_dirs[0]) / "videos/camera/clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"old clip")
    cache = LeRobotGeometryCache(tmp_path / "with_video", history, fixture.geometry, create=True)
    video.write_bytes(b"new clip")
    with pytest.raises(ValueError, match="contract mismatch"):
        LeRobotGeometryCache(cache.root, history, cache.geometry)


def test_cache_rejects_legacy_without_overwriting_it(tmp_path):
    history, cache = _cache(tmp_path)
    manifest_path = cache.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["contract"]["schema"] = 1
    manifest_path.write_text(json.dumps(manifest))
    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="Legacy geometry cache"):
        LeRobotGeometryCache(cache.root, history, cache.geometry, create=True)
    assert manifest_path.read_bytes() == before
