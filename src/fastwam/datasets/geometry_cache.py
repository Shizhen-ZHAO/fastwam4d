"""Lossless, per-episode Track4World cache for LeRobot LIBERO windows."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from fastwam.models.wan22.geometry_features import validate_raw_geometry
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


SCHEMA_VERSION = 1


def _digest_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(raw: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(raw):
        value = raw[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _contract(history, geometry) -> dict:
    extractor = dict(geometry["extractor"])
    extractor.pop("device", None)
    roots = []
    for root in history.dataset_dirs:
        root_path = Path(root)
        roots.append(
            {
                "path": str(root_path.resolve()),
                "info_sha256": _file_digest(root_path / "meta" / "info.json"),
                "episodes_sha256": _file_digest(root_path / "meta" / "episodes.jsonl"),
            }
        )
    return {
        "schema": SCHEMA_VERSION,
        "boundary": "frozen_track4world_raw_before_trainable_tokenizer",
        "storage": "episode_hdf5_float32_bool",
        "dataset_roots": roots,
        "camera_keys": list(history.camera_keys),
        "history_length": history.history_length,
        "history_stride": history.history_stride,
        "history_fps": history.fps,
        "image_size": history.image_size,
        "orientation": "lerobot_rgb_as_stored",
        "padding": "replicate_oldest_and_mask_invalid",
        "resize": "bilinear_align_corners_false_antialias_true",
        "extractor": extractor,
    }


class LeRobotGeometryCache:
    """HDF5 cache keyed by dataset root, episode and original frame index."""

    def __init__(self, root, history, geometry, *, create: bool = False):
        self.root = Path(root).expanduser().resolve()
        self.history = history
        self.geometry = dict(geometry)
        self.contract = _contract(history, geometry)
        self.contract_id = _digest_json(self.contract)
        manifest_path = self.root / "manifest.json"
        if create and not manifest_path.exists():
            manifest = {"contract": self.contract, "contract_id": self.contract_id}
            try:
                _atomic_json(manifest_path, manifest)
            except FileExistsError:
                pass
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing geometry cache manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest != {"contract": self.contract, "contract_id": self.contract_id}:
            raise ValueError("Geometry cache contract mismatch; use a new cache directory")

    def _validate(self, raw, *, batched: bool = False):
        points = int(self.geometry["extractor"].get("grid_size", 8)) ** 2
        validate_raw_geometry(
            raw,
            batched=batched,
            views=2,
            length=self.history.history_length,
            points=points,
            fp32=True,
        )

    def _identity(self, index: int) -> dict:
        return self.history.metadata(int(index))

    def _episode_length(self, identity: dict) -> int:
        dataset = self.history.dataset._datasets[identity["dataset_id"]]
        return int(dataset.meta.episodes[identity["episode_index"]]["length"])

    def entry_path(self, index: int) -> Path:
        identity = self._identity(index)
        episode = {
            "dataset_root": identity["dataset_root"],
            "episode_index": identity["episode_index"],
        }
        key = _digest_json(episode)
        return self.root / "episodes" / key[:2] / f"{key}.h5"

    @contextmanager
    def _open(self, index: int, *, write: bool = False):
        path = self.entry_path(index)
        if not write and not path.is_file():
            raise FileNotFoundError(f"Missing geometry episode: {path}")
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        if not write and not lock_path.is_file():
            raise FileNotFoundError(f"Missing geometry lock: {lock_path}")
        with lock_path.open("a+b" if write else "rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            try:
                with h5py.File(path, "a" if write else "r") as handle:
                    yield handle
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _initialize(self, handle, index: int, raw):
        identity = self._identity(index)
        episode_length = self._episode_length(identity)
        expected = {
            "contract_id": self.contract_id,
            "dataset_root": identity["dataset_root"],
            "episode_index": identity["episode_index"],
        }
        if "contract_id" in handle.attrs:
            if any(handle.attrs.get(key) != value for key, value in expected.items()):
                raise ValueError("Geometry episode identity mismatch")
        else:
            handle.attrs.update(expected)
        handle.require_dataset("written", (episode_length,), dtype=np.uint8, exact=True)
        handle.require_dataset("window_sha256", (episode_length,), dtype="S64", exact=True)
        handle.require_dataset("tensor_sha256", (episode_length,), dtype="S64", exact=True)
        group = handle.require_group("raw")
        for key, tensor in raw.items():
            shape = (episode_length,) + tuple(tensor.shape)
            dtype = np.bool_ if tensor.dtype == torch.bool else np.float32
            group.require_dataset(
                key,
                shape,
                dtype=dtype,
                exact=True,
                chunks=(1,) + tuple(tensor.shape),
                compression="lzf",
                shuffle=True,
                fletcher32=True,
            )

    def write(self, index: int, raw):
        raw = {key: value.detach().cpu().contiguous().clone() for key, value in raw.items()}
        self._validate(raw)
        identity = self._identity(index)
        frame = identity["frame_index"]
        with self._open(index, write=True) as handle:
            self._initialize(handle, index, raw)
            if handle["written"][frame] == 1:
                raise FileExistsError(f"Geometry already committed for index {index}")
            for key, value in raw.items():
                handle["raw"][key][frame] = value.numpy()
            handle["window_sha256"][frame] = _digest_json(identity).encode()
            handle["tensor_sha256"][frame] = _tensor_digest(raw).encode()
            handle.flush()
            handle["written"][frame] = 1
            handle.flush()

    def read(self, index: int):
        identity = self._identity(index)
        frame = identity["frame_index"]
        with self._open(index) as handle:
            if handle.attrs.get("contract_id") != self.contract_id:
                raise ValueError("Geometry cache contract mismatch")
            if handle["written"][frame] != 1:
                raise FileNotFoundError(f"Missing geometry window for index {index}")
            if handle["window_sha256"][frame].decode() != _digest_json(identity):
                raise ValueError("Geometry cache window identity mismatch")
            raw = {key: torch.from_numpy(value[frame]) for key, value in handle["raw"].items()}
            checksum = handle["tensor_sha256"][frame].decode()
        self._validate(raw)
        if _tensor_digest(raw) != checksum:
            raise ValueError("Geometry cache tensor checksum mismatch")
        return raw

    def contains(self, index: int) -> bool:
        identity = self._identity(index)
        frame = identity["frame_index"]
        try:
            with self._open(index) as handle:
                return (
                    handle.attrs.get("contract_id") == self.contract_id
                    and "written" in handle
                    and int(handle["written"][frame]) == 1
                )
        except FileNotFoundError:
            return False

    def coverage(self) -> dict[str, int]:
        expected = written = missing_episodes = 0
        for dataset in self.history.dataset._datasets:
            dataset_root = str(Path(dataset.root).resolve())
            for episode_index in range(int(dataset.meta.total_episodes)):
                episode_length = int(dataset.meta.episodes[episode_index]["length"])
                expected += episode_length
                key = _digest_json(
                    {"dataset_root": dataset_root, "episode_index": episode_index}
                )
                path = self.root / "episodes" / key[:2] / f"{key}.h5"
                if not path.is_file():
                    missing_episodes += 1
                    continue
                lock_path = path.with_suffix(".lock")
                if not lock_path.is_file():
                    missing_episodes += 1
                    continue
                with lock_path.open("rb") as lock:
                    fcntl.flock(lock, fcntl.LOCK_SH)
                    try:
                        with h5py.File(path, "r") as handle:
                            if handle.attrs.get("contract_id") != self.contract_id:
                                raise ValueError(f"Geometry cache contract mismatch in {path}")
                            if handle["written"].shape != (episode_length,):
                                raise ValueError(f"Geometry cache episode length mismatch in {path}")
                            written += int(np.asarray(handle["written"], dtype=np.uint8).sum())
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)
        return {
            "expected_frames": expected,
            "written_frames": written,
            "missing_frames": expected - written,
            "missing_episodes": missing_episodes,
        }

    def assert_complete(self):
        coverage = self.coverage()
        if coverage["missing_frames"]:
            raise FileNotFoundError(
                "Offline geometry cache is incomplete: "
                f"{coverage}. Finish feature extraction before training."
            )
        return coverage


class CachedLeRobotGeometryDataset(Dataset):
    """Official FastWAM supervision plus correctly indexed disk geometry."""

    def __init__(self, supervision, cache: LeRobotGeometryCache):
        if len(supervision) != len(cache.history):
            raise ValueError("Supervision/history dataset lengths differ")
        self.supervision = supervision
        self.cache = cache

    def __len__(self):
        return len(self.supervision)

    def __getitem__(self, index):
        sample = self.supervision[index]
        actual_index = int(sample["sample_index"])
        sample["geometry_raw"] = self.cache.read(actual_index)
        return sample


class OnlineLeRobotGeometryDataset(Dataset):
    """Official FastWAM supervision plus online RGB history for smoke training."""

    def __init__(self, supervision, history):
        if len(supervision) != len(history):
            raise ValueError("Supervision/history dataset lengths differ")
        self.supervision = supervision
        self.history = history

    def __len__(self):
        return len(self.supervision)

    def __getitem__(self, index):
        sample = self.supervision[index]
        actual_index = int(sample["sample_index"])
        history = self.history[actual_index]
        if int(history["sample_index"]) != actual_index:
            raise ValueError("History/sample index mismatch")
        sample.update(history)
        return sample


class GeometryRobotVideoDataset(RobotVideoDataset):
    """Drop-in official FastWAM LeRobot dataset with geometry attached.

    ``offline`` reads frozen Track4World tensors from the lossless cache.
    ``online`` decodes the matching RGB history and lets the model extract
    Track4World tensors during the forward pass.
    """

    def __init__(
        self,
        *args,
        geometry_mode: str,
        geometry_cache_root=None,
        geometry_config=None,
        geometry_history_length: int = 8,
        geometry_history_stride: int = 1,
        geometry_image_size: int = 256,
        geometry_require_complete_cache: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from fastwam.datasets.libero_geometry import LeRobotGeometryHistoryDataset

        self.geometry_mode = str(geometry_mode)
        if self.geometry_mode not in {"offline", "online"}:
            raise ValueError("geometry_mode must be 'offline' or 'online'")
        self.geometry_history = LeRobotGeometryHistoryDataset(
            dataset_dirs=kwargs["dataset_dirs"],
            history_length=geometry_history_length,
            history_stride=geometry_history_stride,
            image_size=geometry_image_size,
        )
        if len(self) != len(self.geometry_history):
            raise ValueError("FastWAM supervision and geometry history lengths differ")
        self.geometry_cache = None
        if self.geometry_mode == "offline":
            if geometry_cache_root is None or geometry_config is None:
                raise ValueError("Offline geometry requires cache root and geometry config")
            self.geometry_cache = LeRobotGeometryCache(
                geometry_cache_root,
                self.geometry_history,
                geometry_config,
                create=False,
            )
            if geometry_require_complete_cache:
                self.geometry_cache.assert_complete()

    def _get(self, idx):
        sample = super()._get(idx)
        actual_index = int(sample["sample_index"])
        if self.geometry_mode == "offline":
            sample["geometry_raw"] = self.geometry_cache.read(actual_index)
        else:
            history = self.geometry_history[actual_index]
            if int(history.pop("sample_index")) != actual_index:
                raise ValueError("History/sample index mismatch")
            sample.update(history)
        return sample

    def __getitem__(self, idx):
        # The official dataset retries an arbitrary random index after decoder
        # errors. That policy is unsafe for a partially written geometry cache:
        # missing entries must stop training rather than silently change data.
        return self._get(int(idx))
