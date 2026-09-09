"""Lossless per-episode cache, indexed by original decision frame (not row rank).

Each window has a commit bit, identity SHA256 and tensor SHA256. Readers never
accept an incomplete slot. Files are opened/closed per operation, so DataLoader
spawn workers inherit no HDF5 handles. POSIX shared/exclusive locks protect IO.
Normal interruption is resumable; HDF5 is NOT power-loss transactional storage.
"""
from collections import defaultdict
from contextlib import contextmanager
import fcntl

import h5py
import numpy as np
import torch

from .geometry_cache import GeometryFeatureCache, digest_json, tensor_digest


class HDF5GeometryFeatureCache(GeometryFeatureCache):
    layout = "episode_hdf5_v1"
    allow_source_subset = True

    def entry_path(self, index):
        source, episode, _ = self.dataset.samples[index]
        key = digest_json({"source": source, "episode": episode})
        return self.root / "episodes" / key[:2] / f"{key}.h5"

    @contextmanager
    def _open(self, index, *, write=False):
        path = self.entry_path(index)
        if not write and not path.is_file():
            raise FileNotFoundError(f"Missing geometry episode {path}; no online fallback")
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
        # Reader uses an existing lock file in read-only mode; a read-only cache
        # mount is supported. A writer creates the lock before the HDF5 file.
        with path.with_suffix(".lock").open("a+b" if write else "rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            try:
                with h5py.File(path, "a" if write else "r") as handle:
                    yield handle
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _check_episode(self, handle, index):
        source, episode, _ = self.dataset.samples[index]
        if (handle.attrs.get("contract_id") != self.contract_id
                or handle.attrs.get("source") != source or handle.attrs.get("episode") != episode):
            raise ValueError("Geometry episode identity/contract mismatch")
        if handle["written"].shape != (self.dataset.episode_lengths[(source, episode)],):
            raise ValueError("Geometry episode length mismatch")

    def _initialize(self, handle, index, raw):
        source, episode, _ = self.dataset.samples[index]
        n = self.dataset.episode_lengths[(source, episode)]
        # Idempotent initialization also handles a normal interruption before
        # the first committed slot. Corrupt HDF5 containers fail closed.
        if "contract_id" in handle.attrs:
            if (handle.attrs["contract_id"] != self.contract_id
                    or handle.attrs.get("source") != source or handle.attrs.get("episode") != episode):
                raise ValueError("Geometry episode identity/contract mismatch")
        else:
            handle.attrs.update(contract_id=self.contract_id, source=source, episode=episode)
        handle.require_dataset("written", (n,), dtype=np.uint8, exact=True)
        handle.require_dataset("window_sha256", (n,), dtype="S64", exact=True)
        handle.require_dataset("tensor_sha256", (n,), dtype="S64", exact=True)
        group = handle.require_group("raw")
        for key, tensor in raw.items():
            shape = (n,) + tuple(tensor.shape)
            dtype = np.bool_ if tensor.dtype == torch.bool else np.float32
            group.require_dataset(key, shape, dtype=dtype, exact=True,
                                  chunks=(1,) + tuple(tensor.shape), compression="lzf",
                                  shuffle=True, fletcher32=True)

    def write(self, index, raw):
        raw = {k: v.detach().cpu().contiguous().clone() for k, v in raw.items()}
        self._validate(raw)
        t = self.dataset.samples[index][2]
        with self._open(index, write=True) as handle:
            self._initialize(handle, index, raw)
            self._check_episode(handle, index)
            if handle["written"][t]:
                raise FileExistsError(f"Refusing to overwrite committed geometry: {self.dataset.samples[index]}")
            for key, value in raw.items():
                handle["raw"][key][t] = value.numpy()
            handle["window_sha256"][t] = digest_json(self.dataset.history_metadata(index)).encode()
            handle["tensor_sha256"][t] = tensor_digest(raw).encode()
            handle.flush()
            handle["written"][t] = 1  # Commit last, after all tensors and checksums.
            handle.flush()

    def read(self, index):
        t = self.dataset.samples[index][2]
        with self._open(index) as handle:
            self._check_episode(handle, index)
            if handle["written"][t] != 1:
                raise FileNotFoundError(f"Missing geometry window {self.dataset.samples[index]}; no online fallback")
            if handle["window_sha256"][t].decode() != digest_json(self.dataset.history_metadata(index)):
                raise ValueError("Geometry cache window identity mismatch")
            raw = {key: torch.from_numpy(value[t]) for key, value in handle["raw"].items()}
            checksum = handle["tensor_sha256"][t].decode()
        self._validate(raw)
        if tensor_digest(raw) != checksum:
            raise ValueError("Geometry cache tensor checksum mismatch")
        return raw

    def completed_indices(self, indices):
        """Read one small commit/identity table per episode; no tensor scan."""
        grouped = defaultdict(list)
        for index in indices:
            grouped[self.dataset.samples[index][:2]].append(index)
        completed = set()
        for members in grouped.values():
            if not self.entry_path(members[0]).is_file():
                continue
            with self._open(members[0]) as handle:
                # An interrupted initialization with no commit table is empty.
                if "written" not in handle:
                    continue
                self._check_episode(handle, members[0])
                written = handle["written"][:]
                identities = handle["window_sha256"][:] if "window_sha256" in handle else None
            if not np.isin(written, [0, 1]).all():
                raise ValueError("Invalid geometry commit table")
            for index in members:
                t = self.dataset.samples[index][2]
                if written[t] == 1:
                    if identities is None or identities[t].decode() != digest_json(self.dataset.history_metadata(index)):
                        raise ValueError("Geometry cache window identity mismatch")
                    completed.add(index)
        return completed

    def contains(self, index):
        return index in self.completed_indices([index])

    def check_coverage(self, indices):
        missing = set(indices) - self.completed_indices(indices)
        if missing:
            example = [self.dataset.samples[i] for i in sorted(missing)[:3]]
            raise FileNotFoundError(f"Missing {len(missing)} geometry windows, e.g. {example}; run extract first; no online fallback")


def open_geometry_cache(cfg, dataset, *, create=False):
    from omegaconf import OmegaConf
    geometry = OmegaConf.to_container(cfg.geometry, resolve=True)
    backend = cfg.get("cache_backend", "pt")
    if backend not in {"pt", "hdf5"}:
        raise ValueError(f"Unknown geometry cache backend: {backend}")
    cls = HDF5GeometryFeatureCache if backend == "hdf5" else GeometryFeatureCache
    return cls(cfg.cache_dir, dataset, geometry, create=create)
