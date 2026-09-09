"""Versioned, per-causal-window frozen features. Never caches trainable tokens.

FP32 tensors + bool masks are stored losslessly. The manifest records preprocessing,
source/weight stat fingerprints and extraction-code SHA256. Tensor SHA256 detects
damaged/swapped payloads. Large weights/data use size+mtime, NOT a content hash.
"""
import hashlib
import copy
import inspect
import json
import os
from pathlib import Path
import tempfile

import torch
from torch.utils.data import Dataset

from fastwam.models.wan22.geometry_features import validate_raw_geometry
from .libero_geometry import CAMERA_KEYS, LiberoHDF5HistoryDataset, resize_rgb


SCHEMA_VERSION = 1


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def extraction_contract(dataset, geometry):
    ext = dict(geometry["extractor"])
    defaults = dict(image_size=256, grid_size=8, iters=4, confidence_threshold=0.25)
    unknown = set(ext) - set(defaults) - {"repo_path", "checkpoint_path", "da3_path", "device"}
    if unknown:
        raise ValueError(f"Unversioned extractor options: {unknown}")
    defaults.update({k: ext[k] for k in defaults if k in ext})
    if (dataset.history_image_size != int(defaults["image_size"])
            or dataset.history_length != int(geometry["history_length"])
            or dataset.history_stride != int(geometry["history_stride"])
            or int(geometry.get("num_views", 2)) != len(CAMERA_KEYS)
            or any(fps != float(geometry["history_fps"]) for fps in dataset.fps.values())):
        raise ValueError("Dataset/inference history configuration mismatch")
    preprocess_code = "\n".join(inspect.getsource(fn) for fn in
                                (resize_rgb, LiberoHDF5HistoryDataset.history_metadata,
                                 LiberoHDF5HistoryDataset.history_item))
    return {"schema": SCHEMA_VERSION, "boundary": "frozen_raw_before_geometry_tokenizer",
            "storage": "float32+bool", "camera_order": list(CAMERA_KEYS),
            "extractor": defaults,
            "weights": {k: str(Path(ext[k]).resolve()) for k in ("checkpoint_path", "da3_path")},
            "history": {"length": dataset.history_length, "stride": dataset.history_stride,
                        "image_size": dataset.history_image_size, "rotate_180": dataset.rotate_180,
                        "fps": dataset.fps, "padding": "replicate_oldest_rgb_but_mask_invalid",
                        "resize": "bilinear_align_corners_false_antialias_true_rgb_0_1",
                        "preprocessing_sha256": hashlib.sha256(preprocess_code.encode()).hexdigest()},
            "sources": [file_fingerprint(p) for p in sorted(dataset.files)]}


def producer_provenance(geometry):
    ext = geometry["extractor"]
    repo = Path(ext["repo_path"]).resolve()
    source_files = sorted((repo / "track4world").rglob("*.py"))
    if not (repo / "track4world/nets/model.py").is_file():
        raise FileNotFoundError("Missing Track4World source for extraction")
    sources = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    wrapper = Path(__file__).parents[1] / "models/wan22/track4world_online.py"
    return {"tracker_source_sha256": digest_json(sources), "tracker_repo": str(repo),
            "extractor_wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "assets": [file_fingerprint(p) for p in
                       (ext["checkpoint_path"], Path(ext["da3_path"]) / "config.json",
                        Path(ext["da3_path"]) / "model.safetensors")],
            "torch": str(torch.__version__), "cuda": torch.version.cuda}


def atomic_create(path, *, payload=None, json_value=None, text_value=None):
    """Same-filesystem atomic publication, never overwrite a completed entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            if text_value is not None:
                handle.write(text_value.encode())
            elif json_value is not None:
                handle.write(json.dumps(json_value, indent=2, sort_keys=True).encode())
            else:
                torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)  # FileExistsError is deliberate; no silent overwrite.
        finally:
            temporary.unlink(missing_ok=True)


def tensor_digest(raw):
    digest = hashlib.sha256()
    for key in sorted(raw):
        value = raw[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class GeometryFeatureCache:
    layout = "window_pt"
    allow_source_subset = False

    def __init__(self, root, dataset, geometry, *, create=False):
        self.root, self.dataset = Path(root), dataset
        self.contract = extraction_contract(dataset, geometry)
        self.contract_id = digest_json(self.contract)
        manifest_path = self.root / "manifest.json"
        if create and not manifest_path.exists():
            manifest = {"contract": self.contract, "contract_id": self.contract_id, "layout": self.layout,
                        "producer": producer_provenance(geometry)}
            try:
                atomic_create(manifest_path, json_value=manifest)
            except FileExistsError:
                pass  # Another extractor published the same manifest; compare below.
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Geometry cache missing: {manifest_path}; run extract first")
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("layout", "window_pt") != self.layout:
            raise ValueError("Geometry cache storage layout mismatch")
        recorded_contract = self.manifest["contract"]
        expected = copy.deepcopy(recorded_contract)
        if self.allow_source_subset:
            selected_paths = set(dataset.files)
            expected["sources"] = [x for x in expected["sources"] if x["path"] in selected_paths]
            expected["history"]["fps"] = {k: v for k, v in expected["history"]["fps"].items() if k in selected_paths}
        if expected != self.contract or self.manifest["contract_id"] != digest_json(recorded_contract):
            raise ValueError("Geometry cache contract mismatch (data/weights/history/preprocessing); use a new cache directory")
        # Packed caches permit selecting a subset of their source files for
        # training, but retain the full producer identity for stored windows.
        self.contract, self.contract_id = recorded_contract, self.manifest["contract_id"]
        # Training never loads tracker weights. If its assets are present, detect
        # in-place updates. A cache-only training host may omit the tracker assets.
        producer = self.manifest["producer"]
        for recorded in producer["assets"]:
            if Path(recorded["path"]).exists() and file_fingerprint(recorded["path"]) != recorded:
                raise ValueError("Geometry cache producer weights changed; re-extract")
        repo = Path(geometry["extractor"]["repo_path"])
        if (repo / "track4world/nets/model.py").exists():
            source_files = sorted((repo / "track4world").rglob("*.py"))
            source_hash = digest_json({str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in source_files})
            if source_hash != producer["tracker_source_sha256"]:
                raise ValueError("Geometry cache Track4World source changed; re-extract")
        wrapper = Path(__file__).parents[1] / "models/wan22/track4world_online.py"
        if hashlib.sha256(wrapper.read_bytes()).hexdigest() != producer["extractor_wrapper_sha256"]:
            raise ValueError("Geometry cache extractor wrapper changed; re-extract")
        if create and producer_provenance(geometry) != producer:
            raise ValueError("Cannot mix extraction producers/environments in one cache")

    def entry_path(self, index):
        key = digest_json({"contract": self.contract_id, "window": self.dataset.history_metadata(index)})
        return self.root / "windows" / key[:2] / f"{key}.pt"

    def contains(self, index):
        return self.entry_path(index).is_file()

    def check_coverage(self, indices):
        for index in indices:
            if not self.contains(index):
                raise FileNotFoundError(f"Missing geometry window {self.dataset.samples[index]}; run extract first")

    def _validate(self, raw):
        validate_raw_geometry(raw, batched=False, views=len(CAMERA_KEYS),
                              length=self.dataset.history_length,
                              points=int(self.contract["extractor"]["grid_size"]) ** 2, fp32=True)
        if any(v.device.type != "cpu" or v.requires_grad for v in raw.values()):
            raise ValueError("Cached features must be detached CPU tensors")

    def write(self, index, raw):
        """raw is a single window WITHOUT a batch dimension."""
        raw = {k: v.detach().cpu().contiguous().clone() for k, v in raw.items()}
        self._validate(raw)
        atomic_create(self.entry_path(index), payload={"contract_id": self.contract_id,
                      "window": self.dataset.history_metadata(index), "raw": raw, "sha256": tensor_digest(raw)})

    def read(self, index):
        path = self.entry_path(index)
        if not path.is_file():
            raise FileNotFoundError(f"Missing geometry window {self.dataset.samples[index]}: {path}; no online fallback")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (payload["contract_id"] != self.contract_id
                or payload["window"] != self.dataset.history_metadata(index)):
            raise ValueError(f"Geometry cache window identity mismatch: {path}")
        self._validate(payload["raw"])
        if tensor_digest(payload["raw"]) != payload["sha256"]:
            raise ValueError(f"Geometry cache tensor checksum mismatch: {path}")
        return payload["raw"]


class CachedLiberoGeometryDataset(Dataset):
    """Disk-backed supervision + raw features; never reads historical RGB."""
    def __init__(self, dataset, cache, indices=None):
        if dataset.load_history or dataset.history_only:
            raise ValueError("Cached training requires load_history=False")
        if dataset is not cache.dataset:
            raise ValueError("Cache and supervision must share the same dataset")
        self.dataset, self.cache = dataset, cache
        self.indices = list(range(len(dataset))) if indices is None else list(indices)
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("Empty or duplicate cached dataset indices")
        if any(not isinstance(i, int) or not 0 <= i < len(dataset) for i in self.indices):
            raise ValueError("Cached dataset index out of range")
        cache.check_coverage(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = self.indices[index]
        raw = self.cache.read(original)  # Fail before expensive supervision IO.
        sample = self.dataset[original]
        sample["geometry_raw"] = raw
        return sample
