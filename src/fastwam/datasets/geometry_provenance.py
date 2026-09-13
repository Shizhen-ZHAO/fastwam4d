"""Content identities: never include machine paths in portable cache keys."""

import hashlib
import json
import os
from pathlib import Path


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _stat_identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _hash_file(path, stat_identity):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if _stat_identity(Path(path)) != stat_identity:
        raise RuntimeError(f"File changed while fingerprinting: {path}")
    return digest.hexdigest()


def file_digest(path):
    path = Path(path).expanduser().resolve()
    # Rehash on each cache open. Filesystems can coalesce timestamps, even ctime;
    # a stat-keyed memo can silently accept different weights with the same size.
    # The stat tuple is only a best-effort concurrent-mutation check.
    return _hash_file(str(path), _stat_identity(path))


def tree_digest(root, *, suffix=None):
    root = Path(root)
    if not root.is_dir():
        return digest_json([])
    entries = []
    for directory, dirs, files in os.walk(root, followlinks=True):
        dirs[:] = sorted(name for name in dirs if name not in {".git", "__pycache__", ".cache"})
        for name in sorted(files):
            path = Path(directory) / name
            if suffix is None or path.suffix == suffix:
                entries.append((path.relative_to(root).as_posix(), file_digest(path)))
    return digest_json(sorted(entries))


def dataset_identity(root):
    root = Path(root)
    # Required metadata plus all parquet/video bytes: copied data is portable,
    # but replacing a clip while retaining its filename cannot reuse stale raw.
    for name in ("info.json", "episodes.jsonl"):
        if not (root / "meta" / name).is_file():
            raise FileNotFoundError(root / "meta" / name)
    content = {name: tree_digest(root / name) for name in ("meta", "data", "videos")}
    return {"dataset_uid": digest_json(content), "content": content}


def extractor_identity(settings):
    settings = dict(settings)
    settings.pop("device", None)
    repo = Path(settings.pop("repo_path")).expanduser().resolve()
    checkpoint = Path(settings.pop("checkpoint_path")).expanduser().resolve()
    da3 = Path(settings.pop("da3_path")).expanduser().resolve()
    if not (repo / "track4world/nets/model.py").is_file():
        raise FileNotFoundError(repo / "track4world/nets/model.py")
    package = Path(__file__).resolve().parents[1]
    producer_files = (
        "models/wan22/track4world_online.py",
        "models/wan22/track4world_compat.py",
        "datasets/libero_geometry.py",
    )
    return {
        "settings": settings,
        "checkpoint_sha256": file_digest(checkpoint),
        "da3_config_sha256": file_digest(da3 / "config.json"),
        "da3_weights_sha256": file_digest(da3 / "model.safetensors"),
        "track4world_source_sha256": tree_digest(repo / "track4world", suffix=".py"),
        "utils3d_source_sha256": tree_digest(repo / "utils3d", suffix=".py"),
        "producer_source_sha256": {name: file_digest(package / name) for name in producer_files},
        "precision": "network_fp16_geometry_fp32",
    }
