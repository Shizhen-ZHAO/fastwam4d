#!/usr/bin/env python3
"""Initialize the four pinned sources and prepare local imports, without weights.

``python scripts/portability/bootstrap_sources.py --check`` is strictly read-only
and exits nonzero when initialization, patching, or links are still needed.
Existing checkouts must already have the pinned HEAD: this script never resets,
forces a checkout, cleans a worktree, or recursively initializes Open-d4rt.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys


SOURCES = {
    "third_party/Track4World": "fbd59ffadf2de9fccba5ea017e13af239075fbe9",
    "third_party/LIBERO": "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    "third_party/utils3d": "2072c024c73f7c0f83e0da23eef5f2d9ac575249",
    "third_party/Pi3": "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b",
}
PATCHES = {
    "third_party/Track4World": "patches/track4world-local-da3.diff",
    "third_party/LIBERO": "patches/libero-torch-load.diff",
}
PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


class SetupError(RuntimeError):
    """A missing prerequisite or user content that must be preserved."""


def git(directory: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    # Do not inherit a caller's alternate repository/index/worktree.
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", *PROXY_KEYS):
        env.pop(key, None)
    result = subprocess.run(
        ["git", "-C", str(directory), *args], env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise SetupError(f"git {' '.join(args)} in {directory}:\n{result.stderr.strip()}")
    return result


def check_manifest(root: Path) -> None:
    if Path(git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve() != root:
        raise SetupError(f"Expected the FastWAM Git root at {root}")
    manifest = root / ".gitmodules"
    if not manifest.is_file():
        raise SetupError("Missing .gitmodules; use the published FastWAM Git checkout.")
    config = git(root, "config", "--file", str(manifest), "--get-regexp",
                 r"^submodule\..*\.path$").stdout
    paths = [line.split(None, 1)[1] for line in config.splitlines()]
    for relative, revision in SOURCES.items():
        if paths.count(relative) != 1:
            raise SetupError(f".gitmodules must declare {relative} exactly once")
        expected = f"160000 {revision} 0\t{relative}"
        if git(root, "ls-files", "--stage", "--", relative).stdout.strip() != expected:
            raise SetupError(f"Expected pinned index gitlink: {expected}")
    for relative in PATCHES.values():
        if not (root / relative).is_file():
            raise SetupError(f"Missing required patch: {relative}")


def check_parents(path: Path, root: Path) -> None:
    for parent in path.relative_to(root).parents:
        candidate = root / parent
        if candidate.is_symlink():
            raise SetupError(f"Refusing to write through symlinked parent: {candidate}")


def check_source(root: Path, relative: str, revision: str) -> bool:
    path = root / relative
    check_parents(path, root)
    if path.is_symlink():
        raise SetupError(f"Expected a submodule directory, not a symlink: {path}")
    if not (path / ".git").exists():
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise SetupError(f"Uninitialized source contains existing files: {path}")
        return False
    if Path(git(path, "rev-parse", "--show-toplevel").stdout.strip()).resolve() != path:
        raise SetupError(f"Not an independent submodule checkout: {path}")
    actual = git(path, "rev-parse", "HEAD").stdout.strip()
    if actual != revision:
        raise SetupError(f"{relative}: expected HEAD {revision}, found {actual}; "
                         "preserving checkout and local changes. Resolve manually.")
    return True


def patch_state(source: Path, patch: Path) -> str:
    # A reverse check recognizes a complete previous application, including
    # unrelated local modifications outside the patch's context.
    if git(source, "apply", "--reverse", "--check", str(patch), check=False).returncode == 0:
        return "applied"
    forward = git(source, "apply", "--check", str(patch), check=False)
    if forward.returncode == 0:
        return "pending"
    raise SetupError(f"Patch conflict in {source}: {patch.name}\n{forward.stderr.strip()}")


def check_clean_before_update(source: Path) -> None:
    """Never run submodule update over staged, unstaged, or untracked user work."""
    for args in (("diff", "--quiet", "--ignore-submodules=none", "--"),
                 ("diff", "--cached", "--quiet", "--ignore-submodules=none", "--")):
        if git(source, *args, check=False).returncode:
            raise SetupError(f"Local changes in {source}; refusing submodule update. "
                             "Preserve your work and initialize missing sources manually.")
    if git(source, "ls-files", "--others", "--exclude-standard").stdout.strip():
        raise SetupError(f"Untracked files in {source}; refusing submodule update.")


def content_tree(directory: Path) -> dict:
    """Compare source copies without following links or considering Git/cache data."""
    result = {}
    for base, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "__pycache__"})
        for name in dirs + sorted(files):
            if name == ".git" or name.endswith((".pyc", ".pyo")):
                continue
            path = Path(base) / name
            key = str(path.relative_to(directory))
            if path.is_symlink():
                result[key] = ("link", os.readlink(path))
            elif path.is_dir():
                result[key] = ("directory",)
            elif path.is_file():
                result[key] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
            else:
                raise SetupError(f"Unsupported existing source content: {path}")
    return result


def source_links(root: Path) -> list[tuple[Path, Path]]:
    utils = root / "third_party/utils3d"
    pi3 = root / "third_party/Pi3/pi3"
    # This utils3d commit is itself the package, with no nested utils3d/ directory
    # and no setup.py/pyproject.toml. Do not guess a conventional src layout.
    for path in (utils / "__init__.py", utils / "numpy/__init__.py",
                 utils / "torch/__init__.py", pi3 / "__init__.py",
                 pi3 / "models/pi3.py"):
        if not path.is_file():
            raise SetupError(f"Pinned source has an unexpected module layout: {path}")
    tracker = root / "third_party/Track4World"
    return [(tracker / "utils3d", utils),
            (tracker / "track4world/nets/external/pi3", pi3)]


def link_state(link: Path, target: Path, root: Path) -> str:
    check_parents(link, root)
    if link.is_symlink():
        if not Path(os.readlink(link)).is_absolute() and link.resolve() == target.resolve():
            return "ready"
        raise SetupError(f"Conflicting or nonportable symlink: {link} -> {os.readlink(link)}")
    if link.exists():
        if link.is_dir() and content_tree(link) == content_tree(target):
            return "ready"
        raise SetupError(f"Existing content differs from pinned source; preserving {link}")
    return "pending"


def bootstrap(root: Path, check: bool = False) -> None:
    root = root.resolve()
    check_manifest(root)
    missing = [path for path, sha in SOURCES.items() if not check_source(root, path, sha)]
    if check and missing:
        raise SetupError("Uninitialized submodules: " + ", ".join(missing))
    if not check and missing:
        for relative in SOURCES:
            if relative not in missing:
                check_clean_before_update(root / relative)
        # No --recursive: Track4World's optional Open-d4rt is intentionally omitted.
        git(root, "-c", "submodule.recurse=false", "submodule", "update", "--init",
            "--", *SOURCES)
        for path, sha in SOURCES.items():
            if not check_source(root, path, sha):
                raise SetupError(f"Submodule update did not initialize {path}")
    # Already initialized/pinned trees need no Git update. This makes repeated
    # runs work with our applied patches and preserves unrelated user changes.
    # Preflight every patch and destination before modifying any of them.
    patches = [(root / path, root / patch, patch_state(root / path, root / patch))
               for path, patch in PATCHES.items()]
    links = [(link, target, link_state(link, target, root))
             for link, target in source_links(root)]
    pending = [str(patch.relative_to(root)) for _, patch, state in patches if state == "pending"]
    pending += [str(link.relative_to(root)) for link, _, state in links if state == "pending"]
    if check and pending:
        raise SetupError("Setup still needed: " + ", ".join(pending))
    for source, patch, state in patches:
        if state == "pending":
            git(source, "apply", "--check", str(patch))
            git(source, "apply", str(patch))
        print(f"Patch ready: {patch.name}")
    for link, target, state in links:
        if state == "pending":
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)
        print(f"Import path ready: {link.relative_to(root)}")
    print("All four pinned sources are ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="read-only readiness check; no network")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="FastWAM Git root (defaults to this script's repository)")
    args = parser.parse_args(argv)
    try:
        bootstrap(args.root, args.check)
    except (SetupError, OSError, ValueError) as exc:
        print(f"Source setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
