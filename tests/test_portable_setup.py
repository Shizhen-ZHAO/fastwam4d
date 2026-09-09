"""Offline setup safety tests; all mutations are confined to pytest temp dirs."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bootstrap_sources", ROOT / "scripts/portability/bootstrap_sources.py"
)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """Real git-apply fixtures; no cloning, staging, or committing, even in tests."""
    root = tmp_path / "Fast WAM"
    for relative in setup.SOURCES:
        path = root / relative
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    for relative in setup.PATCHES:
        write(root / relative / "setting.py", "value = 'upstream'\n")
        write(root / setup.PATCHES[relative],
              "diff --git a/setting.py b/setting.py\n"
              "--- a/setting.py\n+++ b/setting.py\n@@ -1 +1 @@\n"
              "-value = 'upstream'\n+value = 'portable'\n")
    for relative in ("utils3d/__init__.py", "utils3d/numpy/__init__.py",
                     "utils3d/torch/__init__.py", "Pi3/pi3/__init__.py",
                     "Pi3/pi3/models/pi3.py"):
        write(root / "third_party" / relative, "# pinned source\n")
    monkeypatch.setattr(setup, "check_manifest", lambda root: None)
    monkeypatch.setattr(setup, "check_source", lambda *args: True)
    calls = []
    real_git = setup.git

    def git(directory, *args, **kwargs):
        calls.append(args)
        if "submodule" in args:
            return subprocess.CompletedProcess(args, 0, "", "")
        return real_git(directory, *args, **kwargs)

    monkeypatch.setattr(setup, "git", git)
    return SimpleNamespace(root=root, calls=calls)


def test_bootstrap_idempotent_and_uses_relative_links(sources):
    root = sources.root
    write(root / "third_party/Track4World/user_notes.txt", "keep me\n")
    setup.bootstrap(root)
    before = setup.content_tree(root)
    setup.bootstrap(root)
    setup.bootstrap(root, check=True)
    assert setup.content_tree(root) == before
    for link, target in setup.source_links(root):
        assert link.is_symlink()
        assert not Path(os.readlink(link)).is_absolute()
        assert link.resolve() == target
    for relative in setup.PATCHES:
        assert (root / relative / "setting.py").read_text() == "value = 'portable'\n"
    updates = [args for args in sources.calls if "submodule" in args]
    assert not updates  # Never update already pinned, locally patched worktrees.


def test_normal_initialization_updates_only_the_four_top_level_sources(sources, monkeypatch):
    monkeypatch.setattr(setup, "check_source", lambda *args: any(
        "submodule" in call for call in sources.calls
    ))
    setup.bootstrap(sources.root)
    updates = [args for args in sources.calls if "submodule" in args]
    assert updates == [("-c", "submodule.recurse=false", "submodule", "update", "--init",
                        "--", *setup.SOURCES)]


@pytest.mark.parametrize("staged", [False, True])
def test_dirty_existing_source_prevents_update_of_missing_sources(sources, monkeypatch, staged):
    real_git = setup.git

    def git(directory, *args, **kwargs):
        if args[0] == "diff":
            sources.calls.append(args)
            return subprocess.CompletedProcess(args, int(("--cached" in args) == staged), "", "")
        return real_git(directory, *args, **kwargs)

    monkeypatch.setattr(setup, "git", git)
    monkeypatch.setattr(setup, "check_source", lambda root, path, sha: path != "third_party/Pi3")
    before = setup.content_tree(sources.root)
    with pytest.raises(setup.SetupError, match="Local changes"):
        setup.bootstrap(sources.root)
    assert not any("submodule" in args for args in sources.calls)
    assert setup.content_tree(sources.root) == before


def test_untracked_existing_source_prevents_update(sources, monkeypatch):
    monkeypatch.setattr(setup, "check_source", lambda root, path, sha: path != "third_party/Pi3")
    with pytest.raises(setup.SetupError, match="Untracked files"):
        setup.bootstrap(sources.root)
    assert not any("submodule" in args for args in sources.calls)


def test_check_pending_does_not_clone_patch_or_create_links(sources):
    before = setup.content_tree(sources.root)
    with pytest.raises(setup.SetupError, match="Setup still needed"):
        setup.bootstrap(sources.root, check=True)
    assert setup.content_tree(sources.root) == before
    assert sources.calls
    assert all(args[0] == "apply" and "--check" in args for args in sources.calls)


def test_check_ready_is_read_only(sources):
    setup.bootstrap(sources.root)
    sources.calls.clear()
    before = setup.content_tree(sources.root)
    setup.bootstrap(sources.root, check=True)
    assert setup.content_tree(sources.root) == before
    assert all("--check" in args for args in sources.calls)


def test_check_uninitialized_never_calls_update(sources, monkeypatch):
    monkeypatch.setattr(setup, "check_source", lambda *args: False)
    with pytest.raises(setup.SetupError, match="Uninitialized submodules"):
        setup.bootstrap(sources.root, check=True)
    assert not sources.calls


def test_patch_conflict_is_preflighted_before_any_patch_write(sources):
    write(sources.root / "third_party/LIBERO/setting.py", "user modification\n")
    before = setup.content_tree(sources.root)
    with pytest.raises(setup.SetupError, match="Patch conflict"):
        setup.bootstrap(sources.root)
    assert setup.content_tree(sources.root) == before


def test_wrong_link_is_preserved_and_prevents_patch_writes(sources):
    link = sources.root / "third_party/Track4World/utils3d"
    link.symlink_to("missing-user-directory")
    before = setup.content_tree(sources.root)
    with pytest.raises(setup.SetupError, match="symlink"):
        setup.bootstrap(sources.root)
    assert setup.content_tree(sources.root) == before
    assert os.readlink(link) == "missing-user-directory"


def test_existing_matching_copy_is_kept(sources):
    root = sources.root
    target = root / "third_party/utils3d"
    destination = root / "third_party/Track4World/utils3d"
    shutil.copytree(target, destination, ignore=shutil.ignore_patterns(".git"))
    write(destination / "__pycache__/local.pyc", "cache is not source")
    setup.bootstrap(root)
    assert destination.is_dir() and not destination.is_symlink()
    setup.bootstrap(root, check=True)


def test_different_copy_is_never_overwritten(sources):
    destination = sources.root / "third_party/Track4World/utils3d"
    write(destination / "__init__.py", "user implementation\n")
    before = setup.content_tree(sources.root)
    with pytest.raises(setup.SetupError, match="Existing content differs"):
        setup.bootstrap(sources.root)
    assert setup.content_tree(sources.root) == before


def test_unexpected_utils3d_layout_fails(sources):
    (sources.root / "third_party/utils3d/numpy/__init__.py").unlink()
    with pytest.raises(setup.SetupError, match="module layout"):
        setup.bootstrap(sources.root)


def test_symlinked_parent_fails_without_writing_outside_root(sources, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (sources.root / "third_party/Track4World/track4world").symlink_to(outside)
    with pytest.raises(setup.SetupError, match="symlinked parent"):
        setup.bootstrap(sources.root)
    assert not list(outside.iterdir())


def test_partially_applied_patch_is_a_conflict(tmp_path):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    write(tmp_path / "one.txt", "new\n")
    write(tmp_path / "two.txt", "old\n")
    patch = tmp_path / "both.diff"
    write(patch, "".join(
        f"diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n"
        "@@ -1 +1 @@\n-old\n+new\n" for name in ("one.txt", "two.txt")
    ))
    with pytest.raises(setup.SetupError, match="Patch conflict"):
        setup.patch_state(tmp_path, patch)
    assert (tmp_path / "one.txt").read_text() == "new\n"
    assert (tmp_path / "two.txt").read_text() == "old\n"


@pytest.mark.parametrize("bad_index", ["100644", "wrong_sha", "unmerged"])
def test_manifest_rejects_incorrect_gitlinks(tmp_path, monkeypatch, bad_index):
    write(tmp_path / ".gitmodules", "# fixture\n")

    def fake_git(directory, *args, **kwargs):
        if args[0] == "rev-parse":
            output = str(tmp_path)
        elif args[0] == "config":
            output = "\n".join(f"submodule.{i}.path {p}" for i, p in enumerate(setup.SOURCES))
        else:
            path = args[-1]
            mode = "100644" if bad_index == "100644" else "160000"
            sha = "0" * 40 if bad_index == "wrong_sha" else setup.SOURCES[path]
            stage = "2" if bad_index == "unmerged" else "0"
            output = f"{mode} {sha} {stage}\t{path}"
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(setup, "git", fake_git)
    with pytest.raises(setup.SetupError, match="pinned index gitlink"):
        setup.check_manifest(tmp_path)


def test_existing_wrong_head_is_preserved(tmp_path, monkeypatch):
    relative, revision = next(iter(setup.SOURCES.items()))
    source = tmp_path / relative
    (source / ".git").mkdir(parents=True)
    write(source / "user.txt", "keep\n")
    calls = []

    def fake_git(directory, *args):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0,
                                           str(source) if args[-1] == "--show-toplevel" else "0" * 40, "")

    monkeypatch.setattr(setup, "git", fake_git)
    with pytest.raises(setup.SetupError, match="preserving checkout"):
        setup.check_source(tmp_path, relative, revision)
    assert (source / "user.txt").read_text() == "keep\n"
    assert all(args[0] == "rev-parse" for args in calls)


def test_empty_submodule_directory_is_uninitialized(tmp_path):
    relative, revision = next(iter(setup.SOURCES.items()))
    (tmp_path / relative).mkdir(parents=True)
    assert setup.check_source(tmp_path, relative, revision) is False


def test_uninitialized_existing_files_are_preserved(tmp_path):
    relative, revision = next(iter(setup.SOURCES.items()))
    write(tmp_path / relative / "user.txt", "keep\n")
    with pytest.raises(setup.SetupError, match="existing files"):
        setup.check_source(tmp_path, relative, revision)


@pytest.fixture
def installer(tmp_path):
    script = tmp_path / "Fast WAM/scripts/portability/install_environment.sh"
    write(script, (ROOT / "scripts/portability/install_environment.sh").read_text())
    return script


def shell(script, *args, env=None):
    clean = dict(os.environ)
    for key in ("CONDA_PREFIX", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"):
        clean.pop(key, None)
    clean.update(env or {})
    return subprocess.run(["bash", str(script), *args], env=clean,
                          text=True, capture_output=True)


def test_install_script_syntax():
    result = subprocess.run(["bash", "-n", str(ROOT / "scripts/portability/install_environment.sh")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_git_children_do_not_inherit_proxy_or_alternate_repo(tmp_path, monkeypatch):
    keys = (*setup.PROXY_KEYS, "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")
    for key in keys:
        monkeypatch.setenv(key, "do-not-inherit")
    seen = {}

    def run(args, **kwargs):
        seen.update(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(setup.subprocess, "run", run)
    setup.git(tmp_path, "status")
    assert not set(keys) & seen.keys()
    assert seen["GIT_OPTIONAL_LOCKS"] == "0"


def test_installer_clears_proxy_before_invoking_conda(installer, tmp_path):
    command = tmp_path / "commands/conda"
    write(command, "#!/bin/sh\n"
          'if [ -n "$HTTP_PROXY$HTTPS_PROXY$ALL_PROXY$http_proxy$https_proxy$all_proxy" ]; then\n'
          "  echo proxy-leaked >&2; exit 92\nfi\n"
          "echo proxy-clean >&2\nexit 91\n")
    command.chmod(0o755)
    env = {key: "do-not-inherit" for key in setup.PROXY_KEYS}
    env["PATH"] = str(command.parent) + os.pathsep + os.environ["PATH"]
    result = shell(installer, "--execute", env=env)
    assert result.returncode == 91  # Only the fake conda ran; no environment/pip work.
    assert "proxy-clean" in result.stderr
    assert "proxy-leaked" not in result.stderr


def test_default_plan_has_no_side_effects_or_external_commands(installer, tmp_path):
    commands = tmp_path / "commands"
    for name in ("conda", "python", "python3", "pip", "git"):
        command = commands / name
        write(command, "#!/bin/sh\necho 'unexpected external command' >&2\nexit 91\n")
        command.chmod(0o755)
    before = setup.content_tree(installer.parents[2])
    result = shell(installer, env={"PATH": str(commands) + os.pathsep + os.environ["PATH"]})
    assert result.returncode == 0, result.stderr
    assert "PLAN ONLY" in result.stdout
    assert "--no-deps" in result.stdout
    assert "torch==2.5.1+cu121" in result.stdout
    assert setup.content_tree(installer.parents[2]) == before


def test_new_environment_never_reuses_existing_prefix(installer):
    prefix = installer.parents[2] / ".conda/envs/fastwam4d-cu121"
    prefix.mkdir(parents=True)
    write(prefix / "sentinel", "keep")
    result = shell(installer, "--execute")
    assert result.returncode != 0
    assert "Refusing to reuse existing environment" in result.stderr
    assert (prefix / "sentinel").read_text() == "keep"


@pytest.mark.parametrize("args, message", [
    (("--active-env", "--execute"), "requires an activated"),
    (("--python", sys.executable, "--execute"), "requires --active-env"),
    (("--env-name", "../../outside"), "Invalid environment name"),
    (("--env-name", "test", "--active-env"), "mutually exclusive"),
    (("--unknown",), "Unknown argument"),
])
def test_installer_rejects_unsafe_or_ambiguous_targets(installer, args, message):
    result = shell(installer, *args)
    assert result.returncode != 0
    assert message in result.stderr


def test_active_interpreter_must_match_prefix(installer, tmp_path):
    result = shell(installer, "--active-env", "--python", sys.executable, "--execute",
                   env={"VIRTUAL_ENV": str(tmp_path / "different-env")})
    assert result.returncode != 0
    assert "must belong to the explicitly activated environment" in result.stderr
    assert "pip install" not in result.stdout


def test_active_conda_base_is_rejected(installer):
    if not (Path(sys.prefix) / "conda-meta").is_dir() or sys.version_info[:2] != (3, 10):
        pytest.skip("This check uses the current Python 3.10 conda interpreter")
    result = shell(installer, "--active-env", "--python", sys.executable, "--execute",
                   env={"CONDA_PREFIX": sys.prefix, "CONDA_DEFAULT_ENV": "base"})
    assert result.returncode != 0
    assert "base conda environment" in result.stderr


def test_requirements_pin_python310_stack_and_test_dependencies():
    from packaging.requirements import Requirement

    lines = (ROOT / "requirements/fastwam4d-cu121.txt").read_text().splitlines()
    requirements = {r.name.lower().replace("_", "-"): r for r in
                    (Requirement(line) for line in lines if line and not line.startswith(("#", "--")))}
    assert str(requirements["torch"].specifier) == "==2.5.1+cu121"
    assert str(requirements["torchvision"].specifier) == "==0.20.1+cu121"
    assert str(requirements["scipy"].specifier) == "==1.15.3"
    assert str(requirements["numpy"].specifier) == "==1.26.4"
    assert "4.4.2" in requirements["decorator"].specifier
    assert {"pytest", "h5py", "ftfy", "av", "robosuite", "mujoco", "bddl"} <= requirements.keys()
    assert "torchcodec" not in requirements
    assert all(len(r.specifier) == 1 and str(r.specifier).startswith("==") for r in requirements.values())
