"""CPU-only downloader tests: no real Hub requests, model imports, or assets."""

import builtins
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/portability/download_assets.py"
SPEC = importlib.util.spec_from_file_location("portability_download_assets", SCRIPT)
assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets
SPEC.loader.exec_module(assets)


def roots(tmp_path):
    return ["--assets-root", str(tmp_path / "models"), "--data-root", str(tmp_path / "data")]


def test_exact_model_files_and_native_loader_layout(tmp_path):
    plan = assets.build_plan(tmp_path / "models", tmp_path / "data")
    expected = {
        "yuanty/fastwam": ("fastwam", {
            "libero_uncond_2cam224.pt", "libero_uncond_2cam224_dataset_stats.json",
        }),
        "TencentARC/Track4World": ("track4world", {"track4world_da3.pth"}),
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1": (
            "DA3NESTED-GIANT-LARGE-1.1", {"config.json", "model.safetensors"},
        ),
        "Wan-AI/Wan2.2-TI2V-5B": (
            "model_base/Wan-AI/Wan2.2-TI2V-5B",
            {"Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"},
        ),
        "Wan-AI/Wan2.1-T2V-1.3B": (
            "model_base/Wan-AI/Wan2.1-T2V-1.3B", {
                "google/umt5-xxl/special_tokens_map.json", "google/umt5-xxl/spiece.model",
                "google/umt5-xxl/tokenizer.json", "google/umt5-xxl/tokenizer_config.json",
            },
        ),
    }
    assert {entry["repo_id"] for entry in plan["assets"]} == set(expected)
    assert sum(len(entry["files"]) for entry in plan["assets"]) == 11
    assert "DiffSynth" not in json.dumps(plan)
    assert "UNVERIFIED" not in json.dumps(plan)
    for entry in plan["assets"]:
        subdir, files = expected[entry["repo_id"]]
        local_dir = tmp_path / "models" / subdir
        assert entry["local_dir"] == str(local_dir)
        assert set(entry["files"]) == files
        assert set(entry["destinations"]) == {str(local_dir / name) for name in files}
        assert entry["method"] == "hf_hub_download"
        assert entry["repo_type"] == "model"
        assert entry["revision_pinned"] is True
        assert len(entry["revision"]) == 40
        if entry["name"] == "vae_t5":
            assert entry["revision"] == "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
            assert "redirect_common_files=False" in entry["verification"]
    assert list(tmp_path.iterdir()) == []


def test_libero_official_pinned_hdf5_only(tmp_path):
    plan = assets.build_plan(tmp_path / "models", tmp_path / "data", data=True)
    entry, = plan["assets"]
    assert entry["repo_id"] == "yifengzhu-hf/LIBERO-datasets"
    assert entry["revision"] == "f13aa24a3da8c43c7225569f28c562979fa0e35a"
    assert entry["revision_pinned"] is True
    assert entry["repo_type"] == "dataset"
    assert entry["local_dir"] == str(tmp_path / "data")
    assert entry["suite_tasks"] == {
        "libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
        "libero_90": 90, "libero_10": 10,
    }
    assert entry["expected_files"] == sum(entry["suite_tasks"].values()) == 130
    assert set(entry["allow_patterns"]) == {
        "libero_spatial/*_demo.hdf5", "libero_object/*_demo.hdf5", "libero_goal/*_demo.hdf5",
        "libero_90/*_demo.hdf5", "libero_10/*_demo.hdf5",
    }
    assert entry["method"] == "snapshot_download"
    assert "LIBERO-fastwam" not in json.dumps(plan)
    assert ".zip" not in json.dumps(plan)


@pytest.mark.parametrize("flags,names", [
    ([], {"fastwam", "track4world", "da3", "vae_t5", "tokenizer"}),
    (["--models"], {"fastwam", "track4world", "da3", "vae_t5", "tokenizer"}),
    (["--data"], {"libero"}),
    (["--models", "--data"], {"fastwam", "track4world", "da3", "vae_t5", "tokenizer", "libero"}),
    (["--data", "--models", "--data"], {"fastwam", "track4world", "da3", "vae_t5", "tokenizer", "libero"}),
])
def test_dry_run_has_no_network_import_spawn_mkdir_or_env_changes(tmp_path, monkeypatch, capsys, flags, names):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run attempted network, child process, or filesystem write")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in {"huggingface_hub", "torch", "transformers"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    monkeypatch.setenv("HF_TOKEN", "hf_TEST_SECRET_MUST_NOT_APPEAR")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    previous = dict(os.environ)
    assert assets.main([*roots(tmp_path), *flags]) == 0
    output = capsys.readouterr()
    plan = json.loads(output.out)
    assert plan["dry_run"] is True
    assert plan["endpoint"] == "https://hf-mirror.com"
    assert {entry["name"] for entry in plan["assets"]} == names
    assert "hf_TEST_SECRET" not in output.out + output.err
    assert dict(os.environ) == previous
    assert list(tmp_path.iterdir()) == []


def test_default_roots_independent_of_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert assets.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["assets_root"] == str(SCRIPT.parents[2] / "assets_local")
    assert plan["data_root"] == str(SCRIPT.parents[2] / "datasets/libero")


def test_child_environment_is_a_copy_and_clears_case_variants():
    parent = {
        key: "sentinel" for key in (
            "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy",
            "Http_Proxy", "HF_HUB_OFFLINE", "hf_hub_offline", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE", "HF_DEBUG",
        )
    }
    parent.update(HF_TOKEN="hf_private", HF_ENDPOINT="https://huggingface.co", OTHER="keep")
    before = dict(parent)
    child = assets.download_environment(parent, "https://hf-mirror.com/")
    assert parent == before
    assert not (set(key.lower() for key in child) & assets._REMOVED_ENV)
    assert child["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert child["HF_TOKEN"] == "hf_private"
    assert child["OTHER"] == "keep"
    assert child["HF_HUB_VERBOSITY"] == "error"
    assert child["HF_HUB_DISABLE_XET"] == "1"


@pytest.mark.parametrize("returncode", [0, 7])
def test_execute_spawns_with_private_env_and_suppresses_library_output(tmp_path, monkeypatch, capsys, returncode):
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET")
    monkeypatch.setenv("HTTPS_PROXY", "http://secret-proxy.invalid")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    previous = dict(os.environ)
    runner = Mock(return_value=SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(subprocess, "run", runner)
    assert assets.main([*roots(tmp_path), "--models", "--data", "--execute"]) == int(bool(returncode))
    command, = runner.call_args.args
    kwargs = runner.call_args.kwargs
    assert command[:2] == [sys.executable, str(SCRIPT)]
    assert all(flag in command for flag in ("--execute", "--_download-worker", "--models", "--data"))
    assert kwargs["env"]["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert kwargs["env"]["HF_TOKEN"] == "hf_SECRET"
    assert "HF_HUB_OFFLINE" not in kwargs["env"]
    assert "HTTPS_PROXY" not in kwargs["env"]
    assert kwargs["stdout"] == kwargs["stderr"] == kwargs["stdin"] == subprocess.DEVNULL
    assert dict(os.environ) == previous
    output = capsys.readouterr()
    assert not json.loads(output.out)["dry_run"]
    assert "hf_SECRET" not in output.out + output.err + repr(command)
    if returncode:
        assert "resume" in output.err


def test_spawn_errors_never_echo_exception_secrets(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=OSError("hf_EXCEPTION_SECRET")))
    assert assets.main([*roots(tmp_path), "--execute"]) == 1
    captured = capsys.readouterr()
    assert "hf_EXCEPTION_SECRET" not in captured.out + captured.err


@pytest.fixture
def mock_hub(monkeypatch):
    env = assets.download_environment(os.environ, assets.DEFAULT_ENDPOINT)
    monkeypatch.setattr(os, "environ", env)
    module = ModuleType("huggingface_hub")
    module.hf_hub_download = Mock()
    module.snapshot_download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    original_import = builtins.__import__

    def check_import(name, *args, **kwargs):
        if name == "huggingface_hub":
            assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
            assert not (set(key.lower() for key in os.environ) & assets._REMOVED_ENV)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", check_import)
    return module


def test_only_allowlisted_models_and_hub_handles_resume(tmp_path, mock_hub):
    plan = assets.build_plan(tmp_path / "models", tmp_path / "data")
    # Presence alone must not cause a skip: the Hub validates cached metadata and
    # owns partial-download recovery. No forced restart or home-grown copying.
    existing = tmp_path / "models/fastwam/libero_uncond_2cam224.pt"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"existing fixture")
    assets._download(plan)
    assets._download(plan)
    assert mock_hub.hf_hub_download.call_count == 22
    mock_hub.snapshot_download.assert_not_called()
    expected = {(entry["repo_id"], name) for entry in plan["assets"] for name in entry["files"]}
    assert {(call.kwargs["repo_id"], call.kwargs["filename"]) for call in mock_hub.hf_hub_download.call_args_list} == expected
    for call in mock_hub.hf_hub_download.call_args_list:
        assert call.kwargs["endpoint"] == "https://hf-mirror.com"
        assert call.kwargs["local_files_only"] is False
        assert call.kwargs["revision"]
        assert not ({"force_download", "resume_download", "token"} & call.kwargs.keys())
    wan_calls = [
        call.kwargs for call in mock_hub.hf_hub_download.call_args_list
        if call.kwargs["repo_id"] == "Wan-AI/Wan2.2-TI2V-5B"
    ]
    assert len(wan_calls) == 4  # Two native files, requested on both resume attempts.
    assert {call["filename"] for call in wan_calls} == {
        "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth",
    }
    assert all(call["revision"] == "921dbaf3f1674a56f47e83fb80a34bac8a8f203e" for call in wan_calls)
    assert all(call["local_dir"] == str(tmp_path / "models/model_base/Wan-AI/Wan2.2-TI2V-5B") for call in wan_calls)
    assert existing.read_bytes() == b"existing fixture"


def test_snapshot_uses_only_five_hdf5_patterns_and_retains_existing_files(tmp_path, mock_hub):
    plan = assets.build_plan(tmp_path / "models", tmp_path / "data", data=True)
    data_root = tmp_path / "data"

    def snapshot(**kwargs):
        for suite, count in assets.LIBERO_SUITE_TASKS.items():
            directory = data_root / suite
            directory.mkdir(parents=True, exist_ok=True)
            for task in range(count):
                (directory / f"task_{task}_demo.hdf5").touch(exist_ok=True)

    mock_hub.snapshot_download.side_effect = snapshot
    existing = data_root / "libero_10/task_0_demo.hdf5"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"preserved fixture")
    assets._download(plan)
    mock_hub.hf_hub_download.assert_not_called()
    call = mock_hub.snapshot_download.call_args.kwargs
    assert call["repo_type"] == "dataset"
    assert call["revision"] == assets.LIBERO_REVISION
    assert call["endpoint"] == "https://hf-mirror.com"
    assert call["local_dir"] == str(data_root)
    assert call["allow_patterns"] == list(assets.LIBERO_ALLOW_PATTERNS)
    assert existing.read_bytes() == b"preserved fixture"


def test_incomplete_libero_snapshot_is_not_reported_as_success(tmp_path, mock_hub):
    plan = assets.build_plan(tmp_path / "models", tmp_path / "data", data=True)
    with pytest.raises(RuntimeError, match="suite file count"):
        assets._download(plan)


def test_worker_rejects_unsanitized_env_before_import(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="child environment"):
        assets._download(assets.build_plan(tmp_path / "models", tmp_path / "data"))


@pytest.mark.parametrize("endpoint", [
    "http://hf-mirror.com", "https://user:hf_SECRET@hf-mirror.com", "https://hf-mirror.com?token=hf_SECRET",
    "https://hf-mirror.com/#hf_SECRET", "https://hf-mirror.com/path", "https://", "https://hf-mirror.com:bad",
])
def test_invalid_endpoint_is_rejected_without_echoing_credentials(endpoint, tmp_path, capsys):
    assert assets.main([*roots(tmp_path), "--endpoint", endpoint]) == 2
    output = capsys.readouterr()
    assert "hf_SECRET" not in output.out + output.err
    assert output.out == ""


@pytest.mark.parametrize("path", ["", " ", "/", "https://example.com/assets", "bad\x00path"])
def test_invalid_root_rejected(path):
    with pytest.raises((ValueError, OSError)):
        assets.validate_root(path)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "C:/outside", "a\\b", "a/../b", "a//b", "a/./b"])
def test_relative_paths_cannot_traverse(path):
    with pytest.raises(ValueError, match="unsafe"):
        assets._relative_path(path)


@pytest.mark.parametrize("collision", ["root_file", "parent_file", "asset_directory", "symlink", "cache_symlink", "data_symlink"])
def test_path_collisions_and_symlinks_fail_before_download(tmp_path, monkeypatch, capsys, collision):
    root = tmp_path / "models"
    root.mkdir()
    if collision == "root_file":
        root = tmp_path / "file"
        root.touch()
    elif collision == "parent_file":
        (root / "parent").touch()
        root = root / "parent/child"
    elif collision == "asset_directory":
        (root / "fastwam/libero_uncond_2cam224.pt").mkdir(parents=True)
    elif collision == "symlink":
        (root / "fastwam").symlink_to(tmp_path / "outside", target_is_directory=True)
    elif collision == "cache_symlink":
        cache = root / "fastwam/.cache/huggingface/download"
        cache.mkdir(parents=True)
        (cache / "libero_uncond_2cam224.pt.metadata").symlink_to(tmp_path / "outside")
    else:
        (tmp_path / "data").mkdir()
        (tmp_path / "data/libero_90").symlink_to(tmp_path / "outside", target_is_directory=True)
    runner = Mock()
    monkeypatch.setattr(subprocess, "run", runner)
    assert assets.main(["--assets-root", str(root), "--data-root", str(tmp_path / "data"), "--models", "--data", "--execute"]) == 2
    runner.assert_not_called()
    assert capsys.readouterr().out == ""


def test_unsupported_token_argument_and_worker_flag_cannot_leak_or_download(tmp_path, capsys):
    for flags in (["--token", "hf_ARG_SECRET"], ["--_download-worker"]):
        with pytest.raises(SystemExit) as caught:
            assets.main([*roots(tmp_path), *flags])
        assert caught.value.code == 2
        output = capsys.readouterr()
        assert "hf_ARG_SECRET" not in output.out + output.err


@pytest.mark.parametrize("fail", [False, True])
def test_real_child_import_environment_and_token_output_isolation(tmp_path, monkeypatch, fail):
    """Exercise an actual fresh interpreter; the only Hub library is this stub."""
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "huggingface_hub.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert os.environ['HF_ENDPOINT'] == 'https://hf-mirror.com'\n"
        "assert not any(k.lower() in {'http_proxy', 'https_proxy', 'all_proxy', 'hf_hub_offline', 'hf_debug'} for k in os.environ)\n"
        "Path(os.environ['FAKE_IMPORT_MARKER']).touch()\n"
        "def hf_hub_download(**kwargs):\n"
        "    raise AssertionError('data-only should never request a model')\n"
        "def snapshot_download(**kwargs):\n"
        "    assert kwargs['endpoint'] == 'https://hf-mirror.com'\n"
        "    assert kwargs['revision'] == 'f13aa24a3da8c43c7225569f28c562979fa0e35a'\n"
        "    assert len(kwargs['allow_patterns']) == 5\n"
        "    print(os.environ['HF_TOKEN'], flush=True)\n"
        "    print(os.environ['HF_TOKEN'], file=sys.stderr, flush=True)\n"
        "    if os.environ['FAKE_FAIL'] == '1':\n"
        "        raise RuntimeError(os.environ['HF_TOKEN'])\n"
        "    for suite, count in [('libero_10',10), ('libero_90',90), ('libero_spatial',10), ('libero_goal',10), ('libero_object',10)]:\n"
        "        directory = Path(kwargs['local_dir']) / suite\n"
        "        directory.mkdir(parents=True, exist_ok=True)\n"
        "        for i in range(count):\n"
        "            (directory / (str(i) + '_demo.hdf5')).touch()\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(stub_dir))
    monkeypatch.setenv("HF_TOKEN", "hf_SUBPROCESS_SECRET")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DEBUG", "1")
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://unreachable.invalid:1")
    marker = tmp_path / "imported"
    monkeypatch.setenv("FAKE_IMPORT_MARKER", str(marker))
    monkeypatch.setenv("FAKE_FAIL", str(int(fail)))
    before = dict(os.environ)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *roots(tmp_path), "--data", "--execute"],
        capture_output=True, text=True, timeout=20, check=False,
    )
    assert result.returncode == int(fail), result.stderr
    assert marker.exists(), "the fake Hub must actually be imported in the child"
    assert "hf_SUBPROCESS_SECRET" not in result.stdout + result.stderr
    assert dict(os.environ) == before
    if not fail:
        assert len(list((tmp_path / "data").glob("*/*_demo.hdf5"))) == 130
