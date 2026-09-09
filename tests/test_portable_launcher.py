"""CPU-only migration tests. No download, GPU, or full data is required."""
import argparse
import importlib.util
from pathlib import Path

from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fastwam4d_entry", ROOT / "scripts/fastwam4d.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def config(root):
    cfg = OmegaConf.load(ROOT / "configs/experiments/libero_portable.yaml")
    return entry.resolve_config(cfg, root)


def args(**changes):
    values = dict(adapter=None, output_dir=None, steps=None, batch_size=None, num_workers=None,
                  train_samples=None, inference_steps=None, cache_dir=None,
                  suite="libero_spatial", task_id=0, num_trials=1)
    values.update(changes)
    return argparse.Namespace(**values)


def test_template_resolves_on_another_machine(tmp_path):
    cfg = config(tmp_path)
    text = OmegaConf.to_yaml(cfg)
    assert "/mnt/homes/zhaoshizhen" not in text
    assert cfg.geometry.extractor.repo_path == str(tmp_path / "third_party/Track4World")
    assert cfg.base_checkpoint == str(tmp_path / "assets_local/fastwam/libero_uncond_2cam224.pt")
    assert cfg.geometry.extractor.device == "cuda:0"
    assert cfg.train_samples == 0 and cfg.train_indices is None
    assert set(cfg.data.episode_ids).isdisjoint(cfg.validation_episode_ids)
    assert len(cfg.portable_suites) == 5


def test_environment_does_not_inherit_proxy_or_old_paths(tmp_path):
    cfg = config(tmp_path)
    env = entry.child_environment(cfg, tmp_path, {k: "http://must-not-be-used" for k in entry.PROXY_KEYS})
    assert all(k not in env for k in entry.PROXY_KEYS)
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["DIFFSYNTH_MODEL_BASE_PATH"] == cfg.model_base_path
    assert str(tmp_path / "src") in env["PYTHONPATH"]
    assert str(tmp_path / "third_party") in env["PYTHONPATH"].split(entry.os.pathsep)
    assert cfg.paths.utils3d_repo not in env["PYTHONPATH"].split(entry.os.pathsep)
    assert env["LIBERO_CONFIG_PATH"] == str(tmp_path / "configs/local/libero_runtime")
    assert not (tmp_path / "configs").exists()


@pytest.mark.parametrize("mode,runner", [("train-offline", "train"), ("train-online", "train-online"), ("test-online", "test")])
def test_training_and_testing_modes_use_shared_runner(tmp_path, mode, runner):
    cfg = config(tmp_path)
    options = args(adapter="adapter.pt", output_dir="outputs/new", steps=10)
    cmd = entry.build_command(mode, cfg, tmp_path / "local.yaml", options, [], tmp_path)
    assert Path(cmd[1]).name == "experiment_vae_geometry_cached.py"
    assert cmd[2] == runner
    assert cmd[cmd.index("--adapter") + 1] == str(tmp_path / "adapter.pt")
    assert "--cache-dir" not in cmd


def test_extraction_command_keeps_all_suites_and_honors_smoke_cache(tmp_path):
    cfg = config(tmp_path)
    cmd = entry.build_command("extract", cfg, tmp_path / "local.yaml", args(cache_dir="feature_cache/smoke"),
                              ["--smoke", "--train-steps", "10"], tmp_path)
    assert cmd[cmd.index("--cache-dir") + 1] == str(tmp_path / "feature_cache/smoke")
    assert "libero_90" in cmd and "--smoke" in cmd
    assert cmd[cmd.index("--base-config") + 1] == str(tmp_path / "local.yaml")


def test_rollout_requires_explicit_adapter_and_local_weights(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="requires --adapter"):
        entry.build_command("rollout-online", cfg, tmp_path / "local.yaml", args(), [], tmp_path)
    cmd = entry.build_command("rollout-online", cfg, tmp_path / "local.yaml", args(adapter="adapter.pt"), [], tmp_path)
    assert f"ckpt={cfg.base_checkpoint}" in cmd
    assert f"+EVALUATION.geometry_adapter={tmp_path / 'adapter.pt'}" in cmd
    assert not any("cache_dir" in value for value in cmd)


def test_config_publication_refuses_overwrite_and_libero_avoids_home(tmp_path):
    cfg = config(tmp_path)
    env = entry.child_environment(cfg, tmp_path, {})
    entry.prepare_libero_config(cfg, env)
    entry.prepare_libero_config(cfg, env)
    saved = OmegaConf.load(Path(env["LIBERO_CONFIG_PATH"]) / "config.yaml")
    assert saved.assets == str(tmp_path / "third_party/LIBERO/libero/libero/assets")
    cfg.paths.data_root = str(tmp_path / "different_dataset")
    with pytest.raises(FileExistsError):
        entry.prepare_libero_config(cfg, env)


def test_discovery_rejects_missing_suite_and_selects_130(tmp_path):
    with pytest.raises(ValueError, match="download data first"):
        entry.discover(tmp_path, list(entry.SUITES))
    for suite, count in entry.SUITES.items():
        directory = tmp_path / suite
        directory.mkdir()
        for i in range(count):
            (directory / f"task_{i}_demo.hdf5").touch()
    assert len(entry.discover(tmp_path, list(entry.SUITES))) == 130
    with pytest.raises(ValueError, match="distinct"):
        entry.discover(tmp_path, ["libero_10", "libero_10"])


def test_dry_run_does_not_launch_subprocess_or_create_outputs(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    path = tmp_path / "local.yaml"
    OmegaConf.save(cfg, path)
    def forbidden(*a, **kw):
        raise AssertionError("dry-run launched a process")
    monkeypatch.setattr(entry.subprocess, "call", forbidden)
    assert entry.main(["train-online", "--config", str(path), "--dry-run"]) == 0
    assert not (tmp_path / "outputs").exists()
