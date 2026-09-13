"""本地功能验证；隔离模型/仿真依赖，不冒充真实 LIBERO 成功率测试。"""
import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.libero_plus import eval_utils, parallel_workers, task_utils
from experiments.libero_plus.run_libero_plus_manager import prepare_tasks, write_worker_config


def config(variant="uncond"):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="sim_libero_plus", overrides=[f"task=libero_{variant}_2cam224_1e-4"])
    cfg.EVALUATION.output_dir = "/tmp/plus-test"
    return cfg


@pytest.mark.parametrize("variant", ["uncond", "joint", "idm"])
def test_launcher_final_config(variant, tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text("NUM_TRIALS: 2\nINFER_INCREMENT_SEED: false\n")
    env = os.environ.copy()
    env.update(DRY_RUN="1", PYTHON_BIN=sys.executable, EVAL_MODE=variant, NUM_GPUS="1",
               MAX_TASKS_PER_GPU="1", NUM_TRIALS="3", CKPT="/tmp/trained.pt",
               OUTPUT_DIR=str(tmp_path / "dry"))
    proc = subprocess.run(["bash", str(ROOT / "experiments/libero_plus/run_eval.sh"),
                           "--config", str(override), "EVALUATION.num_trials=4"],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    cfg = OmegaConf.create(proc.stdout.split("\n", 1)[1])
    assert eval_utils.validate_eval_config(cfg) == variant
    assert cfg.EVALUATION.num_trials == 4
    assert cfg.EVALUATION.infer_increment_seed is False
    assert cfg.EVALUATION.skip_unused_render is True
    assert cfg.model.load_text_encoder is True
    assert cfg.model.action_dit_pretrained_path is None
    assert not (tmp_path / "dry").exists()


def test_config_rejects_incompatible_rope_and_compile():
    for key, value in [("model.action_dit_config.action_rope_mode", "3d_aligned"),
                       ("EVALUATION.compile_action_infer", True),
                       ("EVALUATION.replan_steps", 0)]:
        cfg = config()
        OmegaConf.update(cfg, key, value)
        with pytest.raises(ValueError):
            eval_utils.validate_eval_config(cfg)


@pytest.mark.parametrize("variant", ["uncond", "joint", "idm"])
def test_worker_snapshot_keeps_config_and_disables_shared_log(tmp_path, variant):
    cfg = config(variant)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    write_worker_config(resolved, tmp_path)
    with initialize_config_dir(version_base="1.3", config_dir=str(tmp_path)):
        worker = compose(config_name="worker_config", return_hydra_config=True)
    assert OmegaConf.to_container(worker.model, resolve=True) == resolved["model"]
    assert OmegaConf.to_container(worker.EVALUATION, resolve=True) == resolved["EVALUATION"]
    assert worker.hydra.job.chdir is False
    assert worker.hydra.output_subdir is None
    assert "file" not in worker.hydra.job_logging.get("handlers", {})


def test_asset_preflight(tmp_path, monkeypatch):
    cfg = config()
    cfg.model.redirect_common_files = False
    cfg.model.model_id = "wan"
    cfg.model.tokenizer_model_id = str(tmp_path / "tokenizer")
    monkeypatch.setenv("DIFFSYNTH_MODEL_BASE_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        eval_utils.validate_model_assets(cfg)
    (tmp_path / "wan").mkdir()
    for name in ("Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth"):
        (tmp_path / "wan" / name).write_bytes(b"fixture")
    tokenizer = tmp_path / "tokenizer/google/umt5-xxl"
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer_config.json").write_text("{}")
    assert eval_utils.validate_model_assets(cfg)["tokenizer"] == str(tokenizer)


def test_create_only_full_launcher_with_benchmark_fixture(tmp_path):
    root = tmp_path / "plus_fixture"
    package = root / "libero/libero"
    package.mkdir(parents=True)
    (root / "libero/__init__.py").write_text("")
    (package / "__init__.py").write_text('''import os, yaml
from pathlib import Path
config_file = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'
def get_libero_path(key):
    return yaml.safe_load(config_file.read_text())[key]
''')
    (package / "benchmark.py").write_text('''from types import SimpleNamespace
class Suite:
    n_tasks = 2
    def get_task(self, index):
        return SimpleNamespace(name=f'task{index}', bddl_file=f'task{index}.bddl')
def get_benchmark_dict():
    return {name: Suite for name in ('libero_10', 'libero_goal', 'libero_spatial', 'libero_object')}
''')
    for name in ["bddl_files", "init_files"]:
        (package / name).mkdir()
    assets = root / "assets"
    assets.mkdir()
    output = tmp_path / "created"
    env = os.environ.copy()
    env.update(PYTHON_BIN=sys.executable, EVAL_MODE="uncond", NUM_GPUS="1", CUDA_VISIBLE_DEVICES="0",
               MAX_TASKS_PER_GPU="1", TASK_FILE="auto", OUTPUT_DIR=str(output),
               LIBERO_PLUS_ROOT=str(root), LIBERO_PLUS_ASSETS_DIR=str(assets), SHARE_BACKUP="false")
    env.pop("CKPT", None)
    command = ["bash", str(ROOT / "experiments/libero_plus/run_eval.sh"), "MULTIRUN.create_only=true"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(task_utils.read_task_file(output / "tasks.txt")) == 8
    assert json.loads((output / "eval_meta.json").read_text())["create_only"]
    assert not (output / "worker_results").exists()
    rerun = subprocess.run(command, env=env, capture_output=True, text=True)
    assert rerun.returncode != 0 and "输出目录非空" in rerun.stderr


def test_stats_and_checkpoint_metadata(tmp_path):
    cfg = config()
    ckpt = tmp_path / "checkpoints/weights/step.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"test fixture")
    cfg.ckpt = str(ckpt)
    (tmp_path / "dataset_stats.json").write_text("{}")
    assert eval_utils.resolve_stats_path(cfg) == tmp_path / "dataset_stats.json"
    cfg.EVALUATION.dataset_stats_path = str(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        eval_utils.resolve_stats_path(cfg)
    training = tmp_path / "config.yaml"
    training.write_text("model:\n  action_dit_config:\n    action_rope_mode: 3d_aligned\n")
    with pytest.raises(ValueError, match="配置冲突"):
        eval_utils.validate_checkpoint_config(cfg)
    training.write_text("model:\n  action_dit_config:\n    action_rope_mode: ${oc.env:FASTWAM_ACTION_ROPE_MODE,3d_aligned}\n")
    assert "model.action_dit_config.action_rope_mode" in eval_utils.validate_checkpoint_config(cfg)["unknown"]


def test_initial_state_compatibility_and_restore(monkeypatch):
    original = torch.load
    calls = []
    monkeypatch.setattr(torch, "load", lambda *a, **kw: calls.append(kw) or [np.zeros(3)])
    before = torch.load
    suite = types.SimpleNamespace(get_task_init_states=lambda _: torch.load("fixture"))
    states = eval_utils.get_task_init_states(suite, 0)
    assert calls == [{"weights_only": False}]
    assert torch.load is before
    for array in (states, np.arange(6).reshape(2, 3), torch.arange(6).reshape(2, 3)):
        repeated = eval_utils.repeat_initial_states(array, 5)
        np.testing.assert_array_equal(repeated[0], repeated[2])
        assert len(repeated) == 5
    with pytest.raises(ValueError):
        eval_utils.repeat_initial_states([], 1)
    def explode(_):
        raise RuntimeError("fixture error")
    with pytest.raises(RuntimeError):
        eval_utils.get_task_init_states(types.SimpleNamespace(get_task_init_states=explode), 0)
    assert torch.load is before


def test_lpt_and_explicit_subset_preserved(tmp_path):
    lpt = ROOT / "experiments/libero_plus/full_10030_lpt.txt"
    assert task_utils.file_sha256(lpt) == "e0e2c9b2291c2b0c92e7c3a5eafe6141ba4ea7caccdbede4906411be235f1e08"
    tasks = task_utils.read_task_file(lpt)
    assert len(tasks) == 10030
    subset = tmp_path / "subset.txt"
    subset.write_text("libero_10,2\nlibero_10,0\n")
    cfg = config()
    cfg.MULTIRUN.task_file = str(subset)
    output = tmp_path / "out"
    output.mkdir()
    suite = types.SimpleNamespace(n_tasks=3, get_task=lambda i: types.SimpleNamespace(name=f"task{i}", bddl_file=f"t{i}.bddl"))
    result = prepare_tasks(cfg, output, {"libero_10": lambda: suite})
    assert result.read_bytes() == subset.read_bytes() == b"libero_10,2\nlibero_10,0\n"
    for gpus, density in ((["3"], 1), (["2", "7"], 2), ([str(i) for i in range(16)], 3)):
        out = tmp_path / f"shards_{len(gpus)}"
        out.mkdir()
        workers = parallel_workers.prepare_workers(lpt, out, gpus, density)
        by_id = {w["id"]: w for w in workers}
        for i, task in enumerate(tasks):
            w = by_id[i % (len(gpus) * density)]
            assert w["tasks"][i // (len(gpus) * density)] == task
            assert w["gpu"] == gpus[w["id"] % len(gpus)]
    with pytest.raises(ValueError):
        task_utils.validate_gpu_ids("0,1", 1)
    with pytest.raises(ValueError):
        task_utils.validate_gpu_ids("1,1", 2)


def write_results(output):
    output.mkdir(exist_ok=True)
    tasks = [("libero_10", 0), ("libero_10", 1), ("libero_goal", 0)]
    (output / "tasks.txt").write_text("".join(f"{s},{i}\n" for s, i in tasks))
    (output / "eval_meta.json").write_text(json.dumps({"num_trials": 1}))
    workers = parallel_workers.prepare_workers(output / "tasks.txt", output, ["0", "1"], 1)
    for worker in workers:
        results = []
        for suite, task_id in worker["tasks"]:
            success = int(suite == "libero_10")
            results.append(dict(task_suite=suite, task_id=task_id, task_name=f"{suite}_{task_id}",
                                successes=success, total_episodes=1, duration=1.0,
                                success_episodes=[0] if success else [], failure_episodes=[] if success else [0]))
        worker["result"].write_text(json.dumps(dict(format_version=task_utils.RESULT_FORMAT,
                    worker_id=str(worker["id"]), total_tasks=len(results), completed_tasks=len(results),
                    failed_tasks=[], results=results)))
    return workers


def test_summary_and_completion_gate(tmp_path):
    from experiments.libero_plus.summarize_results import summarize_results
    workers = write_results(tmp_path)
    summarize_results(tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["overall"]["average_success_rate"] == 50.0  # 宏平均，与 2/3 微平均不同
    assert json.loads((tmp_path / "coverage.json").read_text())["tasks"] == 3
    original = workers[0]["result"].read_text()
    payload = json.loads(original)
    payload["results"].pop()
    workers[0]["result"].write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        summarize_results(tmp_path)
    workers[0]["result"].write_text(original)
    duplicate = tmp_path / "worker_results/worker99_results.json"
    duplicate.write_text(original)
    with pytest.raises(ValueError):
        task_utils.validate_run(tmp_path)


def extract(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if getattr(n, "name", None) in names]
    for node in nodes:
        node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


class FakeEnv:
    def __init__(self, done_at, fail_at=None):
        self.env = types.SimpleNamespace(sim=self)
        self.done_at, self.fail_at = done_at, fail_at
        self.t, self.renders, self.actions = 0, 0, []
    def render(self, width=None, height=None, **kwargs):
        self.renders += 1
        return np.full((height, width, 3), self.t + 1, np.uint8)
    def obs(self):
        return {"image": self.render(width=2, height=2)}
    def reset(self):
        self.t = 0
        return self.obs()
    def set_init_state(self, state):
        return self.obs()
    def step(self, action):
        self.actions.append(list(action))
        self.t += 1
        if self.t == self.fail_at:
            raise RuntimeError("sim fixture failure")
        return self.obs(), 0, self.t >= self.done_at, {}


def rollout(repo, *, wait=30, skip=True, increment=True, fail=False):
    consumed = []
    def predict(**kw):
        value = int(kw["obs"]["image"][0, 0, 0])
        seed = kw.get("infer_seed", 42)
        consumed.append((value, seed))
        return np.full((32, 7), float(value + seed)), kw["obs"], None
    cfg = OmegaConf.create(dict(seed=42, EVALUATION=dict(task_suite_name="libero_10", replan_steps=10,
            num_steps_wait=wait, use_action_ensembler=False, visualize_future_video=False,
            skip_unused_render=skip, save_video=False, infer_increment_seed=increment)))
    namespace = dict(np=np, logging=logging, _SKIP_RENDER_LOGGED=False, _INFER_INCREMENT_SEED_LOGGED=False,
            _get_max_steps=lambda _: 400, _get_future_frame_capture_steps=lambda _: [0, 4, 8],
            _predict_action_chunk=predict, get_libero_dummy_action=lambda: [0.] * 7,
            get_libero_image=lambda obs: obs,
            tqdm=lambda **kw: types.SimpleNamespace(update=lambda *a: None, close=lambda: None))
    ns = extract(repo / "experiments/libero/eval_libero_single.py",
                 {"run_single_episode", "_resolve_infer_seed", "_maybe_install_render_gate", "_SimRenderGate"}, namespace)
    env = FakeEnv(wait + 23, 2 if fail else None)
    original = env.render
    if fail:
        with pytest.raises(RuntimeError):
            ns["run_single_episode"](env, None, "test", None, None, cfg, 0,
                action_horizon=32, input_w=224, input_h=224, model_device="cpu")
        assert env.render == original
        return
    result = ns["run_single_episode"](env, None, "test", None, None, cfg, 0,
        action_horizon=32, input_w=224, input_h=224, model_device="cpu")
    assert env.render == original
    return consumed, env.actions, result[0], env.renders


@pytest.mark.parametrize("wait", [0, 1, 30])
@pytest.mark.parametrize("increment", [False, True])
def test_render_gate_preserves_consumed_frames_and_actions(wait, increment):
    off = rollout(ROOT, wait=wait, skip=False, increment=increment)
    on = rollout(ROOT, wait=wait, skip=True, increment=increment)
    assert off[:3] == on[:3]
    assert on[3] < off[3]
    assert [seed for _, seed in on[0]] == ([42, 43, 44] if increment else [42] * 3)


def test_render_restored_on_failure():
    rollout(ROOT, fail=True)


def test_rollout_matches_reference():
    reference = os.environ.get("FASTWAM_REFERENCE_REPO")
    if not reference:
        pytest.skip("设置 FASTWAM_REFERENCE_REPO 执行两仓库真实函数的对照")
    for wait in [0, 1, 30]:
        for increment in [False, True]:
            for skip in [False, True]:
                assert rollout(ROOT, wait=wait, skip=skip, increment=increment) == rollout(
                    Path(reference), wait=wait, skip=skip, increment=increment)


def test_save_video_flag_and_env_cleanup():
    closed, saved = [], []
    env = types.SimpleNamespace(close=lambda: closed.append(True))
    namespace = dict(get_libero_env=lambda *a: (env, "test"), LIBERO_ENV_RESOLUTION=256,
                     run_single_episode=lambda **kw: (True, [], [], None), np=np,
                     save_rollout_video=lambda *a, **kw: saved.append(True))
    ns = extract(ROOT / "experiments/libero/eval_libero_single.py", {"run_single_task"}, namespace)
    for video in [False, True]:
        cfg = OmegaConf.create(dict(seed=42, EVALUATION=dict(num_trials=1, task_id=0, save_video=video)))
        ns["run_single_task"](None, [None], None, None, cfg, Path("/unused"), Path("/unused"),
            action_horizon=32, input_w=224, input_h=224, model_device="cpu")
    assert len(closed) == 2 and len(saved) == 1
    def explode(**kw):
        raise RuntimeError("fixture")
    ns["run_single_episode"] = explode
    with pytest.raises(RuntimeError):
        ns["run_single_task"](None, [None], None, None, cfg, Path("/unused"), Path("/unused"),
            action_horizon=32, input_w=224, input_h=224, model_device="cpu")
    assert len(closed) == 3


@pytest.mark.parametrize("variant", ["uncond", "joint", "idm"])
def test_persistent_worker_loads_model_once(tmp_path, monkeypatch, variant):
    import platform
    import time
    cfg = config(variant)
    cfg.EVALUATION.output_dir = str(tmp_path)
    cfg.EVALUATION.device = "cpu"
    ckpt = tmp_path / "step.pt"
    ckpt.write_bytes(b"fixture")
    cfg.ckpt = str(ckpt)
    stats = tmp_path / "dataset_stats.json"
    stats.write_text("{}")
    task_file = tmp_path / "tasks.txt"
    task_file.write_text("libero_10,0\nlibero_goal,0\n")
    monkeypatch.setenv("LIBERO_PLUS_WORKER_TASK_FILE", str(task_file))
    monkeypatch.setenv("LIBERO_PLUS_WORKER_ID", "2")
    result_file = tmp_path / "worker_results/worker2_gpu0_results.json"
    monkeypatch.setenv("LIBERO_PLUS_WORKER_RESULT_FILE", str(result_file))
    models, executed, seeds = [], [], []
    model = types.SimpleNamespace()
    model.to = lambda *_: model
    model.eval = lambda: model
    processor = types.SimpleNamespace(set_normalizer_from_stats=lambda _: None)
    processor.eval = lambda: processor
    def instantiate(cfg, **kwargs):
        if str(cfg._target_).startswith("fastwam.runtime"):
            models.append(model)
            return model
        return processor
    def task(**kwargs):
        executed.append((str(kwargs["cfg"].EVALUATION.task_suite_name), kwargs["model"]))
        return dict(successes=1, success_episodes=[0], failure_episodes=[], task_description="test")
    suite = types.SimpleNamespace(get_task=lambda _: types.SimpleNamespace(name="task", bddl_file="task.bddl"),
                                 get_task_init_states=lambda _: np.zeros((1, 4)))
    namespace = dict(json=json, logging=logging, os=os, time=time, Path=Path, platform=platform,
            instantiate=instantiate, NumpyEncoder=json.JSONEncoder,
            set_global_seed=lambda value, **kw: seeds.append(value),
            _validate_visualize_future_video_cfg=lambda _: None,
            _resolve_eval_device=lambda _: "cpu", _mixed_precision_to_model_dtype=lambda _: torch.float32,
            _load_model_checkpoint=lambda *a: None,
            load_dataset_stats_from_json=lambda p: json.loads(Path(p).read_text()),
            read_task_file=task_utils.read_task_file, run_single_task=task,
            benchmark=types.SimpleNamespace(get_benchmark_dict=lambda: {s: lambda: suite for s in task_utils.SUITES}),
            get_task_init_states=eval_utils.get_task_init_states,
            repeat_initial_states=eval_utils.repeat_initial_states,
            resolve_stats_path=eval_utils.resolve_stats_path,
            validate_checkpoint_config=eval_utils.validate_checkpoint_config,
            validate_eval_config=eval_utils.validate_eval_config)
    ns = extract(ROOT / "experiments/libero_plus/eval_libero_plus_worker.py", {"main", "_write_worker_results"}, namespace)
    ns["main"](cfg)
    assert models == [model] and seeds == [44]
    assert [s for s, m in executed] == ["libero_10", "libero_goal"]
    assert all(m is model for s, m in executed)
    task_utils.validate_worker_result(result_file, task_utils.read_task_file(task_file), 1)


def test_reference_summary_numerical_parity(tmp_path):
    reference = os.environ.get("FASTWAM_REFERENCE_REPO")
    if not reference:
        pytest.skip("设置 FASTWAM_REFERENCE_REPO 运行参考汇总对照")
    from experiments.libero_plus.summarize_results import summarize_results
    spec = importlib.util.spec_from_file_location("reference_plus_summary", Path(reference) / "experiments/libero/summarize_results.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    write_results(tmp_path)
    classification = tmp_path / "classification.json"
    classification.write_text(json.dumps({"libero_10": [dict(id=1, name="libero_10_0", category="visual", difficulty_level="easy"),
                                                       dict(id=2, name="libero_10_1", category="visual", difficulty_level="hard")],
                                          "libero_goal": [dict(id=1, name="libero_goal_0", category="language", difficulty_level="easy")]}))
    summarize_results(tmp_path, task_classification_path=str(classification))
    paths = ["summary.json", "summary.csv", "task_success_rates.csv", "category_success_rates.csv", "difficulty_success_rates.csv"]
    before = {name: (tmp_path / name).read_bytes() for name in paths}
    ref.summarize_results(tmp_path, task_classification_path=str(classification))
    assert before == {name: (tmp_path / name).read_bytes() for name in paths}
