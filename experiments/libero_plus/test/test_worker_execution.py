"""执行真实 worker launcher/监督循环；用本地子进程替代 tmux 和大模型。"""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.libero_plus import parallel_workers
from experiments.libero_plus.trace_worker import install_trace, compatible_instantiate
from experiments.libero_plus.compare_traces import compare, compare_run_metadata


@pytest.mark.parametrize("mode", ["success", "error", "dead", "existing"])
def test_supervisor_executes_launchers_and_propagates_failure(tmp_path, monkeypatch, mode):
    output = tmp_path / "run with space and ' quote"
    output.mkdir()
    task_file = output / "tasks.txt"
    task_file.write_text("libero_10,0\nlibero_goal,0\nlibero_10,1\n")
    (output / "eval_meta.json").write_text(json.dumps({"num_trials": 1}))
    dummy = tmp_path / "python fixture"
    dummy.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
from pathlib import Path
if not sys.argv[1].endswith('eval_libero_plus_worker.py'):
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
if ''' + repr(mode) + ''' == 'error':
    sys.exit(7)
tasks = Path(os.environ['LIBERO_PLUS_WORKER_TASK_FILE']).read_text().splitlines()
rows = [dict(task_suite=s, task_id=int(i), task_name=s+i, successes=1, total_episodes=1,
             duration=1., success_episodes=[0], failure_episodes=[])
        for s, i in (line.split(',') for line in tasks)]
payload = dict(format_version='libero_plus_worker_results_v1',
               worker_id=os.environ['LIBERO_PLUS_WORKER_ID'], total_tasks=len(rows),
               completed_tasks=len(rows), results=rows, failed_tasks=[])
Path(os.environ['LIBERO_PLUS_WORKER_RESULT_FILE']).write_text(json.dumps(payload))
''')
    dummy.chmod(0o700)
    for key, value in dict(OUTPUT_DIR=output, NUM_GPUS="2", CUDA_VISIBLE_DEVICES="2,7",
                           MAX_TASKS_PER_GPU="1", NUM_TRIALS="1", PYTHON_BIN=dummy,
                           EXTRA_ARGS="--config-name worker_config", SESSION_NAME="plus_unit_test",
                           WORKER_START_WAVE_INTERVAL="0", MONITORING_INTERVAL="0.01").items():
        monkeypatch.setenv(key, str(value))
    processes, killed = {}, []
    monkeypatch.setattr(parallel_workers.shutil, "which", lambda _: "/test/tmux")
    def tmux(*args, check=True):
        command = args[0]
        stdout, rc = "", 0
        if command == "has-session":
            rc = 0 if mode == "existing" else 1
        elif command == "new-window":
            pane = f"%{len(processes) + 1}"
            processes[pane] = None if mode == "dead" else subprocess.Popen(shlex.split(args[-1]))
            stdout = pane + "\n"
        elif command == "list-panes":
            stdout = "%0\n" + "\n".join(p for p, process in processes.items() if process is not None and process.poll() is None)
        elif command == "kill-session":
            killed.append(args[-1])
            for process in processes.values():
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
        return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="")
    monkeypatch.setattr(parallel_workers, "tmux", tmux)
    monkeypatch.setattr(parallel_workers.signal, "signal", lambda *a: None)
    if mode == "success":
        parallel_workers.run(task_file)
        assert json.loads((output / "coverage.json").read_text())["passed"]
        assert len(list((output / "worker_results").glob("*.json"))) == 2
    else:
        with pytest.raises((RuntimeError, FileExistsError)):
            parallel_workers.run(task_file)
    assert bool(killed) == (mode != "existing")


def test_joint_factory_forwards_compile_flag(monkeypatch):
    namespace = {"__name__": "fastwam.runtime", "__package__": "fastwam", "DictConfig": DictConfig,
                 "OmegaConf": OmegaConf, "torch": torch}
    captured = []
    module = types.ModuleType("fastwam.models.wan22.fastwam_joint")
    module.FastWAMJoint = types.SimpleNamespace(from_wan22_pretrained=lambda **kw: captured.append(kw) or kw)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    node = next(n for n in ast.parse((ROOT / "src/fastwam/runtime.py").read_text()).body
                if getattr(n, "name", "") == "create_fastwam_joint")
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "runtime.py", "exec"), namespace)
    factory = namespace["create_fastwam_joint"]
    kwargs = dict(model_id="wan", tokenizer_model_id="t5", video_dit_config={},
                  action_scheduler=dict(train_shift=5., infer_shift=5., num_train_timesteps=1000))
    factory(**kwargs)
    factory(**kwargs, compile_training_denoise=False)
    factory(**kwargs, compile_training_denoise=True)
    assert [call["compile_training_denoise"] for call in captured] == [False, False, True]


def test_trace_instrumentation_does_not_change_actions(tmp_path):
    import torch
    raw_calls = []
    def inputs(*args, **kwargs):
        return torch.ones(1, 3, 2, 2), torch.ones(1, 8), {}
    def denorm(action, processor):
        return action.numpy()[None] * 2
    core = types.SimpleNamespace(_obs_to_model_input=inputs, _denormalize_action=denorm)
    def predict(**kwargs):
        raw_calls.append(kwargs["infer_seed"])
        core._obs_to_model_input()
        action = core._denormalize_action(torch.ones(2, 7), None)[0]
        return action, {}, None
    core._predict_action_chunk = predict
    core._extract_sim_state = lambda obs: np.asarray(obs["state"], dtype=np.float32)
    def episode(**kwargs):
        for seed in [42, 43]:
            action, _, _ = core._predict_action_chunk(cfg=kwargs["cfg"], infer_seed=seed,
                                                     task_description="test", obs={})
            kwargs["env"].step(action[0])
        return True, [], [], None
    core.run_single_episode = episode
    cfg = OmegaConf.create(dict(seed=42, EVALUATION=dict(task_suite_name="libero_10", task_id=0)))
    for directory in [tmp_path / "a", tmp_path / "b"]:
        # 每次构造新的 wrapper，第二次沿用同一 core 时会产生双层装饰，需显式恢复。
        old = dict(vars(core))
        install_trace(core, directory)
        env = types.SimpleNamespace(step=lambda action: ({"state": np.zeros(8)}, 0, True, {}))
        core.run_single_episode(cfg=cfg, env=env, initial_state=np.zeros(4), episode_idx=0)
        vars(core).update(old)
    assert raw_calls == [42, 43, 42, 43]
    assert compare(tmp_path / "a", tmp_path / "b")["passed"]
    changed = next((tmp_path / "b").rglob("replan_00000.npz"))
    with np.load(changed) as data:
        fields = {key: data[key] for key in data.files}
    fields["executed_action_chunk"][0, 0] += 1
    np.savez(changed, **fields)
    with pytest.raises(ValueError, match="首次偏离"):
        compare(tmp_path / "a", tmp_path / "b")


def test_reference_factory_only_omits_false(monkeypatch):
    import hydra.utils
    monkeypatch.setattr(hydra.utils, "get_method", lambda _: (lambda model_id: None))
    observed = []
    wrapped = compatible_instantiate(lambda cfg, **kw: observed.append(cfg) or cfg)
    cfg = OmegaConf.create(dict(_target_="fastwam.runtime.create_fastwam", compile_training_denoise=False))
    wrapped(cfg)
    assert "compile_training_denoise" not in observed[0]
    assert "compile_training_denoise" in cfg
    cfg.compile_training_denoise = True
    with pytest.raises(ValueError):
        wrapped(cfg)


def test_trace_gate_rejects_partial_completed_runs(tmp_path):
    for name in ["left", "right"]:
        root = tmp_path / name
        episode = root / "traces/worker0/libero_10/task0_trial0"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text("{}")
        (root / "coverage.json").write_text('{"passed": true}')
        (root / "eval_meta.json").write_text(json.dumps(dict(eval_mode="uncond", num_trials=1,
                    ckpt_sha256="a", dataset_stats_sha256="b", task_file_sha256="c")))
        (root / "task_manifest.json").write_text(json.dumps([dict(suite="libero_10", task_id=0)]))
        OmegaConf.save(OmegaConf.create(dict(EVALUATION={}, MULTIRUN={})), root / "manager_config.yaml")
    compare_run_metadata(tmp_path / "left/traces", tmp_path / "right/traces")
    (tmp_path / "left/coverage.json").write_text('{"passed": false}')
    with pytest.raises(ValueError, match="完整性"):
        compare_run_metadata(tmp_path / "left/traces", tmp_path / "right/traces")
    (tmp_path / "left/coverage.json").write_text('{"passed": true}')
    (tmp_path / "left/traces/worker0/libero_10/task0_trial0/episode.json").unlink()
    with pytest.raises(ValueError, match="完整覆盖"):
        compare_run_metadata(tmp_path / "left/traces", tmp_path / "right/traces")
