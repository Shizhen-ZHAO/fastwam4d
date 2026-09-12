"""只读评测审查探针：抽取实际函数，隔离模拟器和大模型；不是 LIBERO 成功率测试。"""
from __future__ import annotations

import ast
import argparse
import contextlib
import importlib.machinery
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--reference-repo', required=True, type=Path)
parser.add_argument('--target-repo', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path)
args = parser.parse_args()
REPOS = [args.target_repo.resolve(), args.reference_repo.resolve()]
RESULT = {"scope": "AST/真实 Python 函数 + 假环境；无 CUDA、LIBERO 或真实模型权重"}


def extract(path, names, namespace=None):
    namespace = {} if namespace is None else namespace
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    for n in nodes:
        n.decorator_list = []
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


shared_names = ["_center_crop_resize", "_normalize_proprio", "_obs_to_model_input", "_extract_sim_state", "_denormalize_action", "_get_num_video_frames", "_get_max_steps", "_resolve_dataset_stats_path", "_load_model_checkpoint"]
trees = [{n.name: ast.dump(n, include_attributes=False) for n in ast.parse((r / "experiments/libero/eval_libero_single.py").read_text()).body if isinstance(n, ast.FunctionDef)} for r in REPOS]
RESULT["identical_eval_functions"] = [k for k in shared_names if trees[0][k] == trees[1][k]]
assert len(RESULT["identical_eval_functions"]) == len(shared_names)

with tempfile.TemporaryDirectory(prefix="fastwam-eval-probes-") as tmp:
    tmp = Path(tmp)
    fake_site = tmp / "installed_site"
    (fake_site / "fastwam").mkdir(parents=True)
    (fake_site / "fastwam/__init__.py").write_text("")
    imports = []
    for repo in REPOS:
        path = repo / "experiments/libero/eval_libero_single.py"
        body = ast.parse(path.read_text()).body
        setup = []
        for n in body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"project_root", "PROJECT_ROOT", "SRC_ROOT", "src_root"} for t in n.targets):
                setup.append(n)
            elif isinstance(n, ast.If) and "sys.path" in ast.unparse(n.test):
                setup.append(n)
            elif isinstance(n, ast.Expr) and "sys.path.insert" in ast.unparse(n):
                setup.append(n)
        local_sys = types.SimpleNamespace(path=[str(fake_site)])
        exec(compile(ast.Module(body=setup, type_ignores=[]), str(path), "exec"), {"Path": Path, "sys": local_sys, "__file__": str(path)})
        origin = importlib.machinery.PathFinder.find_spec("fastwam", local_sys.path).origin
        imports.append({"repo": repo.name, "selected_local_repo": str(repo / "src") in origin, "origin": origin})
    assert all(item["selected_local_repo"] for item in imports)
    RESULT["eval_source_resolution_with_other_installation"] = imports

    ckpt = tmp / "run/checkpoints/weights/step_000001.pt"
    ckpt.parent.mkdir(parents=True)
    fallback = tmp / "run/dataset_stats.json"
    fallback.write_text("{}")
    cfg = OmegaConf.create({"ckpt": str(ckpt), "EVALUATION": {"dataset_stats_path": str(tmp / "missing-explicit.json")}})
    stats = []
    for repo in REPOS:
        ns = extract(repo / "experiments/libero/eval_libero_single.py", {"_resolve_dataset_stats_path"}, {"Path": Path, "os": os})
        selected = ns["_resolve_dataset_stats_path"](cfg)
        assert selected == fallback.resolve()
        stats.append({"repo": repo.name, "missing_explicit_path_falls_back": True})
    RESULT["stats_resolution"] = stats

    old_result = tmp / "old_output/libero_spatial/gpu0_task0_results.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text(json.dumps({"successes": 50, "total_episodes": 50}))
    ns = extract(REPOS[0] / "experiments/libero/eval_libero_single.py", {"_result_file"}, {"Path": Path})
    RESULT["official_existing_result"] = {"old_result_is_accepted_without_ckpt_or_trial_check": ns["_result_file"](old_result.parent.parent, "libero_spatial", 0) == old_result}

    worker_module = REPOS[0] / "experiments/libero/worker_pool.py"
    import_code = "import importlib.util,sys; from pathlib import Path; s=importlib.util.spec_from_file_location('pool',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
    crash_code = import_code + "exec('with m.task_queue_lock(Path(sys.argv[2])):\\n import os; os._exit(99)')"
    lock = tmp / "queue.lock"
    crashed = subprocess.run([sys.executable, "-c", crash_code, str(worker_module), str(lock)], capture_output=True, text=True)
    assert crashed.returncode == 99, crashed.stderr
    blocked_code = import_code + "m.pending_task_count(Path(sys.argv[3]),Path(sys.argv[2]))"
    pending = tmp / "pending.txt"
    pending.write_text("libero_spatial,0\n")
    blocked = subprocess.Popen([sys.executable, "-c", blocked_code, str(worker_module), str(lock), str(pending)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        blocked.communicate(timeout=0.6)
        raise AssertionError("Expected orphan queue lock to block")
    except subprocess.TimeoutExpired:
        blocked.kill()
        blocked.communicate()
    RESULT["official_queue_orphan_lock"] = {"crash_exit_code": 99, "lock_directory_survived": lock.with_name(lock.name + ".lockdir").exists(), "next_operation_blocked_until_probe_timeout": True}


class Progress:
    def update(self, *_):
        pass

    def close(self):
        pass


class FakeEnv:
    def __init__(self, done_at):
        self.env = types.SimpleNamespace(sim=self)
        self.done_at = done_at
        self.t = 0
        self.actions = []
        self.render_calls = 0

    def render(self, width=None, height=None, **kwargs):
        self.render_calls += 1
        return np.full((height, width, 3), self.t + 1, np.uint8)

    def observation(self):
        return {"image": self.render(width=2, height=2)}

    def reset(self):
        self.t = 0
        return self.observation()

    def set_init_state(self, _):
        return self.observation()

    def step(self, action):
        self.actions.append(list(action))
        self.t += 1
        return self.observation(), 0, self.t >= self.done_at, {}


rollouts = []
for wait_steps in (0, 1, 30):
    outputs = []
    for repo, skip in [(REPOS[0], False), (REPOS[1], False), (REPOS[1], True)]:
        consumed = []
        cfg = OmegaConf.create({"seed": 42, "EVALUATION": {"task_suite_name": "libero_spatial", "replan_steps": 10, "num_steps_wait": wait_steps, "use_action_ensembler": False, "visualize_future_video": False, "skip_unused_render": skip, "save_video": False, "infer_increment_seed": False}})

        def predict(**kwargs):
            obs = kwargs["obs"]
            value = int(obs["image"][0, 0, 0])
            consumed.append((value, kwargs.get("infer_seed", 42)))
            return np.full((32, 7), float(value)), obs, None

        names = {"run_single_episode", "_resolve_infer_seed", "_maybe_install_render_gate", "_SimRenderGate"}
        ns = extract(repo / "experiments/libero/eval_libero_single.py", names, {"np": np, "logging": logging, "_SKIP_RENDER_LOGGED": False, "_INFER_INCREMENT_SEED_LOGGED": False, "_get_max_steps": lambda _: 400, "_get_future_frame_capture_steps": lambda _: [0, 4, 8], "_predict_action_chunk": predict, "get_libero_dummy_action": lambda: [0.] * 7, "get_libero_image": lambda obs: obs, "tqdm": lambda **_: Progress()})
        env = FakeEnv(wait_steps + 23)
        value = ns["run_single_episode"](env, None, "dummy", None, None, cfg, 0, action_horizon=32, input_w=224, input_h=224, model_device="cpu")
        outputs.append({"repo": repo.name, "skip_unused_render": skip, "consumed": consumed, "actions": env.actions, "success": value[0], "render_calls": env.render_calls})
    for result in outputs[1:]:
        assert result["consumed"] == outputs[0]["consumed"]
        assert result["actions"] == outputs[0]["actions"]
        assert result["success"] == outputs[0]["success"]
    rollouts.append({"wait_steps": wait_steps, "identical_consumed_frames_seeds_actions_and_done": True, "consumed": outputs[0]["consumed"], "render_calls": [x["render_calls"] for x in outputs]})
RESULT["rollout_control_flow_parity"] = rollouts
text = json.dumps(RESULT, ensure_ascii=False, indent=2) + '\n'
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
print(text)
