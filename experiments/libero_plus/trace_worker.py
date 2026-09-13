"""只用于小规模 A/B：记录实际 worker 的输入、动作和环境步骤，不替换模型。

FASTWAM_PLUS_TRACE_REPO 可指向参考仓库；源码不用修改，worker 配置沿用 manager 快照。
"""
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def compatible_instantiate(original):
    """参考工厂没有 compile 开关时，只移除等价的 false，不放宽其他参数。"""
    from hydra.utils import get_method
    from omegaconf import OmegaConf, open_dict

    def instantiate(cfg, *args, **kwargs):
        if str(cfg.get("_target_", "")).startswith("fastwam.runtime.") and "compile_training_denoise" in cfg:
            factory = get_method(str(cfg._target_))
            if "compile_training_denoise" not in inspect.signature(factory).parameters:
                if bool(cfg.compile_training_denoise):
                    raise ValueError("参考工厂不支持编译，不能移除 true 开关")
                cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                with open_dict(cfg):
                    del cfg.compile_training_denoise
        return original(cfg, *args, **kwargs)
    return instantiate


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def install_trace(core, output):
    context = {}
    original_predict = core._predict_action_chunk
    original_input = core._obs_to_model_input
    original_denorm = core._denormalize_action
    original_episode = core.run_single_episode

    def model_input(*args, **kwargs):
        result = original_input(*args, **kwargs)
        context["input_image"] = array(result[0])
        if result[1] is not None:
            context["input_proprio"] = array(result[1])
        return result

    def denormalize(action, processor):
        context["normalized_action"] = array(action)
        result = original_denorm(action, processor)
        context["denormalized_action"] = array(result).copy()
        return result

    def predict(*args, **kwargs):
        if args:
            raise ValueError("轨迹记录器要求 _predict_action_chunk 以具名参数调用")
        cfg = kwargs["cfg"]
        result = original_predict(**kwargs)
        payload = {key: context[key] for key in ("input_image", "input_proprio", "normalized_action", "denormalized_action") if key in context}
        payload["executed_action_chunk"] = array(result[0])
        payload["seed"] = np.asarray(kwargs.get("infer_seed", cfg.seed), dtype=np.int64)
        payload["task_description"] = np.asarray(kwargs["task_description"])
        path = context["directory"] / f"replan_{context['replan']:05d}.npz"
        np.savez(path, **payload)
        context["replan"] += 1
        return result

    def episode(*args, **kwargs):
        if args:
            raise ValueError("轨迹记录器要求 run_single_episode 以具名参数调用")
        cfg, env = kwargs["cfg"], kwargs["env"]
        directory = output / str(cfg.EVALUATION.task_suite_name) / f"task{cfg.EVALUATION.task_id}_trial{kwargs['episode_idx']}"
        directory.mkdir(parents=True, exist_ok=False)
        context.clear()
        context.update(directory=directory, replan=0)
        np.save(directory / "initial_state.npy", array(kwargs["initial_state"]), allow_pickle=False)
        actions, states, done_flags = [], [], []
        original_step = env.step

        def step(action):
            result = original_step(action)
            actions.append(array(action).copy())
            states.append(core._extract_sim_state(result[0]))
            done_flags.append(bool(result[2]))
            return result

        env.step = step
        try:
            result = original_episode(**kwargs)
            (directory / "episode.json").write_text(json.dumps({"success": bool(result[0]),
                "steps": len(actions), "replans": context["replan"], "state_kind": "observed_proprio"}, indent=2))
            return result
        finally:
            env.step = original_step
            np.savez(directory / "steps.npz", action=np.asarray(actions), observed_state=np.asarray(states),
                     done=np.asarray(done_flags, dtype=bool))

    core._obs_to_model_input = model_input
    core._denormalize_action = denormalize
    core._predict_action_chunk = predict
    core.run_single_episode = episode


def main():
    own_repo = Path(__file__).resolve().parents[2]
    repo = Path(os.environ.get("FASTWAM_PLUS_TRACE_REPO", own_repo)).resolve()
    if not (repo / "experiments/libero_plus/eval_libero_plus_worker.py").is_file():
        raise FileNotFoundError(f"选定仓库缺少 Plus worker：{repo}")
    # 移除另一仓库的源码路径，确保实际导入被测代码。
    sys.path[:] = [p for p in sys.path if p and own_repo not in Path(p).resolve().parents and Path(p).resolve() != own_repo]
    sys.path[:0] = [str(repo / "experiments/libero"), str(repo / "src"), str(repo)]
    core = importlib.import_module("experiments.libero.eval_libero_single")
    worker = importlib.import_module("experiments.libero_plus.eval_libero_plus_worker")
    worker_file = Path(os.environ["LIBERO_PLUS_WORKER_RESULT_FILE"])
    output = worker_file.parent.parent / "traces" / f"worker{os.environ['LIBERO_PLUS_WORKER_ID']}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "source.json").write_text(json.dumps({"repo": str(repo), "eval_source": core.__file__,
                                                  "worker_source": worker.__file__}, indent=2))
    install_trace(core, output)
    worker.instantiate = compatible_instantiate(worker.instantiate)
    worker.main()


if __name__ == "__main__":
    main()
