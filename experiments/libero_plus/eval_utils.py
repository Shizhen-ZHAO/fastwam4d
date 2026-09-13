"""Plus 专用输入检查；模型推理仍复用现有 FastWAM 实现。"""
import logging
import os
from pathlib import Path

from omegaconf import OmegaConf

FACTORIES = {
    "uncond": "fastwam.runtime.create_fastwam",
    "joint": "fastwam.runtime.create_fastwam_joint",
    "idm": "fastwam.runtime.create_fastwam_idm",
}


def expanded_path(value):
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def validate_eval_config(cfg):
    factory = str(cfg.model._target_)
    if factory not in FACTORIES.values():
        raise ValueError(f"Plus 入口只支持 uncond/joint/idm：{factory}")
    if str(cfg.model.action_dit_config.get("action_rope_mode")) != "1d":
        raise ValueError("此入口要求 action RoPE=1d，必须与训练 checkpoint 一致")
    if cfg.model.video_dit_config.get("rope_mode", "3d") != "3d":
        raise ValueError("此入口使用原始 3d video RoPE")
    if cfg.model.get("compile_training_denoise", False) or cfg.EVALUATION.get("compile_action_infer", False):
        raise ValueError("对齐入口要求 eager：两个 compile 开关必须为 false")
    if not cfg.model.load_text_encoder:
        raise ValueError("Plus 任务文本需要本地 T5，load_text_encoder 必须为 true")
    if int(cfg.EVALUATION.env_num) != 1 or int(cfg.EVALUATION.num_trials) < 1:
        raise ValueError("要求 env_num=1 且 num_trials>0")
    horizon = cfg.EVALUATION.get("action_horizon") or int(cfg.data.train.num_frames) - 1
    if not 0 < int(cfg.EVALUATION.replan_steps) <= int(horizon):
        raise ValueError("要求 0 < replan_steps <= action_horizon")
    if int(cfg.EVALUATION.num_steps_wait) < 0 or int(cfg.EVALUATION.num_inference_steps) < 1:
        raise ValueError("等待步数须非负，推理步数须为正")
    if cfg.seed is None:
        raise ValueError("对齐评测必须显式设置 seed")
    if int(cfg.MULTIRUN.num_gpus) < 1 or int(cfg.MULTIRUN.max_tasks_per_gpu) < 1:
        raise ValueError("num_gpus 和 max_tasks_per_gpu 必须为正")
    if str(cfg.EVALUATION.device) not in {"cuda", "cuda:0", "cpu"}:
        raise ValueError("worker 只看到一张卡，device 应为 cuda 或 cuda:0")
    if os.environ.get("FASTWAM_CORRUPT_TARGET", "none") != "none" or float(os.environ.get("FASTWAM_CORRUPT_SIGMA", "0")) != 0:
        raise ValueError("1d 对齐评测不启用 corruption")
    return next(key for key, value in FACTORIES.items() if value == factory)


def validate_model_assets(cfg):
    """按现有 loader 的路径规则提前检查本地 T5/VAE/tokenizer。"""
    base = expanded_path(os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./checkpoints"))
    model_root = base / str(cfg.model.model_id)
    tokenizer = base / str(cfg.model.tokenizer_model_id) / "google/umt5-xxl"
    if bool(cfg.model.redirect_common_files):
        common = base / "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
        vae, text = common / "Wan2.2_VAE.safetensors", common / "models_t5_umt5-xxl-enc-bf16.safetensors"
    else:
        vae, text = model_root / "Wan2.2_VAE.pth", model_root / "models_t5_umt5-xxl-enc-bf16.pth"
    for path in (vae, text):
        if not path.is_file():
            raise FileNotFoundError(f"本地模型组件不存在：{path}")
    if not tokenizer.is_dir() or not any(tokenizer.iterdir()):
        raise FileNotFoundError(f"本地 tokenizer 不存在或为空：{tokenizer}")
    if not cfg.model.skip_dit_load_from_pretrain and not list(model_root.glob("diffusion_pytorch_model*.safetensors")):
        raise FileNotFoundError(f"Wan DiT 初始化文件不存在：{model_root}")
    if cfg.model.action_dit_pretrained_path and not expanded_path(cfg.model.action_dit_pretrained_path).is_file():
        raise FileNotFoundError(f"ActionDiT 初始化权重不存在：{cfg.model.action_dit_pretrained_path}")
    return {"vae": str(vae), "text_encoder": str(text), "tokenizer": str(tokenizer)}


def resolve_stats_path(cfg):
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    if explicit is not None:
        path = expanded_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"指定 stats 不存在：{path}")
        return path
    for parent in list(expanded_path(cfg.ckpt).parents)[:4]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("没有找到训练 stats，请设置 DATASET_STATS_PATH")


def validate_checkpoint_config(cfg):
    """只校验可确认的训练字段；缺失或未解析字段明确记为未知。"""
    ckpt = expanded_path(cfg.ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"训练 checkpoint 不存在：{ckpt}")
    result = {"path": None, "checked": {}, "unknown": []}
    candidate = next((p / "config.yaml" for p in list(ckpt.parents)[:4]
                      if (p / "config.yaml").is_file()), None)
    fields = {
        "model._target_": str(cfg.model._target_),
        "model.action_dit_config.action_rope_mode": "1d",
        "model.video_dit_config.rope_mode": "3d",
    }
    if candidate is None:
        result["unknown"] = list(fields)
        logging.warning("checkpoint 附近无训练 config.yaml，模式未核验：%s", ckpt)
        return result
    result["path"] = str(candidate)
    # 不用当前环境解析训练时的 ${oc.env:...}，否则会误将历史 3d 判成当前 1d。
    data = OmegaConf.to_container(OmegaConf.load(candidate), resolve=False)
    for key, expected in fields.items():
        value = data
        for component in key.split("."):
            value = value.get(component) if isinstance(value, dict) else None
        if value is None or "${" in str(value):
            result["unknown"].append(key)
        elif str(value) != expected:
            raise ValueError(f"checkpoint 配置冲突：{candidate}: {key}={value!r}, eval={expected!r}")
        else:
            result["checked"][key] = value
    if result["unknown"]:
        logging.warning("训练配置中未核验字段：%s", result["unknown"])
    return result


def get_task_init_states(task_suite, task_id):
    import torch
    original = torch.load

    def load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = load_compat
    try:
        return task_suite.get_task_init_states(task_id)
    finally:
        torch.load = original


def repeat_initial_states(initial_states, num_trials):
    if num_trials < 1 or len(initial_states) == 0:
        raise ValueError("初始状态不能为空，num_trials 必须大于 0")
    # 保持参考的顺序循环；不要求 numpy/tensor 具有 list.extend。
    return [initial_states[i % len(initial_states)] for i in range(num_trials)]
