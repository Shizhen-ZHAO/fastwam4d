"""仅用于对照运行的被动记录器；两repo共用，不改参考源码或替换训练循环。"""
import copy
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import random

import numpy as np
import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy()
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(memoryview(raw)).hexdigest()}
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return str(value)


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        # 每次仅搬运一个张量；不保留整个CPU state_dict，且不进行任何参数修改。
        item = {'name': name, 'requires_grad': parameter.requires_grad,
                'tensor': fingerprint(parameter)}
        digest.update(json.dumps(item, sort_keys=True).encode())
    return digest.hexdigest()


def rng_snapshot(device):
    result = {
        'python': hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        'numpy': hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        'torch_cpu': fingerprint(torch.get_rng_state()),
    }
    # 只读取当前rank使用的GPU；不在其他卡创建CUDA上下文。
    if torch.device(device).type == 'cuda':
        result['torch_cuda'] = fingerprint(torch.cuda.get_rng_state(device))
    return result


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'accelerate', 'deepspeed', 'numpy', 'datasets',
                 'av', 'torchcodec', 'pyarrow', 'hydra-core', 'pillow', 'transformers', 'einops']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {'python': platform.python_version(), 'packages': packages,
              'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
              'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
              'deterministic': torch.are_deterministic_algorithms_enabled(),
              'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
              'tf32_cudnn': torch.backends.cudnn.allow_tf32,
              'sdp_math': torch.backends.cuda.math_sdp_enabled(),
              'sdp_flash': torch.backends.cuda.flash_sdp_enabled(),
              'sdp_mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled()}
    if torch.cuda.is_available():
        result['gpu_name'] = torch.cuda.get_device_name(torch.cuda.current_device())
        result['gpu_capability'] = list(torch.cuda.get_device_capability(torch.cuda.current_device()))
    return result


class TrainingTrace:
    def __init__(self, trainer, model, output, parameter_every):
        from omegaconf import OmegaConf

        self.trainer, self.model = trainer, model
        self.parameter_every = parameter_every
        self.active, self.pending = False, None
        self.microbatch = 0
        self.device = model.device
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        self.stream = (output / f'rank_{trainer.accelerator.process_index:05d}.jsonl').open('x')
        cfg = OmegaConf.to_container(trainer.cfg, resolve=True)
        model_cfg = dict(cfg['model'])
        for key in ['model_id', 'tokenizer_model_id', 'redirect_common_files', 'action_dit_pretrained_path']:
            model_cfg.pop(key, None)
        model_cfg.setdefault('compile_training_denoise', False)
        if hasattr(model, 'video_cond_noise_prob'):
            model_cfg['video_cond_noise_prob'] = float(model.video_cond_noise_prob)
        data_cfg = copy.deepcopy(cfg['data'])
        for dataset_cfg in data_cfg.values():
            if isinstance(dataset_cfg, dict):
                dataset_cfg.setdefault('use_text_embed_cache', not dataset_cfg.pop('skip_text_embeds', False))
                for key in ['dataset_dirs', 'text_embedding_cache_dir', 'pretrained_norm_stats']:
                    dataset_cfg.pop(key, None)
        data_cfg['train'].setdefault('latent_cache_dir', None)
        contract = {key: cfg.get(key) for key in [
            'batch_size', 'gradient_accumulation_steps', 'num_workers', 'learning_rate', 'weight_decay',
            'num_epochs', 'seed', 'mixed_precision', 'lr_scheduler_type', 'max_grad_norm', 'eval_every',
            'save_every', 'log_every', 'eval_num_inference_steps', 'keep_last_ckpts',
        ]}
        contract.update(model=model_cfg, data=data_cfg, max_steps=trainer.max_steps,
                        dataset_length=len(trainer.train_dataset), world_size=trainer.accelerator.num_processes)
        stats = trainer.cfg.data.train.get('pretrained_norm_stats')
        stats_path = Path(str(stats)) if stats else Path(trainer.output_dir) / 'dataset_stats.json'
        ds_plugin = getattr(trainer.accelerator.state, 'deepspeed_plugin', None)
        scales = getattr(getattr(model, 'vae', None), 'scale', [])
        self.write({'event': 'start', 'rank': trainer.accelerator.process_index,
                    'initial_global_step': trainer.global_step,
                    'contract': fingerprint(contract), 'environment': environment(),
                    'distributed_type': str(trainer.accelerator.distributed_type),
                    'deepspeed_config': fingerprint(ds_plugin.deepspeed_config) if ds_plugin else None,
                    'initial_parameters': parameter_hash(model),
                    'parameter_order': [name for name, _ in model.named_parameters()],
                    'vae_scale_effective': fingerprint([x.to(dtype=model.torch_dtype) for x in scales]),
                    'rng': rng_snapshot(self.device), 'dataset_stats_sha256': file_hash(stats_path),
                    'parameter_hash_every': parameter_every,
                    'model_module': str(type(model).__module__), 'resolved_config': cfg})

    def write(self, item):
        self.stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        self.stream.flush()

    def finish_microbatch(self, updated):
        if self.pending is None:
            raise RuntimeError('Missing pending loss record')
        record = self.pending
        record['update_boundary'] = updated
        record['optimizer_step_was_skipped'] = bool(self.trainer.accelerator.optimizer_step_was_skipped) if updated else False
        record['rng_after_backward'] = rng_snapshot(self.device)
        if updated and self.parameter_every and (self.trainer.global_step + 1) % self.parameter_every == 0:
            record['parameters_after_update'] = parameter_hash(self.model)
        self.write(record)
        self.pending = None

    def install(self):
        model, trainer = self.model, self.trainer
        original_loss = model.training_loss
        original_inputs = model.build_inputs

        @functools.wraps(original_inputs)
        def inputs(*args, **kwargs):
            value = original_inputs(*args, **kwargs)
            if self.active and self.pending is not None:
                self.pending['inputs'] = fingerprint(value)
            return value

        @functools.wraps(original_loss)
        def loss(sample, *args, **kwargs):
            if not self.active:
                return original_loss(sample, *args, **kwargs)
            if self.pending is not None:
                raise RuntimeError('Previous microbatch was not finalized')
            self.microbatch += 1
            self.pending = {'event': 'microbatch', 'microbatch': self.microbatch,
                            'global_step_before': trainer.global_step, 'epoch': trainer.epoch,
                            'batch_in_epoch': trainer.batch_in_epoch,
                            'lr_before': [float(group['lr']) for group in trainer.optimizer.param_groups],
                            'sample': fingerprint(sample), 'rng_before': rng_snapshot(self.device),
                            'noise_calls': []}
            value, metrics = original_loss(sample, *args, **kwargs)
            self.pending.update(loss=float(value.detach()), metrics={k: float(v) for k, v in metrics.items()},
                                rng_after_forward=rng_snapshot(self.device))
            return value, metrics

        model.build_inputs, model.training_loss = inputs, loss
        for role in ['video', 'action']:
            scheduler = getattr(model, f'train_{role}_scheduler')
            original_noise = scheduler.add_noise

            def make_noise_wrapper(bound, name):
                @functools.wraps(bound)
                def add_noise(original_samples, noise, timestep):
                    value = bound(original_samples, noise, timestep)
                    if self.active and self.pending is not None:
                        self.pending['noise_calls'].append({'role': name, 'noise': fingerprint(noise),
                                                           'timestep': fingerprint(timestep), 'noised': fingerprint(value)})
                    return value
                return add_noise

            scheduler.add_noise = make_noise_wrapper(original_noise, role)

        original_backward = trainer.accelerator.backward

        @functools.wraps(original_backward)
        def backward(*args, **kwargs):
            result = original_backward(*args, **kwargs)
            if self.active and not trainer.accelerator.sync_gradients:
                self.finish_microbatch(updated=False)
            return result

        trainer.accelerator.backward = backward
        original_step = trainer.optimizer.step

        @functools.wraps(original_step)
        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            if self.active:
                self.finish_microbatch(updated=True)
            return result

        trainer.optimizer.step = step

    def close(self, completed):
        try:
            if completed:
                if self.pending is not None:
                    raise RuntimeError('Unfinished microbatch at end of training')
                self.write({'event': 'end', 'completed': True, 'microbatches': self.microbatch,
                            'global_step': self.trainer.global_step,
                            'final_parameters': parameter_hash(self.model), 'rng': rng_snapshot(self.device)})
        finally:
            self.stream.close()


def install_trace(trainer_class, trace_dir, parameter_every=1):
    """包装实际trainer的入口；不代替loss、backward、optimizer或数据迭代实现。"""
    original_init, original_train, original_evaluate = trainer_class.__init__, trainer_class.train, trainer_class.evaluate

    @functools.wraps(original_init)
    def init(self, model, *args, **kwargs):
        original_init(self, model, *args, **kwargs)
        self._alignment_trace = TrainingTrace(self, self.accelerator.unwrap_model(self.model), trace_dir, parameter_every)
        self._alignment_trace.install()

    @functools.wraps(original_train)
    def train(self, *args, **kwargs):
        trace = self._alignment_trace
        trace.active = True
        completed = False
        try:
            result = original_train(self, *args, **kwargs)
            completed = self.global_step == self.max_steps
            return result
        finally:
            trace.active = False
            trace.close(completed)

    @functools.wraps(original_evaluate)
    def evaluate(self, *args, **kwargs):
        trace = self._alignment_trace
        previous = trace.active
        trace.active = False
        try:
            return original_evaluate(self, *args, **kwargs)
        finally:
            trace.active = previous

    trainer_class.__init__, trainer_class.train, trainer_class.evaluate = init, train, evaluate
