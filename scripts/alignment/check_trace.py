"""验证记录器不改变随机流/多步更新，并验证比较器拒绝微小差异和不完整记录。"""
import argparse
import json
from pathlib import Path
import random
import tempfile
import types

import numpy as np
import torch
from omegaconf import OmegaConf

from _trace import install_trace, parameter_hash, rng_snapshot
from compare_traces import compare_directories


class NoiseScheduler:
    def add_noise(self, original_samples, noise, timestep):
        return original_samples + noise * timestep


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.device, self.torch_dtype = torch.device('cpu'), torch.float32
        self.train_video_scheduler = NoiseScheduler()
        self.train_action_scheduler = NoiseScheduler()

    def build_inputs(self, sample):
        return {'x': sample['x']}

    def training_loss(self, sample):
        x = self.build_inputs(sample)['x']
        for scheduler in [self.train_video_scheduler, self.train_action_scheduler]:
            x = scheduler.add_noise(x, torch.randn_like(x), torch.rand(()))
        loss = self.linear(x).square().mean()
        return loss, {'loss_action': float(loss.detach())}


class TinyTrainer:
    def __init__(self, model, *, cfg):
        self.model, self.cfg = model, cfg
        self.output_dir = cfg.output_dir
        self.train_dataset = list(range(16))
        self.global_step, self.max_steps, self.epoch, self.batch_in_epoch = 0, 4, 0, 0
        self.accelerator = types.SimpleNamespace(
            process_index=0, num_processes=1, distributed_type='CPU_TEST',
            state=types.SimpleNamespace(deepspeed_plugin=None),
            unwrap_model=lambda x: x, optimizer_step_was_skipped=False,
            backward=lambda loss: (loss / 2).backward(), sync_gradients=False,
        )
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=.9)
        self.losses = []

    def train(self):
        generator = torch.Generator().manual_seed(913)
        for step in range(self.max_steps):
            for micro in range(2):
                self.accelerator.sync_gradients = micro == 1
                self.batch_in_epoch += 1
                value, _ = self.model.training_loss({'x': torch.randn(2, 4, generator=generator)})
                self.losses.append(float(value.detach()))
                self.accelerator.backward(value)
                if self.accelerator.sync_gradients:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
            if step % 2 == 1:
                self.evaluate()

    def evaluate(self):
        with torch.no_grad():
            return self.model.training_loss({'x': torch.randn(2, 4)})


def run(root, traced):
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    root.mkdir()
    stats = root / 'stats.json'
    stats.write_text('{}')
    cfg = OmegaConf.create({'output_dir': str(root), 'model': {'_target_': 'TinyModel'},
                           'data': {'train': {'pretrained_norm_stats': str(stats)}},
                           'batch_size': 2, 'gradient_accumulation_steps': 2,
                           'num_workers': 0, 'learning_rate': 1e-3, 'seed': 42})
    class CurrentTrainer(TinyTrainer):
        pass
    if traced:
        install_trace(CurrentTrainer, root / 'trace')
    model = TinyModel()
    trainer = CurrentTrainer(model, cfg=cfg)
    trainer.train()
    return trainer.losses, parameter_hash(model), rng_snapshot(model.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='fastwam-trace-check-') as tmp:
        root = Path(tmp)
        plain = run(root / 'plain', False)
        left = run(root / 'left', True)
        right = run(root / 'right', True)
        assert plain == left == right, '记录器改变了loss、最终参数或随机流'
        a, b = root / 'left/trace', root / 'right/trace'
        result = compare_directories(a, b)
        assert result['passed'], result
        path = b / 'rank_00000.jsonl'
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        rows[1]['loss'] += 1e-10
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        assert not compare_directories(a, b)['passed']
        rows = [json.loads(line) for line in original.splitlines()]
        rows[2]['parameters_after_update'] = 'changed'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        assert not compare_directories(a, b)['passed']
        path.write_text('\n'.join(original.splitlines()[:-1]) + '\n')
        try:
            compare_directories(a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('不完整记录应失败')
        result.update(passive_trace_matches_plain_training=True, detects_1e_10_loss_difference=True,
                      detects_parameter_difference=True, rejects_incomplete_trace=True,
                      scope='CPU toy trainer；实际autograd/AdamW/随机流，非DeepSpeed集成测试')
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
