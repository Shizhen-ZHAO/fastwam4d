"""实际launcher DRY_RUN + 两repo Hydra配置对照；不启动GPU。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

RECIPES = {
    'uncond': ('fastwam_3d', 4, 2),
    'joint': ('fastwam_joint_ppu', 8, 1),
    'idm': ('fastwam_idm', 8, 1),
}


def launcher_argv(repo, variant, env):
    result = subprocess.run(
        ['bash', str(repo / f'scripts/train_libero_{variant}_16gpu.sh')],
        env={**env, 'REPO_ROOT': str(repo), 'DRY_RUN': '1'},
        capture_output=True, text=True, check=True,
    )
    return shlex.split(result.stdout.splitlines()[-1])


def compose_train(repo, overrides):
    with initialize_config_dir(config_dir=str(repo / 'configs'), version_base=None):
        cfg = compose(config_name='train', overrides=overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-repo', required=True, type=Path)
    parser.add_argument('--target-repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target, reference = args.target_repo.resolve(), args.reference_repo.resolve()
    env = dict(os.environ)
    for key in ['BATCH_SIZE', 'GRAD_ACCUM', 'NUM_WORKERS', 'NNODES', 'GPUS_PER_NODE', 'NODE_RANK',
                'MASTER_ADDR', 'MASTER_PORT', 'RUN_ID', 'OUTPUT_DIR', 'RESUME_STATE', 'MAX_STEPS']:
        env.pop(key, None)
    keys = ['batch_size', 'gradient_accumulation_steps', 'num_workers', 'learning_rate',
            'weight_decay', 'num_epochs', 'seed', 'mixed_precision', 'lr_scheduler_type',
            'max_grad_norm', 'log_every', 'eval_every', 'save_every', 'eval_num_inference_steps',
            'keep_last_ckpts', 'model._target_', 'model.mot_checkpoint_mixed_attn',
            'model.action_dit_config', 'model.video_dit_config', 'model.action_scheduler',
            'model.video_scheduler', 'model.loss']
    results = []
    for variant, (ref_model, batch, gas) in RECIPES.items():
        for topology in [{}, {'NNODES': '2', 'GPUS_PER_NODE': '8', 'NODE_RANK': '1',
                             'MASTER_ADDR': '192.0.2.1', 'RUN_ID': 'config-check'}]:
            command = launcher_argv(target, variant, {**env, **topology})
            target_cfg = compose_train(target, command[command.index('scripts/train.py') + 1:])
            ref_cfg = compose_train(reference, [
                f'task=libero_{variant}_2cam224_1e-4', f'model={ref_model}',
                f'batch_size={batch}', f'gradient_accumulation_steps={gas}', 'num_workers=8',
                'model.action_dit_config.action_rope_mode=1d',
                'data.train.latent_cache_dir=null',
            ])
            left, right = {}, {}
            for key in keys:
                a, b = OmegaConf.select(target_cfg, key), OmegaConf.select(ref_cfg, key)
                a = OmegaConf.to_container(a, resolve=True) if OmegaConf.is_config(a) else a
                b = OmegaConf.to_container(b, resolve=True) if OmegaConf.is_config(b) else b
                assert a == b, (variant, key, a, b)
                left[key], right[key] = a, b
            assert target_cfg.model.compile_training_denoise is False
            data_left = OmegaConf.to_container(target_cfg.data.train, resolve=True)
            data_right = OmegaConf.to_container(ref_cfg.data.train, resolve=True)
            for data in [data_left, data_right]:
                assert data.get('latent_cache_dir') is None
                data.setdefault('latent_cache_dir', None)
                # 参考版默认读取文本cache，目标版将同一行为显式写成配置项。
                data.setdefault('use_text_embed_cache', not data.pop('skip_text_embeds', False))
                for path_key in ['dataset_dirs', 'text_embedding_cache_dir', 'pretrained_norm_stats']:
                    data.pop(path_key, None)
            assert data_left == data_right, (variant, 'data preprocessing config', data_left, data_right)
            left['data_preprocessing'] = data_left
            assert target_cfg.model.mot_checkpoint_mixed_attn is False
            assert target_cfg.model.action_dit_config.action_rope_mode == '1d'
            assert command[command.index('--num_processes') + 1] == '16'
            assert 16 * target_cfg.batch_size * target_cfg.gradient_accumulation_steps == 128
            results.append({'variant': variant, 'topology': topology or '1x16', 'effective': left})
    payload = json.dumps({'scope': '真实Hydra配置；资产路径单独核对，未运行训练', 'cases': results}, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
