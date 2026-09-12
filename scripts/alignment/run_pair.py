"""在同一环境/拓扑顺序运行两repo的真实trainer，并严格比较所有rank的trace。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from check_configs import RECIPES, compose_train, launcher_argv


def replace_overrides(items, replacements):
    def key(item):
        return item.split('=', 1)[0].lstrip('+')
    result = [item for item in items if key(item) not in replacements]
    for name, value in replacements.items():
        result.append(name + '=' + str(value))
    return result


def source_manifest(repo):
    entries = {}
    for folder in ['src', 'configs', 'scripts']:
        for path in (repo / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.sh', '.yaml', '.json'} and '__pycache__' not in path.parts:
                entries[str(path.relative_to(repo))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if (repo / 'pyproject.toml').is_file():
        entries['pyproject.toml'] = hashlib.sha256((repo / 'pyproject.toml').read_bytes()).hexdigest()
    commit = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    return {'commit': commit.stdout.strip() if commit.returncode == 0 else None, 'sha256': entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-repo', type=Path, required=True)
    parser.add_argument('--target-repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--variant', choices=['uncond', 'joint', 'idm', 'all'], default='all')
    parser.add_argument('--role', choices=['both', 'reference', 'target'], default='both')
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--parameter-hash-every', type=int, default=1)
    parser.add_argument('--attention-backend', choices=['math', 'default'], default='math')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.steps < 2 or args.workers < 0 or args.eval_every < 0 or args.parameter_hash_every < 0:
        parser.error('Require steps>=2, workers/eval-every/parameter-hash-every>=0')
    for name in ['DATA_ROOT', 'TEXT_CACHE', 'MODEL_BASE', 'DATASET_STATS']:
        if not os.environ.get(name):
            parser.error(f'Set {name} explicitly so both runs use the same assets')
    target, reference, output = args.target_repo.resolve(), args.reference_repo.resolve(), args.output_root.resolve()
    if target == reference:
        parser.error('Target and reference must be different repositories')
    here = Path(__file__).resolve().parent
    env = dict(os.environ)
    # 清除参考版研究分支及checkpoint重定向，避免隐含配置/跨run覆盖。
    for key in list(env):
        if key.startswith('FASTWAM_') and key != 'FASTWAM_ENV':
            env.pop(key)
    env.update(FASTWAM_ACTION_ROPE_MODE='1d', CROSS_ROPE_DEBUG_INTERVAL='0',
               DIFFSYNTH_MODEL_BASE_PATH=env['MODEL_BASE'], DIFFSYNTH_SKIP_DOWNLOAD='true',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='42')
    for key in ['BATCH_SIZE', 'GRAD_ACCUM', 'NUM_WORKERS', 'MAX_STEPS', 'RESUME_STATE']:
        env.pop(key, None)
    env.setdefault('OMP_NUM_THREADS', '8')
    env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    env.setdefault('ACTION_DIT_CHECKPOINT', str(target / 'checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt'))
    if not args.dry_run:
        if env['DATASET_STATS'] == 'null':
            parser.error('For exact comparison, run prepare_stats.py once with the reference repo, then export DATASET_STATS to that generated JSON')
        for name in ['DATA_ROOT', 'TEXT_CACHE', 'MODEL_BASE']:
            if not Path(env[name]).is_dir():
                parser.error(f'{name} directory does not exist: {env[name]}')
        for name in ['DATASET_STATS', 'ACTION_DIT_CHECKPOINT']:
            if not Path(env[name]).is_file():
                parser.error(f'{name} file does not exist: {env[name]}')
    variants = list(RECIPES) if args.variant == 'all' else [args.variant]
    roles = ['reference', 'target'] if args.role == 'both' else [args.role]
    for variant in variants:
        raw = launcher_argv(target, variant, env)
        train_index = raw.index('scripts/train.py')
        launch_args = raw[2:train_index]
        # 使用同一accelerate配置；两侧DS JSON由最终trace再次严格比对。
        config_index = launch_args.index('--config_file') + 1
        launch_args[config_index] = str(target / launch_args[config_index])
        production_overrides = raw[train_index + 1:]
        target_cfg = compose_train(target, production_overrides)
        for role in roles:
            repo = reference if role == 'reference' else target
            role_output = output / variant / role
            overrides = replace_overrides(production_overrides, {
                'output_dir': role_output / 'train', 'resume': 'null', 'max_steps': args.steps,
                'num_workers': args.workers, 'eval_every': args.eval_every, 'save_every': 0,
                'log_every': 1, 'keep_last_ckpts': 0, 'wandb.enabled': 'false',
            })
            if role == 'reference':
                overrides += [f'model={RECIPES[variant][0]}',
                              'model.action_dit_config.action_rope_mode=1d',
                              'data.train.latent_cache_dir=null']
            # 采用同一组已确认的资产解析规则；仅配置位置不同不应改变预训练内容。
            overrides = replace_overrides(overrides, {
                'model.model_id': target_cfg.model.model_id,
                'model.tokenizer_model_id': target_cfg.model.tokenizer_model_id,
                'model.redirect_common_files': str(target_cfg.model.redirect_common_files).lower(),
            })
            # 在提交GPU工作前用所选repo自己的Hydra配置验证所有参数。
            cfg = compose_train(repo, overrides)
            assert cfg.model.action_dit_config.action_rope_mode == '1d'
            command = [args.python, '-m', 'accelerate.commands.launch', *launch_args,
                       str(here / 'train_with_trace.py'), '--repo', str(repo),
                       '--trace-dir', str(role_output / 'trace'),
                       '--parameter-hash-every', str(args.parameter_hash_every),
                       '--attention-backend', args.attention_backend, *overrides]
            print(f'[{variant}/{role}] {shlex.join(command)}', flush=True)
            if args.dry_run:
                continue
            role_output.mkdir(parents=True, exist_ok=True)
            node_rank = int(env.get('NODE_RANK', '0'))
            local_gpus = int(env.get('GPUS_PER_NODE', '16'))
            own_ranks = range(node_rank * local_gpus, (node_rank + 1) * local_gpus)
            if any((role_output / 'trace' / f'rank_{rank:05d}.jsonl').exists() for rank in own_ranks):
                parser.error(f'This node already has trace files; choose a new output root: {role_output}')
            plan = {'variant': variant, 'role': role, 'repo': str(repo), 'argv': command,
                    'source': source_manifest(repo), 'shared_python': args.python,
                    'assets': {key: env[key] for key in ['DATA_ROOT', 'TEXT_CACHE', 'MODEL_BASE', 'DATASET_STATS', 'ACTION_DIT_CHECKPOINT']}}
            with (role_output / f'plan_node_{node_rank}.json').open('x') as stream:
                stream.write(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
            child_env = {**env, 'PYTHONPATH': str(repo / 'src') + os.pathsep + str(repo),
                         'REPO_ROOT': str(repo), 'OUTPUT_DIR': str(role_output / 'train')}
            subprocess.run(command, cwd=repo, env=child_env, check=True)
        if args.role == 'both' and not args.dry_run and int(env.get('NODE_RANK', '0')) == 0:
            subprocess.run([args.python, str(here / 'compare_traces.py'),
                            str(output / variant / 'reference/trace'), str(output / variant / 'target/trace'),
                            '--output', str(output / variant / 'comparison.json')], check=True)


if __name__ == '__main__':
    main()
