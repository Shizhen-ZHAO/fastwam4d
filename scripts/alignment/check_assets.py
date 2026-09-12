"""检查统一 LIBERO 路径约定；只读文件，不加载模型，不保证文件内容/数值正确。"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'text'], default='train')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    required = ['MODEL_BASE', 'MODEL_ID', 'REDIRECT_COMMON_FILES']
    if args.mode != 'text':
        required += ['DATASET_STATS']
    if args.mode == 'train':
        required += ['DATA_ROOT', 'TEXT_CACHE', 'ACTION_DIT_CHECKPOINT']
    if args.mode in ['text', 'eval']:
        required += ['TOKENIZER_MODEL_ID']
    if args.mode == 'eval':
        required += ['CKPT']
    for name in required:
        if not os.environ.get(name):
            parser.error(f'Missing {name}; source scripts/libero_cluster_paths.sh and set remaining assets first')
    report = {'mode': args.mode, 'scope': 'file existence, basic JSON structure, prompt cache coverage; no GPU/model loading',
              'paths': {key: os.environ[key] for key in required}, 'checks': [], 'errors': []}

    def require(path):
        path = Path(path)
        ok = path.is_file() and path.stat().st_size > 0
        report['checks'].append({'file': str(path), 'nonempty': ok})
        if not ok:
            report['errors'].append(f'Missing/empty file: {path}')
        return ok

    base = Path(os.environ['MODEL_BASE'])
    wan = base / os.environ['MODEL_ID']
    redirect = os.environ['REDIRECT_COMMON_FILES'].lower()
    if redirect not in ['true', 'false']:
        parser.error('REDIRECT_COMMON_FILES must be true or false')
    common = base / 'DiffSynth-Studio/Wan-Series-Converted-Safetensors'
    if args.mode != 'text':
        require(common / 'Wan2.2_VAE.safetensors' if redirect == 'true' else wan / 'Wan2.2_VAE.pth')
        stats_value = os.environ['DATASET_STATS']
        stats_path = None if stats_value == 'null' else Path(stats_value)
        if stats_path is None and args.mode == 'eval':
            # 与 eval_libero_single._resolve_dataset_stats_path 相同的上级搜索范围。
            candidates = [parent / 'dataset_stats.json' for parent in list(Path(os.environ['CKPT']).resolve().parents)[:4]]
            stats_path = next((path for path in candidates if path.is_file()), None)
            if stats_path is None:
                report['errors'].append('No dataset_stats.json beside checkpoint; set DATASET_STATS explicitly')
        report['stats_source'] = str(stats_path) if stats_path is not None else 'calculate from training data at startup'
        if stats_path is not None and require(stats_path):
            try:
                stats = json.loads(stats_path.read_text())
                for kind, dim in [('action', 7), ('state', 8)]:
                    for key in ['global_min', 'global_max']:
                        values = stats[kind]['default'][key]
                        assert len(values) == dim, (kind, key, len(values), dim)
                report['stats_sha256'] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                report['errors'].append(f'Invalid default 7D action/8D state min/max stats: {exc}')
    if args.mode == 'train':
        shards = sorted(wan.glob('diffusion_pytorch_model*.safetensors'))
        if not shards:
            report['errors'].append(f'No diffusion_pytorch_model*.safetensors under {wan}')
        for shard in shards:
            require(shard)
        index = wan / 'diffusion_pytorch_model.safetensors.index.json'
        if index.is_file():
            for filename in sorted(set(json.loads(index.read_text())['weight_map'].values())):
                require(wan / filename)
        require(os.environ['ACTION_DIT_CHECKPOINT'])
        # 从实际源码取 prompt 模板，保持与 dataset 一致。
        repo = Path(__file__).resolve().parents[2]
        tree = ast.parse((repo / 'src/fastwam/datasets/lerobot/robot_video_dataset.py').read_text())
        prompt_template = next(ast.literal_eval(node.value) for node in tree.body
                               if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_PROMPT' for t in node.targets))
        prompts = set()
        for suite in ['spatial', 'object', 'goal', '10']:
            dataset = Path(os.environ['DATA_ROOT']) / f'libero_{suite}_no_noops_lerobot'
            require(dataset / 'meta/info.json')
            require(dataset / 'meta/episodes.jsonl')
            tasks = dataset / 'meta/tasks.jsonl'
            if require(tasks):
                for line in tasks.read_text().splitlines():
                    if line.strip():
                        prompts.add(prompt_template.format(task=str(json.loads(line)['task'])))
        for prompt in sorted(prompts):
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            require(Path(os.environ['TEXT_CACHE']) / f'{digest}.t5_len128.wan22ti2v5b.pt')
        report['unique_prompt_count'] = len(prompts)
    if args.mode in ['text', 'eval']:
        require(common / 'models_t5_umt5-xxl-enc-bf16.safetensors' if redirect == 'true' else wan / 'models_t5_umt5-xxl-enc-bf16.pth')
        tokenizer = base / os.environ['TOKENIZER_MODEL_ID'] / 'google/umt5-xxl'
        require(tokenizer / 'tokenizer_config.json')
        if not any((tokenizer / name).is_file() for name in ['spiece.model', 'tokenizer.json']):
            report['errors'].append(f'Missing tokenizer vocabulary under {tokenizer}')
    if args.mode == 'eval':
        require(os.environ['CKPT'])
    report['passed'] = not report['errors']
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
