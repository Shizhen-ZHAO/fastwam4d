"""从实际评测 launcher 展开命令，比较三种模型的 Hydra 评测配置。"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from check_configs import RECIPES


def compose_eval(repo, overrides):
    with initialize_config_dir(config_dir=str(repo / 'configs'), version_base=None):
        return compose(config_name='sim_libero', overrides=overrides)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-repo', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[2]
    reference = args.reference_repo.resolve()
    # 本检查只解析路径，不访问这些示例资产。
    env = {**os.environ, 'REPO_ROOT': str(target), 'DRY_RUN': '1',
           'CKPT': '/not-loaded/weights.pt', 'NUM_TRIALS': '50', 'NUM_EVAL_GPUS': '16'}
    cases = []
    for variant, (ref_model, _, _) in RECIPES.items():
        for mode in ['single', 'manager']:
            result = subprocess.run(['bash', str(target / 'scripts/eval_libero.sh'), variant],
                                    env={**env, 'EVAL_MODE': mode}, text=True, capture_output=True, check=True)
            command = shlex.split(result.stdout.splitlines()[-1])
            expected = 'eval_libero_single.py' if mode == 'single' else 'run_libero_manager.py'
            assert command[1].endswith(expected)
            overrides = command[2:]
            left = compose_eval(target, overrides)
            right = compose_eval(reference, [item for item in overrides
                                            if not item.startswith('EVALUATION.compile_action_infer=')]
                                 + [f'model={ref_model}'])
            keys = ['seed', 'mixed_precision', 'model._target_', 'model.video_dit_config',
                    'model.action_dit_config', 'model.video_scheduler', 'model.action_scheduler',
                    'model.model_id', 'model.tokenizer_model_id', 'model.redirect_common_files',
                    'model.load_text_encoder', 'model.skip_dit_load_from_pretrain',
                    'model.action_dit_pretrained_path', 'data.train.processor',
                    'data.train.video_size', 'data.train.num_frames', 'data.train.action_video_freq_ratio']
            keys += ['EVALUATION.' + key for key in left.EVALUATION
                     if key not in ['output_dir', 'compile_action_infer']]
            checked = {}
            for key in keys:
                a, b = OmegaConf.select(left, key), OmegaConf.select(right, key)
                a = OmegaConf.to_container(a, resolve=True) if OmegaConf.is_config(a) else a
                b = OmegaConf.to_container(b, resolve=True) if OmegaConf.is_config(b) else b
                assert a == b, (variant, mode, key, a, b)
                checked[key] = a
            assert left.EVALUATION.compile_action_infer is False
            assert right.EVALUATION.infer_increment_seed is False
            assert right.EVALUATION.skip_unused_render is False
            assert right.EVALUATION.save_video is True
            assert right.MULTIRUN.trials_per_shard == 0
            cases.append({'variant': variant, 'entry': mode, 'effective': checked})
    payload = json.dumps({'scope': 'actual launcher and Hydra; no assets/GPU/simulator', 'cases': cases},
                         ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == '__main__':
    main()
