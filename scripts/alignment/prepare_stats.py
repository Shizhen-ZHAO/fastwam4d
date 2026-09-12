"""使用所选repo的原始数据构造流程计算stats一次；不加载模型，不修改参考源码。"""
import argparse
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-repo', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('DATA_ROOT'):
        parser.error('Source scripts/libero_cluster_paths.sh first')
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        parser.error('Run once with plain python, before starting distributed training')
    repo, output = args.reference_repo.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f'Output already exists; reuse it or choose a new filename: {output}')
    sys.path[:0] = [str(repo / 'src'), str(repo)]
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    import fastwam
    from fastwam.utils import misc
    from fastwam.utils.pytorch_utils import set_global_seed
    from fastwam.utils.config_resolvers import register_default_resolvers

    assert Path(fastwam.__file__).resolve().is_relative_to(repo / 'src'), fastwam.__file__
    register_default_resolvers()
    dirs = ','.join(str(Path(os.environ['DATA_ROOT']) / f'libero_{suite}_no_noops_lerobot')
                    for suite in ['spatial', 'object', 'goal', '10'])
    with initialize_config_dir(config_dir=str(repo / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[
            'task=libero_uncond_2cam224_1e-4', f'data.train.dataset_dirs=[{dirs}]',
            '+data.train.pretrained_norm_stats=null', 'data.train.latent_cache_dir=null',
        ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare_stats_', dir=output.parent) as work:
        misc.register_work_dir(work)
        set_global_seed(int(cfg.seed))
        # 实际 RobotVideoDataset 构造函数调用 BaseLerobotDataset.get_dataset_stats。
        # 不调用 __getitem__，无需加载图像视频、文本cache、VAE、ActionDiT。
        dataset = instantiate(cfg.data.train)
        payload = (Path(work) / 'dataset_stats.json').read_bytes()
        with output.open('xb') as stream:
            stream.write(payload)
        print(f'Saved stats for {len(dataset)} samples: {output}')
    print(f'export DATASET_STATS={output}')


if __name__ == '__main__':
    main()
