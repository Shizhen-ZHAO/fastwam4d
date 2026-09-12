"""用所选repo的真实run_training运行并记录；参考repo无需修改或安装为editable。"""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--trace-dir', type=Path, required=True)
    parser.add_argument('--parameter-hash-every', type=int, default=1)
    parser.add_argument('--attention-backend', choices=['math', 'default'], default='math')
    args, hydra_args = parser.parse_known_args()
    if args.parameter_hash_every < 0:
        parser.error('--parameter-hash-every must be >= 0')
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'src'))
    sys.argv = [sys.argv[0], *hydra_args]
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    import contextlib
    import hydra
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import fastwam
    from fastwam import runtime
    from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
    from fastwam.utils.config_resolvers import register_default_resolvers
    from _trace import install_trace

    assert Path(fastwam.__file__).resolve().is_relative_to(repo / 'src'), fastwam.__file__
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    register_default_resolvers()

    # 对照时坏样本必须明确报错，不能通过随机换样本掩盖数据读取差异。
    RobotVideoDataset.__getitem__ = RobotVideoDataset._get
    install_trace(runtime.Wan22Trainer, args.trace_dir, args.parameter_hash_every)

    @hydra.main(config_path=str(repo / 'configs'), config_name='train', version_base='1.3')
    def run(cfg):
        if cfg.resume:
            raise ValueError('Paired initial-training verification requires resume=null')
        if cfg.model.action_dit_config.action_rope_mode != '1d':
            raise ValueError('Alignment requires action RoPE=1d')
        if cfg.model.mot_checkpoint_mixed_attn or cfg.model.get('compile_training_denoise', False):
            raise ValueError('Alignment baseline requires eager execution and checkpoint=false')
        context = sdpa_kernel(SDPBackend.MATH) if args.attention_backend == 'math' else contextlib.nullcontext()
        with context:
            runtime.run_training(cfg)
        # 每个rank的最终trace均已写完后才结束进程，便于rank0立即比较多机结果。
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    run()


if __name__ == '__main__':
    main()
