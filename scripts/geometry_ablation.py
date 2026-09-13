#!/usr/bin/env python3
"""Path preflight and routing only; execute unchanged official train/eval entrypoints."""
import argparse
import os
from pathlib import Path
import runpy
import sys
from omegaconf import OmegaConf
from libero_track4world import load_paths,print_paths,validate_paths,validate_model_assets,apply_path_overrides

ROOT=Path(__file__).resolve().parents[1]


def plus_environment(paths):
    env = dict(os.environ)
    dry = env.get('DRY_RUN') == '1'
    for key, setting in (('LIBERO_PLUS_ROOT', 'libero_plus_repo'),
                         ('LIBERO_PLUS_ASSETS_DIR', 'libero_plus_assets')):
        value = env.get(key) or paths.get(setting)
        if not value and not dry:
            raise ValueError(f'Set paths.{setting} or {key}; ordinary LIBERO is not a Plus fallback')
        if value and not dry and not Path(value).expanduser().is_dir():
            raise FileNotFoundError(f'{key}: {value}')
        env[key] = str(value or f'/configure/{setting}')
    env.update(TASK_CONFIG='libero_geometry_ablation', EVAL_MODE='uncond',
        MODEL_BASE=str(paths['model_base']), MODEL_ID=str(paths['model_id']),
        TOKENIZER_MODEL_ID=str(paths['tokenizer_model_id']),
        REDIRECT_COMMON_FILES=str(paths['redirect_common_files']).lower(),
        DIFFSYNTH_MODEL_BASE_PATH=str(paths['model_base']),
        PYTHON_BIN=sys.executable, HF_ENDPOINT=str(paths['hf_endpoint']))
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(ROOT),
        str(paths['track4world_extra_pythonpath']),env.get('PYTHONPATH','')])
    defaults = dict(NUM_GPUS='1', MAX_TASKS_PER_GPU='1', NUM_TRIALS='1',
        OUTPUT_BASE_DIR=str(Path(paths['output_root'])/'libero_plus'),
        DATASET_STATS_PATH=str(paths['dataset_stats']), SHARE_BACKUP='false', OSS_BACKUP='false')
    for key,value in defaults.items():
        env.setdefault(key,value)
    return env


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','eval','eval-plus'])
    p.add_argument('--paths',default=str(ROOT/'configs/paths/libero_track4world_local.yaml'))
    p.add_argument('--manager',action='store_true')
    args,overrides=p.parse_known_args()
    paths=apply_path_overrides(load_paths(args.paths),overrides)
    if Path(args.paths).resolve().parent != (ROOT/'configs/paths').resolve():
        raise ValueError('Place your YAML in configs/paths')
    print_paths(paths)
    switches=[x.split('=',1)[1].lower() for x in overrides if x.startswith('geometry_enabled=')]
    if switches and switches[-1] not in ('true','false'):
        raise ValueError('geometry_enabled must be true or false')
    enabled=not switches or switches[-1]=='true'
    if args.mode == 'eval-plus':
        env = plus_environment(paths)
        command=['bash',str(ROOT/'experiments/libero_plus/run_eval.sh'),
                 f'+paths={Path(args.paths).stem}',*overrides]
        sys.stdout.flush()
        os.execvpe('bash',command,env)
    if enabled:validate_paths(paths,cache_required=args.mode=='train')
    validate_model_assets(paths,evaluation=args.mode=='eval')
    os.environ.update(DIFFSYNTH_MODEL_BASE_PATH=str(paths['model_base']),HF_ENDPOINT=str(paths['hf_endpoint']))
    entries=[str(ROOT/'src'),str(ROOT),str(paths['track4world_extra_pythonpath']),
             str(paths['libero_repo']),str(ROOT/'experiments/libero')]
    sys.path[:0]=entries
    os.environ['PYTHONPATH']=os.pathsep.join(entries+[os.environ.get('PYTHONPATH','')])
    original_trainer = None
    if args.mode=='train':
        if enabled:
            # Scoped routing to a subclass; no edits to the baseline runtime/trainer loop.
            import fastwam.runtime as runtime
            from fastwam.geometry.trainer import GeometryTrainer
            original_trainer = runtime.Wan22Trainer
            runtime.Wan22Trainer = GeometryTrainer
        entry=ROOT/'scripts/train.py'
    else:
        root=Path(paths['libero_repo'])/'libero/libero'
        cfgdir=Path(paths['output_root'])/'.libero_config'
        cfgdir.mkdir(parents=True,exist_ok=True)
        OmegaConf.save(dict(benchmark_root=str(root),bddl_files=str(root/'bddl_files'),
            init_states=str(root/'init_files'),assets=str(root/'assets'),
            datasets=str(Path(paths['train_datasets'][0]).parent)),cfgdir/'config.yaml')
        os.environ['LIBERO_CONFIG_PATH']=str(cfgdir)
        os.environ.setdefault('MUJOCO_GL','egl')
        entry=ROOT/'experiments/libero'/('run_libero_manager.py' if args.manager else 'eval_libero_single.py')
    sys.argv=[str(entry),'task=libero_geometry_ablation',f'+paths={Path(args.paths).stem}',*overrides]
    try:
        runpy.run_path(str(entry),run_name='__main__')
    finally:
        if original_trainer is not None:
            runtime.Wan22Trainer = original_trainer


if __name__=='__main__':main()
