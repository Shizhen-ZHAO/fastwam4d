import os
from pathlib import Path
import shlex
import subprocess
import pytest
from hydra import compose,initialize_config_dir

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('enabled',['true','false'])
@pytest.mark.parametrize('nodes,gpus,rank',[(1,16,0),(2,8,0),(2,8,1)])
def test_train_shell_composes(enabled,nodes,gpus,rank):
    env=dict(os.environ,DRY_RUN='1',FASTWAM_ENV='',GEOMETRY_ENABLED=enabled,
        NNODES=str(nodes),GPUS_PER_NODE=str(gpus),NODE_RANK=str(rank),MASTER_ADDR='192.0.2.1',RUN_ID='test')
    env.pop('OUTPUT_DIR',None)
    if nodes>1:env['OUTPUT_DIR']='/shared/test'
    output=subprocess.check_output(['bash','scripts/train_ablation_16gpu.sh'],cwd=ROOT,env=env,text=True)
    args=shlex.split(output)
    assert args[args.index('--num_processes')+1]=='16'
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=['task=libero_geometry_ablation','+paths=libero_track4world_local',*args[args.index('--paths')+2:]])
    assert cfg.batch_size*cfg.gradient_accumulation_steps*16==128
    assert cfg.geometry_enabled==(enabled=='true')
    assert str(cfg.output_dir).startswith('/shared' if nodes>1 else '/home/')


@pytest.mark.parametrize('mode',['single','manager'])
def test_eval_shell_composes(mode):
    env=dict(os.environ,DRY_RUN='1',FASTWAM_ENV='',CKPT='/trained/weights.pt',EVAL_MODE=mode)
    env.pop('EVAL_OUTPUT_DIR',None)
    output=subprocess.check_output(['bash','scripts/eval_ablation.sh'],cwd=ROOT,env=env,text=True)
    args=shlex.split(output)
    overrides=[a for a in args[args.index('--paths')+2:] if a!='--manager']
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='sim_libero',overrides=['task=libero_geometry_ablation','+paths=libero_track4world_local',*overrides])
    assert cfg.model.load_text_encoder and cfg.model.skip_dit_load_from_pretrain
    assert cfg.model.action_dit_pretrained_path is None
    assert cfg.EVALUATION.output_dir.startswith('/home/')
