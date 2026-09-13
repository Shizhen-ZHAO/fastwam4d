from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from accelerate import Accelerator
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from test_ablation import geometry
from fastwam.geometry.trainer import register_geometry_state_hooks, GeometryTrainer
from fastwam.trainer import Wan22Trainer
from experiments.libero_plus.eval_utils import validate_eval_config
from experiments.libero_plus.run_libero_plus_manager import write_worker_config
from experiments.libero_plus.test.test_libero_plus import FakeEnv, extract

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('enabled', [True, False])
def test_production_entry_selects_geometry_state_hooks_only_when_enabled(monkeypatch, enabled):
    saved_path = list(sys.path)
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        import geometry_ablation as entry
        import fastwam.runtime as runtime
        original = runtime.Wan22Trainer
        observed = []
        monkeypatch.setattr(entry,'validate_paths',lambda *a,**kw:None)
        monkeypatch.setattr(entry,'validate_model_assets',lambda *a,**kw:None)
        monkeypatch.setattr(entry.runpy,'run_path',lambda *a,**kw:observed.append(runtime.Wan22Trainer))
        monkeypatch.setattr(sys,'argv',['geometry_ablation.py','train',f'geometry_enabled={str(enabled).lower()}'])
        with patch.dict(os.environ):
            entry.main()
        assert observed == [GeometryTrainer if enabled else original]
        assert runtime.Wan22Trainer is original
    finally:
        sys.path[:] = saved_path


def test_full_state_rejects_changed_geometry_or_missing_contract(tmp_path):
    accelerator = Accelerator(cpu=True)
    model = accelerator.prepare(geometry())
    handles = register_geometry_state_hooks(accelerator, model, {'dataset': 'fixture'})
    try:
        accelerator.save_state(str(tmp_path))
        model.geometry_config['same_view_only'] = False
        with pytest.raises(ValueError, match='contract'):
            accelerator.load_state(str(tmp_path))
        model.geometry_config['same_view_only'] = True
        accelerator.load_state(str(tmp_path))
        sidecar = tmp_path / 'geometry_state.json'
        meta = json.loads(sidecar.read_text())
        meta['cache']['dataset'] = 'changed-data'
        sidecar.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match='contract'):
            accelerator.load_state(str(tmp_path))
        sidecar.unlink()
        with pytest.raises(ValueError, match='contract'):
            accelerator.load_state(str(tmp_path))
    finally:
        for handle in handles:
            handle.remove()
        accelerator.end_training()
    for method in ('train','_build_loader','save_checkpoint','load_training_state'):
        assert getattr(GeometryTrainer,method) is getattr(Wan22Trainer,method)


@pytest.mark.parametrize('enabled', [True, False])
def test_plus_geometry_config_worker_snapshot_and_launcher(enabled, tmp_path):
    env = dict(os.environ, DRY_RUN='1', CKPT='/trained/weights.pt', FASTWAM_ENV='',
               GEOMETRY_ENABLED=str(enabled).lower())
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
    output = subprocess.check_output(['bash','scripts/eval_geometry_plus.sh'], cwd=ROOT, env=env, text=True)
    # The geometry wrapper and Plus launcher print paths/argv ahead of the resolved YAML.
    import shlex
    line = next(line for line in output.splitlines() if 'run_libero_plus_manager.py ' in line)
    argv = shlex.split(line)
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg = compose(config_name='sim_libero_plus',overrides=argv[2:])
    assert validate_eval_config(cfg) == 'uncond'
    assert cfg.geometry_enabled == enabled
    assert cfg.model.load_text_encoder and cfg.model.skip_dit_load_from_pretrain
    assert cfg.model.action_dit_pretrained_path is None
    assert cfg.MULTIRUN.max_tasks_per_gpu == 1
    assert cfg.EVALUATION.infer_increment_seed
    resolved = OmegaConf.to_container(cfg, resolve=True)
    write_worker_config(resolved,tmp_path)
    with initialize_config_dir(config_dir=str(tmp_path),version_base='1.3'):
        worker = compose(config_name='worker_config')
    assert OmegaConf.to_container(worker.model,resolve=True) == resolved['model']


def test_python_extraction_defaults_to_concrete_cuda_device():
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        from libero_track4world import make_parser
        for mode in ('extract','parity'):
            args = make_parser().parse_args([mode,'--indices','0'])
            assert torch.device(args.device) == torch.device('cuda:0')
    finally:
        sys.path.remove(str(ROOT/'scripts'))


@pytest.mark.parametrize('wait', [0, 8])
def test_geometry_keeps_all_real_frames_with_plus_render_skip_enabled(wait):
    memories, seeds = [], []
    def predict(**kwargs):
        memories.append(kwargs['geometry_history'].inputs())
        seeds.append(kwargs['infer_seed'])
        return np.ones((32,7)), kwargs['obs'], None
    cfg = OmegaConf.create(dict(seed=42,EVALUATION=dict(task_suite_name='libero_10',
        replan_steps=3,num_steps_wait=wait,use_action_ensembler=False,visualize_future_video=False,
        skip_unused_render=True,save_video=False,infer_increment_seed=True)))
    namespace = dict(np=np,logging=logging,_get_max_steps=lambda _:20,
        _get_future_frame_capture_steps=lambda _:[0,4,8],_predict_action_chunk=predict,
        get_libero_dummy_action=lambda:[0.]*7,
        get_libero_image=lambda obs:dict(image=obs['image'],wrist_image=obs['image']+100),
        tqdm=lambda **kw:SimpleNamespace(update=lambda *a:None,close=lambda:None))
    ns = extract(ROOT/'experiments/libero/eval_libero_single.py',
        {'run_single_episode','_resolve_infer_seed','_maybe_install_render_gate','_SimRenderGate'},namespace)
    model = SimpleNamespace(geometry_config=dict(history_length=8,history_stride=1,
                                                 history_fps=20.,extractor={'image_size':2}))
    for _ in range(2):  # second episode must not reuse the previous history
        memories.clear();seeds.clear()
        env = FakeEnv(wait+7)
        ns['run_single_episode'](env,None,'test',model,None,cfg,0,
            action_horizon=32,input_w=224,input_h=224,model_device='cpu')
        assert env.renders == wait+9  # reset + init + every step
        assert seeds == [42,43,44]
        for k, memory in enumerate(memories):
            current=wait+3*k
            expected=torch.arange(current-6,current+2).clamp_min(1).float()
            rgb=memory['history_images'][0,:,:,0,0,0].mul(255).round()
            torch.testing.assert_close(rgb[0],expected)
            torch.testing.assert_close(rgb[1],expected+100)
            assert memory['history_valid'][0].sum() == min(current+1,8)
