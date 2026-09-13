"""Joint geometry must preserve the original joint semantics, not alias uncond."""
import inspect
import os
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import numpy as np
import pytest
import torch

from test_ablation import ROOT, tiny, config, sample, geometry
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.geometry.model import GeometryFastWAMJoint
from fastwam.geometry.runtime import create_joint_model
from fastwam.trainer import Wan22Trainer
from experiments.libero_plus.eval_utils import validate_eval_config


def joint_geometry():
    model = tiny(GeometryFastWAMJoint)
    model.enable_geometry(config(), provenance={'fixture': True})
    return model


def test_joint_inherits_original_mask_loss_and_initialization():
    base = tiny(FastWAMJoint)
    rng = torch.get_rng_state().clone()
    model = joint_geometry()
    assert GeometryFastWAMJoint.training_loss is FastWAMJoint.training_loss
    assert GeometryFastWAMJoint._build_mot_attention_mask is FastWAMJoint._build_mot_attention_mask
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in base.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    mask = model._build_mot_attention_mask(12, 4, 4, torch.device('cpu'))
    assert mask[12:, :12].all()
    assert not tiny(FastWAM)._build_mot_attention_mask(12, 4, 4, torch.device('cpu'))[12:, 4:12].any()


def test_joint_zero_gate_loss_gradients_and_optimizer_step():
    base, model = tiny(FastWAMJoint), joint_geometry()
    records = []
    for current in (base, model):
        Wan22Trainer._apply_dit_only_train_mode(current)
        optimizer = torch.optim.AdamW(list(current.dit.parameters()) + list(current.proprio_encoder.parameters()), lr=1e-4)
        torch.manual_seed(456)
        loss, metrics = current.training_loss(sample())
        loss.backward()
        records.append((loss.detach(), metrics, torch.get_rng_state().clone(),
            {key: p.grad.clone() for key, p in current.named_parameters()
             if p.grad is not None and key in dict(base.named_parameters())}))
        optimizer.step()
    torch.testing.assert_close(records[0], records[1], rtol=0, atol=0)
    for key, value in base.named_parameters():
        torch.testing.assert_close(dict(model.named_parameters())[key], value, rtol=0, atol=0)
    assert model.mot.geometry_latent_adapter.gates.grad.abs().sum() > 0


@pytest.mark.parametrize('method', ['infer_action', 'infer_joint', 'infer'])
def test_joint_online_inference_zero_gate_parity_and_single_extraction(method):
    base, model, s = tiny(FastWAMJoint), joint_geometry(), sample()
    assert inspect.signature(model.infer_action) == inspect.signature(base.infer_action)
    assert 'num_video_frames' in inspect.signature(model.infer_action).parameters
    kwargs = dict(prompt=None, input_image=s['video'][:, :, 0], num_video_frames=9,
        action_horizon=32, proprio=s['proprio'][:, 0], context=s['context'],
        context_mask=s['context_mask'], num_inference_steps=2, seed=42)
    if method == 'infer':
        kwargs['num_frames'] = kwargs.pop('num_video_frames')
        kwargs['text_cfg_scale'] = 1.0
    expected = getattr(base, method)(**kwargs)
    with patch.object(model, '_raw_online', return_value=s['geometry_raw']) as extract:
        actual = getattr(model, method)(**kwargs, history_images=torch.zeros(1))
        assert extract.call_count == 1
    assert expected.keys() == actual.keys()
    torch.testing.assert_close(expected['action'], actual['action'], rtol=0, atol=0)
    if 'video' in expected:
        assert len(expected['video']) == len(actual['video'])
        for a, b in zip(expected['video'], actual['video']):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert model._active_geometry is None


def test_joint_nonzero_geometry_reaches_all_branches_and_preserves_targets():
    model, s = joint_geometry(), sample()
    Wan22Trainer._apply_dit_only_train_mode(model)
    model.mot.geometry_latent_adapter.gates.data.fill_(.1)
    inputs = model.build_inputs(s)
    torch.testing.assert_close(inputs['input_latents'], model._encode_video_latents(s['video']), rtol=0, atol=0)
    assert not torch.equal(inputs['first_frame_latents'], inputs['input_latents'][:, :, :1])
    loss, _ = model.training_loss(s)
    loss.backward()
    for branch in model.mot.geometry_latent_adapter.attention.branches.values():
        assert branch.to_kv.weight.grad.abs().sum() > 0


def test_joint_checkpoint_roundtrip_and_cross_variant_rejection(tmp_path):
    joint, uncond = joint_geometry(), geometry()
    joint_path, uncond_path = tmp_path/'joint.pt', tmp_path/'uncond.pt'
    joint.save_checkpoint(joint_path)
    uncond.save_checkpoint(uncond_path)
    joint_geometry().load_checkpoint(joint_path)
    with pytest.raises(ValueError, match='sidecar'):
        joint.load_checkpoint(uncond_path)
    with pytest.raises(ValueError, match='sidecar'):
        uncond.load_checkpoint(joint_path)


def test_joint_factory_exact_baseline_arguments_and_disabled_branch():
    from fastwam.runtime import create_fastwam_joint
    kwargs = dict(model_id='test', tokenizer_model_id='test', video_dit_config={'text_dim':20},
        action_dit_config={}, action_scheduler=dict(train_shift=5., infer_shift=5., num_train_timesteps=1000))
    captured = {}
    class Result:
        def enable_geometry(self, config): pass
    def record(name):
        def call(**options):
            captured[name] = options
            return Result()
        return call
    with patch.object(FastWAMJoint, 'from_wan22_pretrained', side_effect=record('base')):
        create_fastwam_joint(**kwargs)
    with patch.object(GeometryFastWAMJoint, 'from_wan22_pretrained', side_effect=record('geometry')):
        create_joint_model(geometry_enabled=True, geometry={}, model_base_path='/unused', extra_pythonpath='', **kwargs)
    assert captured['base'] == captured['geometry']
    with patch('fastwam.geometry.runtime.baseline_joint_factory', return_value='original') as factory:
        assert create_joint_model(geometry_enabled=False, geometry={}, model_base_path='/unused', extra_pythonpath='', probe=3) == 'original'
        factory.assert_called_once_with(probe=3)


@pytest.mark.parametrize('enabled', ['true', 'false'])
@pytest.mark.parametrize('nodes,gpus,rank', [(1,16,0), (2,8,0), (2,8,1)])
def test_joint_train_launcher_and_paired_recipe(enabled, nodes, gpus, rank):
    env = dict(os.environ, DRY_RUN='1', FASTWAM_ENV='', GEOMETRY_ENABLED=enabled,
        NNODES=str(nodes), GPUS_PER_NODE=str(gpus), NODE_RANK=str(rank),
        MASTER_ADDR='192.0.2.1', RUN_ID='test', OUTPUT_DIR='/shared/joint')
    for key in ('BATCH_SIZE','GRAD_ACCUM'):
        env.pop(key, None)
    argv = shlex.split(subprocess.check_output(['bash','scripts/train_joint_ablation_16gpu.sh'], cwd=ROOT, env=env, text=True))
    assert argv[argv.index('--variant')+1] == 'joint'
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        new = compose(config_name='train', overrides=['task=libero_joint_geometry_ablation',
            '+paths=libero_track4world_local', *argv[argv.index('--paths')+2:]])
        base = compose(config_name='train', overrides=['task=libero_joint_2cam224_1e-4', 'batch_size=8', 'gradient_accumulation_steps=1'])
        uncond = compose(config_name='train', overrides=['task=libero_geometry_ablation', '+paths=libero_track4world_local'])
    assert new.model._target_ == 'fastwam.geometry.runtime.create_joint_model'
    assert new.batch_size == 8 and new.gradient_accumulation_steps == 1
    assert new.geometry_enabled == (enabled == 'true')
    for key in ('batch_size','gradient_accumulation_steps','learning_rate','weight_decay','max_grad_norm',
                'num_epochs','lr_scheduler_type','mixed_precision','seed','save_every','eval_every','eval_num_inference_steps'):
        assert new[key] == base[key], key
    for key in ('video_dit_config','action_dit_config','video_scheduler','action_scheduler','loss'):
        assert OmegaConf.to_container(new.model[key], resolve=True) == OmegaConf.to_container(base.model[key], resolve=True)
    # Raw cache contract is variant-independent: no re-extraction for joint.
    assert OmegaConf.to_container(new.model.geometry, resolve=True) == OmegaConf.to_container(uncond.model.geometry, resolve=True)


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_joint_production_entry_selects_joint_task(monkeypatch, enabled):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import geometry_ablation as entry
    original_path = list(sys.path)
    seen = []
    monkeypatch.setattr(entry,'validate_paths',lambda *a,**kw:None)
    monkeypatch.setattr(entry,'validate_model_assets',lambda *a,**kw:None)
    monkeypatch.setattr(entry.runpy,'run_path',lambda *a,**kw:seen.append(list(sys.argv)))
    monkeypatch.setattr(sys,'argv',['geometry_ablation.py','train','--variant','joint', f'geometry_enabled={enabled}'])
    try:
        with patch.dict(os.environ):
            entry.main()
    finally:
        sys.path[:] = original_path
    assert 'task=libero_joint_geometry_ablation' in seen[0]


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_joint_plus_launcher_and_config(enabled):
    env = dict(os.environ, DRY_RUN='1', FASTWAM_ENV='', GEOMETRY_ENABLED=enabled,
        FASTWAM_VARIANT='joint', CKPT='/trained/joint.pt')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env['PATH']
    output = subprocess.check_output(['bash','scripts/eval_geometry_plus.sh'], cwd=ROOT, env=env, text=True)
    line = next(line for line in output.splitlines() if 'run_libero_plus_manager.py ' in line)
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='sim_libero_plus', overrides=shlex.split(line)[2:])
    assert validate_eval_config(cfg) == 'joint'
    assert cfg.geometry_enabled == (enabled == 'true')


@pytest.mark.parametrize('mode', ['single','manager'])
@pytest.mark.parametrize('enabled', ['true','false'])
def test_joint_libero_eval_launcher(mode, enabled):
    env = dict(os.environ, DRY_RUN='1', FASTWAM_ENV='', GEOMETRY_ENABLED=enabled,
        CKPT='/trained/joint.pt', EVAL_MODE=mode)
    output = subprocess.check_output(['bash','scripts/eval_joint_ablation.sh'], cwd=ROOT, env=env, text=True)
    argv = shlex.split(output)
    assert argv[argv.index('--variant')+1] == 'joint'
    assert ('--manager' in argv) == (mode == 'manager')
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        cfg = compose(config_name='sim_libero', overrides=['task=libero_joint_geometry_ablation',
            '+paths=libero_track4world_local', *[a for a in argv[argv.index('--paths')+2:] if a != '--manager']])
    assert cfg.model._target_ == 'fastwam.geometry.runtime.create_joint_model'
    assert cfg.geometry_enabled == (enabled == 'true')
    assert cfg.model.load_text_encoder and cfg.model.skip_dit_load_from_pretrain
    assert cfg.model.action_dit_pretrained_path is None
