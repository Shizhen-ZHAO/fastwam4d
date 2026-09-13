import ast
from copy import deepcopy
import inspect
import os
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_vae import WanVideoVAE38,VideoVAE38_
from fastwam.geometry.model import GeometryFastWAM
from fastwam.geometry.runtime import create_model,GeometryDataset
from fastwam.trainer import Wan22Trainer

ROOT=Path(__file__).resolve().parents[1]
BASE='31770396e672bfa283f4283e59775057ab69ec89'
ALLOWED={'src/fastwam/trainer.py','src/fastwam/datasets/lerobot/robot_video_dataset.py',
         'experiments/libero/eval_libero_single.py','experiments/libero_plus/eval_utils.py'}


def tiny(cls=FastWAM,device='cpu',dtype=torch.float32,text_dim=20):
    torch.manual_seed(42)
    common=dict(text_dim=text_dim,freq_dim=12,eps=1e-6,num_heads=2,attn_head_dim=12,num_layers=2,use_gradient_checkpointing=False)
    v=WanVideoDiT(**common,hidden_dim=24,in_dim=4,out_dim=4,ffn_dim=48,patch_size=(1,2,2),
        has_image_input=False,seperated_timestep=True,fuse_vae_embedding_in_latents=True,
        action_conditioned=False,video_attention_mask_mode='first_frame_causal')
    a=ActionDiT(**common,hidden_dim=16,ffn_dim=32,action_dim=7,action_rope_mode='1d')
    vae=WanVideoVAE38.__new__(WanVideoVAE38)
    torch.nn.Module.__init__(vae)
    vae.model=VideoVAE38_(dim=8,z_dim=4,dec_dim=8,num_res_blocks=1)
    vae.upsampling_factor,vae.temporal_downsample_factor,vae.z_dim=16,4,4
    vae.register_buffer('mean',torch.zeros(4),persistent=False)
    vae.register_buffer('inv_std',torch.ones(4),persistent=False)
    return cls(v,a,MoT({'video':v,'action':a},False),vae,text_dim=text_dim,proprio_dim=8,device=device,torch_dtype=dtype).to(dtype=dtype)


def config():
    return dict(target='vae_latent',train_mode='full',num_views=2,history_length=8,history_stride=1,
        history_fps=20.,memory_dim=32,inner_dim=32,heads=2,temporal_layers=1,latent_channels=4,
        same_view_only=True,extractor={'grid_size':2,'image_size':256})


def sample():
    g=torch.Generator().manual_seed(99)
    rand=lambda *s:torch.randn(*s,generator=g)
    return dict(video=rand(1,3,9,32,64),action=rand(1,32,7),proprio=rand(1,32,8),
        context=rand(1,5,20),context_mask=torch.ones(1,5,dtype=torch.bool),
        image_is_pad=torch.zeros(1,9,dtype=torch.bool),action_is_pad=torch.zeros(1,32,dtype=torch.bool),
        geometry_raw=dict(scene=rand(1,2,4,1024),scene_aux=rand(1,2,4,6),scene_valid=torch.ones(1,2,4,dtype=torch.bool),
            camera=rand(1,2,3072),camera_aux=rand(1,2,13),camera_valid=torch.ones(1,2,dtype=torch.bool),
            track=rand(1,2,8,4,256),track_aux=rand(1,2,8,4,11),track_valid=torch.ones(1,2,8,4,dtype=torch.bool)))


def geometry():
    m=tiny(GeometryFastWAM)
    m.enable_geometry(config(),provenance={'fixture':True})
    return m


def test_only_four_geometry_wiring_files_changed():
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=ROOT,text=True).splitlines()
    changed=[]
    for path in paths:
        before=subprocess.check_output(['git','show',f'{BASE}:{path}'],cwd=ROOT)
        if (ROOT/path).read_bytes()!=before:changed.append(path)
    assert set(changed)==ALLOWED
    assert GeometryFastWAM.training_loss is FastWAM.training_loss
    old=ast.parse(subprocess.check_output(['git','show',f'{BASE}:src/fastwam/trainer.py'],cwd=ROOT,text=True))
    oldcls=next(n for n in old.body if isinstance(n,ast.ClassDef))
    newcls=next(n for n in ast.parse((ROOT/'src/fastwam/trainer.py').read_text()).body if isinstance(n,ast.ClassDef))
    oldmethods={n.name:ast.dump(n) for n in oldcls.body if isinstance(n,ast.FunctionDef)}
    for n in newcls.body:
        if isinstance(n,ast.FunctionDef) and n.name not in ('_to_batched_eval_sample','evaluate'):
            assert ast.dump(n)==oldmethods[n.name],n.name


def test_initialization_preserves_backbone_and_rng():
    base=tiny(); state=torch.get_rng_state().clone()
    enhanced=geometry()
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    for k,v in base.state_dict().items():torch.testing.assert_close(enhanced.state_dict()[k],v,rtol=0,atol=0)


def test_zero_gate_loss_gradients_noise_and_first_optimizer_update():
    base,geo=tiny(),geometry()
    outputs=[]
    for m in (base,geo):
        Wan22Trainer._apply_dit_only_train_mode(m)
        opt=torch.optim.AdamW(list(m.dit.parameters())+list(m.proprio_encoder.parameters()),lr=1e-4,betas=(.9,.95))
        torch.manual_seed(456)
        loss,metrics=m.training_loss(sample())
        loss.backward()
        outputs.append((loss.detach(),metrics,torch.get_rng_state().clone(),
            {k:p.grad.clone() for k,p in m.named_parameters() if p.grad is not None and k in dict(base.named_parameters())}))
        opt.step()
    torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
    for k,p in base.named_parameters():torch.testing.assert_close(dict(geo.named_parameters())[k],p,rtol=0,atol=0)
    assert geo.mot.geometry_latent_adapter.gates.grad.abs().sum()>0


def test_nonzero_geometry_has_gradients_without_changing_targets():
    m=geometry(); Wan22Trainer._apply_dit_only_train_mode(m)
    m.mot.geometry_latent_adapter.gates.data.fill_(.1)
    s=sample(); inputs=m.build_inputs(s)
    torch.testing.assert_close(inputs['input_latents'],m._encode_video_latents(s['video']),rtol=0,atol=0)
    assert not torch.equal(inputs['first_frame_latents'],inputs['input_latents'][:,:,:1])
    loss,_=m.training_loss(s);loss.backward()
    for name,branch in m.mot.geometry_latent_adapter.attention.branches.items():
        assert branch.to_kv.weight.grad.abs().sum()>0,name
    assert not any(p.requires_grad for p in m.vae.parameters())


def test_inference_zero_gate_parity_and_online_extract_once():
    b,g=tiny(),geometry();s=sample()
    kwargs=dict(prompt=None,input_image=s['video'][:,:,0],action_horizon=32,proprio=s['proprio'][:,0],
        context=s['context'],context_mask=s['context_mask'],num_inference_steps=3,seed=42)
    expected=b.infer_action(**kwargs)
    with patch.object(g,'_raw_online',return_value=s['geometry_raw']) as extract:
        actual=g.infer_action(**kwargs,history_images=torch.zeros(1))
        assert extract.call_count==1
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    assert g._active_geometry is None


def test_geometry_checkpoint_sidecar(tmp_path):
    m=geometry();m.mot.geometry_latent_adapter.gates.data.fill_(.2)
    p=tmp_path/'weights.pt';m.save_checkpoint(p,step=2)
    restored=geometry();restored.load_checkpoint(p)
    torch.testing.assert_close(restored.mot.geometry_latent_adapter.gates,m.mot.geometry_latent_adapter.gates)
    p.with_name(p.name+'.geometry.json').unlink()
    with pytest.raises(ValueError,match='sidecar'):restored.load_checkpoint(p)


def test_retry_identity_is_preserved_and_cache_errors_not_retried():
    ds=GeometryDataset.__new__(GeometryDataset)
    ds.base=SimpleNamespace(__getitem__=None)
    class Base:
        def __getitem__(self,i):return {'sample_index':7}
    class Cache:
        def read(self,i):assert i==7;raise FileNotFoundError('bad cache')
    ds.base=Base();ds.geometry_cache=Cache()
    with pytest.raises(FileNotFoundError,match='bad cache'):ds[0]


@pytest.mark.parametrize('enabled',[True,False])
def test_paired_recipe(enabled):
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        base=compose(config_name='train',overrides=['task=libero_uncond_2cam224_1e-4','batch_size=4','gradient_accumulation_steps=2'])
        new=compose(config_name='train',overrides=['task=libero_geometry_ablation','+paths=libero_track4world_local',
            f'geometry_enabled={str(enabled).lower()}','batch_size=4','gradient_accumulation_steps=2'])
    for key in ('batch_size','gradient_accumulation_steps','learning_rate','weight_decay','max_grad_norm',
                'num_epochs','lr_scheduler_type','mixed_precision','seed','save_every','eval_every','eval_num_inference_steps'):
        assert new[key]==base[key],key
    for key in ('video_dit_config','action_dit_config','video_scheduler','action_scheduler','loss'):
        assert OmegaConf.to_container(new.model[key],resolve=True)==OmegaConf.to_container(base.model[key],resolve=True)
    assert new.data.get('val') is None


def test_disabled_factory_returns_original_without_geometry():
    with patch('fastwam.geometry.runtime.baseline_factory',return_value='original') as base:
        # Signature lookup happens only in the enabled branch.
        assert create_model(geometry_enabled=False,geometry={},model_base_path='/unused',extra_pythonpath='',probe=3)=='original'
        base.assert_called_once_with(probe=3)


def test_factory_passes_exact_original_initialization_arguments():
    from fastwam.runtime import create_fastwam
    kwargs=dict(model_id='test',tokenizer_model_id='test',video_dit_config={'text_dim':20},
        action_dit_config={},action_scheduler=dict(train_shift=5.,infer_shift=5.,num_train_timesteps=1000))
    captured={}
    class Result:
        def enable_geometry(self,config):pass
    def record(name):
        def call(**options):captured[name]=options;return Result()
        return call
    with patch.object(FastWAM,'from_wan22_pretrained',side_effect=record('base')):
        create_fastwam(**kwargs)
    with patch.object(GeometryFastWAM,'from_wan22_pretrained',side_effect=record('geometry')):
        create_model(geometry_enabled=True,geometry={},model_base_path='/unused',extra_pythonpath='',**kwargs)
    assert captured['base']==captured['geometry']


def test_joint_diagnostic_reuses_one_geometry_extraction():
    b,g=tiny(),geometry();s=sample()
    kwargs=dict(prompt=None,input_image=s['video'][:,:,0],num_video_frames=9,action_horizon=32,
        proprio=s['proprio'][:,0],context=s['context'],context_mask=s['context_mask'],num_inference_steps=2,seed=42)
    expected=b.infer_joint(**kwargs)
    with patch.object(g,'_raw_online',return_value=s['geometry_raw']) as extract:
        actual=g.infer_joint(**kwargs,history_images=torch.zeros(1))
        assert extract.call_count==1
    torch.testing.assert_close(expected['action'],actual['action'],rtol=0,atol=0)
    for a,c in zip(expected['video'],actual['video']):np.testing.assert_array_equal(np.asarray(a),np.asarray(c))


def test_trainer_validation_keeps_geometry_payload():
    item={k:({n:v[0] for n,v in value.items()} if k=='geometry_raw' else value[0]) for k,value in sample().items()}
    item['prompt']='test'
    batched=Wan22Trainer._to_batched_eval_sample(item)
    assert batched['geometry_raw']['track'].shape==(1,2,8,4,256)
    loss,_=geometry().training_loss(batched)
    assert torch.isfinite(loss)
