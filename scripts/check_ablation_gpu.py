#!/usr/bin/env python3
"""Real cached data + small actual WAM/VAE through ORIGINAL trainer. Not 5B training."""
import argparse
import json
import os
from pathlib import Path
import sys
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch.utils.data import Dataset,default_collate

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'geometry_tests')]
from libero_track4world import load_paths,build_history,geometry_config
from test_ablation import tiny
from fastwam.geometry.model import GeometryFastWAM
from fastwam.datasets.geometry_cache import LeRobotGeometryCache
from fastwam.geometry.trainer import GeometryTrainer as Wan22Trainer
from fastwam.utils import misc


class Samples(Dataset):
    def __init__(self,base,cache):
        self.base=base
        self.geometry_cache=cache
        self.items=[]
        for i in (0,7):
            s=base[i];assert int(s['sample_index'])==i
            s['geometry_raw']=cache.read(i)
            self.items.append(s)
    def __len__(self):return 2
    def __getitem__(self,i):return self.items[i]
    def __getattr__(self,k):
        if k=='base':raise AttributeError(k)
        return getattr(self.base,k)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',default=str(ROOT/'configs/paths/libero_track4world_local.yaml'))
    p.add_argument('--output',required=True)
    p.add_argument('--skip-online',action='store_true')
    args=p.parse_args()
    rank=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(rank)
    out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    paths=load_paths(args.paths);sys.path.insert(0,paths['track4world_extra_pythonpath'])
    misc.register_work_dir(out)
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='train',overrides=['task=libero_geometry_ablation',f'+paths={Path(args.paths).stem}',
            'geometry_enabled=false','data.train.video_size=[32,64]','batch_size=1','gradient_accumulation_steps=2',
            'num_workers=0','max_steps=4','log_every=1','save_every=2','eval_every=2','eval_num_inference_steps=2',
            f'output_dir={out}','wandb.enabled=false'])
    base=instantiate(cfg.data.train)
    history=build_history(paths);geo=geometry_config(paths)
    cache=LeRobotGeometryCache(paths['geometry_cache'],history,geo,create=False)
    samples=Samples(base,cache)
    # Deliberately TWO samples and a SMALL network: production factory keeps
    # full-cache coverage checks and original full-size model defaults.
    model=tiny(GeometryFastWAM,device=f'cuda:{rank}',dtype=torch.bfloat16,text_dim=4096)
    geo.update(latent_channels=4,memory_dim=32,inner_dim=32,heads=2,temporal_layers=1)
    model.enable_geometry(geo,provenance=cache.contract['extractor'])
    before={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    trainer=Wan22Trainer(model,samples,val_dataset=samples,cfg=cfg)
    trainer.train()
    model=trainer.accelerator.unwrap_model(trainer.model)
    assert model._geometry_extractor is None,'Offline train/val must not load Track4World'
    changed=[k for k,p in model.named_parameters() if not torch.equal(before[k],p.detach().cpu())]
    for prefix in ('video_expert.','action_expert.','mot.geometry_tokenizer.','mot.geometry_latent_adapter.'):
        assert any(k.startswith(prefix) for k in changed),prefix
    assert all(torch.isfinite(p).all() for p in model.parameters())
    gate=model.mot.geometry_latent_adapter.gates.detach().clone()
    path=Path(trainer.weights_dir)/'step_000004.pt'
    model.load_checkpoint(path)
    with torch.no_grad():model.mot.geometry_latent_adapter.gates.add_(1)
    trainer.load_training_state(str(Path(trainer.state_dir)/'step_000004'))
    torch.testing.assert_close(model.mot.geometry_latent_adapter.gates,gate,rtol=0,atol=0)
    report=dict(scope='small actual WAM/VAE, real two LeRobot samples; original trainer/optimizer/sampler',
        distributed=str(trainer.accelerator.distributed_type),world_size=trainer.accelerator.num_processes,
        optimizer_steps=trainer.global_step,updated_parameter_tensors=len(changed),offline_tracker_calls=0,
        checkpoint_and_state_reload=True)
    if not args.skip_online:
        item=default_collate([samples.items[1]]);h=history[7]
        model.eval().requires_grad_(False)
        with torch.no_grad(),trainer.accelerator.autocast():
            action=model.infer_action(prompt=None,input_image=item['video'][:,:,0],action_horizon=32,
                proprio=item['proprio'][:,0],context=item['context'],context_mask=item['context_mask'],
                num_inference_steps=3,seed=42,history_images=h['history_images'][None],
                history_timestamps=h['history_timestamps'][None],history_valid=h['history_valid'][None])['action']
        assert action.shape==(32,7) and torch.isfinite(action).all()
        assert model._geometry_extractor.calls==1
        report['online_inference']=dict(action_shape=list(action.shape),extractor_calls=1,denoising_steps=3)
        model._geometry_extractor.close()
    report['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1e9
    (out/f'report_rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    trainer.accelerator.end_training()


if __name__=='__main__':main()
