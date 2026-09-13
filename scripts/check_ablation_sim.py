#!/usr/bin/env python3
"""Two-step real LIBERO smoke of the trained SMALL geometry model, cached T5.

No fake RGB/geometry. Not a full 5B model or a task-success benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'geometry_tests')]
from libero_track4world import load_paths,geometry_config
from test_ablation import tiny
from fastwam.geometry.model import GeometryFastWAM
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paths',default=str(ROOT/'configs/paths/libero_track4world_local.yaml'))
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    paths=load_paths(args.paths);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    for entry in (paths['track4world_extra_pythonpath'],paths['libero_repo'],str(ROOT/'experiments/libero')):
        sys.path.insert(0,entry)
    root=Path(paths['libero_repo'])/'libero/libero';runtime=out/'libero';runtime.mkdir()
    OmegaConf.save(dict(benchmark_root=str(root),bddl_files=str(root/'bddl_files'),init_states=str(root/'init_files'),
        assets=str(root/'assets'),datasets=str(Path(paths['train_datasets'][0]).parent)),runtime/'config.yaml')
    os.environ.update(LIBERO_CONFIG_PATH=str(runtime),MUJOCO_GL='egl')
    from experiments.libero import eval_libero_single as evaluation
    with initialize_config_dir(config_dir=str(ROOT/'configs'),version_base='1.3'):
        cfg=compose(config_name='sim_libero',overrides=['task=libero_geometry_ablation','+paths=libero_track4world_local',
            'EVALUATION.replan_steps=1','EVALUATION.num_steps_wait=8','EVALUATION.num_inference_steps=3'])
    geo=geometry_config(paths);geo.update(latent_channels=4,memory_dim=32,inner_dim=32,heads=2,temporal_layers=1)
    model=tiny(GeometryFastWAM,device='cuda:0',dtype=torch.bfloat16,text_dim=4096)
    model.enable_geometry(geo);model.load_checkpoint(args.checkpoint);model.eval().requires_grad_(False)
    # Use real task-specific cached T5 embeddings with baseline padding, only
    # for this bounded small-model check. Production evaluation loads T5.
    def encode_prompt(prompt):
        context,mask=RobotVideoDataset._get_cached_text_context(
            SimpleNamespace(text_embedding_cache_dir=paths['text_embedding_cache'],context_len=128),prompt)
        context=context.clone();mask=mask.bool()
        context[~mask]=0
        return context[None].to('cuda:0',torch.bfloat16),torch.ones_like(mask)[None].to('cuda:0')
    model.encode_prompt=encode_prompt
    processor=instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(paths['dataset_stats']))
    suite=evaluation.benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(0)
    initial=torch.load(root/'init_files'/task.problem_folder/task.init_states_file,weights_only=False)[0]
    env,description=evaluation.get_libero_env(task,256,42,env_num=1)
    original_max=evaluation._get_max_steps;original_predict=evaluation._predict_action_chunk
    checks=[]
    def predict(*a,**kw):
        result=original_predict(*a,**kw)
        assert np.isfinite(result[0]).all()
        checks.append(list(result[0].shape));return result
    evaluation._get_max_steps=lambda suite:2
    evaluation._predict_action_chunk=predict
    try:
        evaluation.run_single_episode(env,initial,description,model,processor,cfg,0,
            action_horizon=32,input_w=448,input_h=224,model_device='cuda:0')
        assert model._geometry_extractor.calls==2 and len(checks)==2
        report=dict(scope='trained small WAM + real LIBERO + real online Track4World + cached real T5',
            control_steps=2,extractor_calls=2,finite_actions=True,action_shapes=checks,
            gate_values=model.mot.geometry_latent_adapter.gates.detach().float().cpu().tolist(),
            peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    finally:
        evaluation._get_max_steps=original_max;evaluation._predict_action_chunk=original_predict
        env.close()
        if model._geometry_extractor is not None:model._geometry_extractor.close()


if __name__=='__main__':main()
