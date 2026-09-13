"""New factories; baseline runtime.py stays byte-for-byte unchanged."""
import inspect
import os
import sys
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from fastwam.runtime import create_fastwam as baseline_factory
from fastwam.runtime import create_fastwam_joint as baseline_joint_factory
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from .model import GeometryFastWAM, GeometryFastWAMJoint
from .rng import preserve_rng


def create_model(*, geometry_enabled, geometry, model_base_path, extra_pythonpath, **kwargs):
    return _create_model(baseline_factory, GeometryFastWAM, geometry_enabled, geometry,
                         model_base_path, extra_pythonpath, kwargs)


def create_joint_model(*, geometry_enabled, geometry, model_base_path, extra_pythonpath, **kwargs):
    return _create_model(baseline_joint_factory, GeometryFastWAMJoint, geometry_enabled, geometry,
                         model_base_path, extra_pythonpath, kwargs)


def _create_model(factory, model_class, geometry_enabled, geometry, model_base_path, extra_pythonpath, kwargs):
    os.environ['DIFFSYNTH_MODEL_BASE_PATH'] = str(model_base_path)
    if not geometry_enabled:
        return factory(**kwargs)
    if extra_pythonpath not in sys.path: sys.path.insert(0,str(extra_pythonpath))
    bound = inspect.signature(factory).bind(**kwargs)
    bound.apply_defaults()
    options = dict(bound.arguments)
    options['torch_dtype'] = options.pop('model_dtype')
    options['action_dit_config'] = options['action_dit_config'] or {}
    for group in ('video_dit_config','action_dit_config','video_scheduler','action_scheduler','loss'):
        if OmegaConf.is_config(options[group]):
            options[group] = OmegaConf.to_container(options[group],resolve=True)
    video = options.pop('video_scheduler') or {}
    action = options.pop('action_scheduler')
    loss = options.pop('loss') or {}
    for key,default in (('train_shift',5.0),('infer_shift',5.0),('num_train_timesteps',1000)):
        options['video_'+key] = video.get(key,default)
        options['action_'+key] = action[key]
    options.update(loss_lambda_video=loss.get('lambda_video',1.0),loss_lambda_action=loss.get('lambda_action',1.0))
    model = model_class.from_wan22_pretrained(**options)
    config = OmegaConf.to_container(geometry,resolve=True) if OmegaConf.is_config(geometry) else geometry
    model.enable_geometry(config)
    return model


class GeometryDataset(Dataset):
    def __init__(self, base, config, cache_root):
        self.base = base
        from fastwam.datasets.libero_geometry import LeRobotGeometryHistoryDataset
        from fastwam.datasets.geometry_cache import LeRobotGeometryCache
        with preserve_rng():
            self.history = LeRobotGeometryHistoryDataset(
                base.lerobot_dataset.dataset_dirs,history_length=config['history_length'],
                history_stride=config['history_stride'],image_size=config['extractor']['image_size'])
            if len(base)!=len(self.history) or self.history.fps!=config['history_fps']:
                raise ValueError('Supervision and geometry history indices/FPS differ')
            self.geometry_cache = LeRobotGeometryCache(cache_root,self.history,config,create=False)
            self.geometry_cache.assert_complete()

    def __len__(self):return len(self.base)
    def __getattr__(self,name):
        if name=='base':raise AttributeError(name)
        return getattr(self.base,name)
    def __getitem__(self,index):
        sample = self.base[index]  # keep original retries/padding/sample choice
        sample['geometry_raw'] = self.geometry_cache.read(int(sample['sample_index']))
        return sample


def create_dataset(*,geometry_enabled,geometry,geometry_cache_root,**kwargs):
    base = RobotVideoDataset(**kwargs)
    if not geometry_enabled:return base
    config = OmegaConf.to_container(geometry,resolve=True) if OmegaConf.is_config(geometry) else geometry
    return GeometryDataset(base,config,geometry_cache_root)
