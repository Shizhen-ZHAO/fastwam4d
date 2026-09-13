"""Geometry-only subclass. No override of loss, scheduler, denoising or optimizer."""
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
import json
from pathlib import Path

import torch
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.fastwam_joint import FastWAMJoint
from fastwam.models.wan22.geometry_adapter import GeometryTokenizer, VAELatentGeometryAdapter
from fastwam.models.wan22.geometry_features import validate_raw_geometry
from .rng import preserve_rng


class _GeometryConditioning:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_config = None
        self._geometry_extractor = None
        self._active_geometry = None
        self.geometry_provenance = None

    def enable_geometry(self, config, *, provenance=None):
        if self.geometry_config is not None:
            raise ValueError('Geometry already enabled')
        config = deepcopy(dict(config))
        if config.get('target') != 'vae_latent' or config.get('num_views',2) != 2:
            raise ValueError('Only two-view current VAE latent geometry is supported')
        if config.get('train_mode','full') != 'full':
            raise ValueError('This ablation retains baseline full training; no adapter-only mode')
        if not self.video_expert.fuse_vae_embedding_in_latents:
            raise ValueError('Geometry requires current-frame VAE conditioning')
        if not 1 <= int(config.get('history_length',8)) <= 8:
            raise ValueError('history_length must be in [1,8]')
        self.geometry_config = config
        dim, heads = int(config.get('memory_dim',512)), int(config.get('heads',8))
        # Register on the EXISTING dit/MoT so the UNCHANGED trainer includes
        # these parameters in its original freeze logic and optimizer groups.
        with preserve_rng(self.device):
            self.mot.geometry_tokenizer = GeometryTokenizer(dim,heads,int(config.get('temporal_layers',2)),2).to(self.device)
            self.mot.geometry_latent_adapter = VAELatentGeometryAdapter(
                int(config.get('latent_channels',self.vae.z_dim)),dim,
                int(config.get('inner_dim',dim)),heads,2,
                bool(config.get('same_view_only',True))).to(self.device)
        if provenance is None:
            from fastwam.datasets.geometry_provenance import extractor_identity
            provenance = extractor_identity(config['extractor'])
        self.geometry_provenance = provenance

    def _raw_online(self, images, timestamps, valid):
        if images is None or timestamps is None or valid is None:
            raise ValueError('Online geometry requires causal RGB history, timestamps and validity')
        from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor
        with preserve_rng(self.device):
            if self._geometry_extractor is None:
                settings = dict(self.geometry_config['extractor'])
                if settings.get('device') in (None,'same','model'):
                    settings['device'] = str(self.device)
                self._geometry_extractor = OnlineTrack4WorldExtractor(**settings)
            return self._geometry_extractor(images,timestamps,valid,output_device=self.device)

    def _memory(self, raw=None, history_images=None, history_timestamps=None, history_valid=None):
        if raw is None:
            raw = self._raw_online(history_images,history_timestamps,history_valid)
        settings = self.geometry_config['extractor']
        validate_raw_geometry(raw,views=2,length=int(self.geometry_config.get('history_length',8)),
                              points=int(settings.get('grid_size',8))**2)
        raw = {k:v.detach().to(self.device) for k,v in raw.items()}
        return self.mot.geometry_tokenizer(raw)

    def build_inputs(self, sample, tiled=False):
        inputs = super().build_inputs(sample,tiled=tiled)
        memory = self._memory(sample.get('geometry_raw'),sample.get('history_images'),
                              sample.get('history_timestamps'),sample.get('history_valid'))
        # Do NOT mutate input_latents: the inherited loss creates its original
        # noised future and flow targets from that tensor.
        inputs['first_frame_latents'] = self.mot.geometry_latent_adapter(inputs['first_frame_latents'],memory)
        return inputs

    def _encode_input_image_latents_tensor(self, *args, **kwargs):
        latent = super()._encode_input_image_latents_tensor(*args,**kwargs)
        if self._active_geometry is None:
            raise ValueError('Geometry inference must enter through infer/infer_action/infer_joint')
        return self.mot.geometry_latent_adapter(latent,self._active_geometry)

    @contextmanager
    def _inference_geometry(self, kwargs):
        fields = {k:kwargs.pop(k,None) for k in ('geometry_raw','history_images','history_timestamps','history_valid')}
        if self._active_geometry is not None:
            yield  # inherited infer -> infer_joint -> diagnostic infer_action
            return
        self._active_geometry = self._memory(fields['geometry_raw'],fields['history_images'],
                                              fields['history_timestamps'],fields['history_valid'])
        try:
            yield
        finally:
            self._active_geometry = None

    @torch.no_grad()
    @wraps(FastWAM.infer_joint)
    def infer_joint(self,*args,**kwargs):
        self.eval()
        with self._inference_geometry(kwargs):
            return super().infer_joint(*args,**kwargs)

    @torch.no_grad()
    @wraps(FastWAM.infer)
    def infer(self,*args,**kwargs):
        self.eval()
        with self._inference_geometry(kwargs):
            return super().infer(*args,**kwargs)

    def _geometry_metadata(self):
        config = deepcopy(self.geometry_config)
        for key in ('repo_path','checkpoint_path','da3_path','device'):
            config['extractor'].pop(key,None)
        return dict(format='fastwam_geometry_ablation_v1',config=config,producer=self.geometry_provenance)

    def save_checkpoint(self,path,optimizer=None,step=None):
        # Original checkpoint layout and save timing; geometry tensors already
        # belong to mot. Only add a geometry-specific portable sidecar.
        super().save_checkpoint(path,optimizer=optimizer,step=step)
        metadata = Path(str(path)+'.geometry.json')
        metadata.write_text(json.dumps(self._geometry_metadata(),sort_keys=True,indent=2)+'\n')

    def load_checkpoint(self,path,optimizer=None):
        metadata = Path(str(path)+'.geometry.json')
        if not metadata.is_file() or json.loads(metadata.read_text()) != self._geometry_metadata():
            raise ValueError('Missing/mismatched geometry sidecar: copy .pt AND .pt.geometry.json; check local producer/config')
        payload = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        expected = {k for k in self.mot.state_dict() if k.startswith(('geometry_tokenizer.','geometry_latent_adapter.'))}
        if not expected.issubset(payload.get('mot',{})):
            raise ValueError('Checkpoint is missing trained geometry tensors')
        del payload
        return super().load_checkpoint(path,optimizer=optimizer)


class GeometryFastWAM(_GeometryConditioning, FastWAM):
    @torch.no_grad()
    @wraps(FastWAM.infer_action)
    def infer_action(self, *args, **kwargs):
        self.eval()
        with self._inference_geometry(kwargs):
            return super().infer_action(*args, **kwargs)


class GeometryFastWAMJoint(_GeometryConditioning, FastWAMJoint):
    """Retain joint's full-video attention and joint video/action denoising."""

    @torch.no_grad()
    @wraps(FastWAMJoint.infer_action)
    def infer_action(self, *args, **kwargs):
        # LIBERO inspects this signature to supply joint's num_video_frames.
        self.eval()
        with self._inference_geometry(kwargs):
            return super().infer_action(*args, **kwargs)

    def _geometry_metadata(self):
        metadata = super()._geometry_metadata()
        # Joint/uncond weights can have identical shapes but different semantics.
        # Keep old uncond sidecars compatible; reject cross-variant loading.
        metadata['variant'] = 'joint'
        return metadata
