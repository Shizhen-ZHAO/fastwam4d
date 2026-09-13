"""Geometry-only state provenance hooks; baseline training loop stays inherited."""
from copy import deepcopy
import json
import os
from pathlib import Path

from fastwam.trainer import Wan22Trainer


def register_geometry_state_hooks(accelerator, model, cache_contract):
    if getattr(model, 'geometry_config', None) is None:
        raise ValueError('Geometry state hooks require a geometry model')
    if cache_contract is None:
        raise ValueError('Offline geometry trainer requires a verified cache contract')
    contract = deepcopy(cache_contract)

    def metadata():
        return dict(format='fastwam_geometry_training_state_v1',
                    model=model._geometry_metadata(), cache=contract)

    def save(models, weights, output_dir):
        if accelerator.is_main_process:
            path = Path(output_dir) / 'geometry_state.json'
            temporary = path.with_name(f'.{path.name}.{os.getpid()}.pending')
            temporary.write_text(json.dumps(metadata(), sort_keys=True, indent=2) + '\n')
            temporary.replace(path)

    def load(models, input_dir):
        path = Path(input_dir) / 'geometry_state.json'
        if not path.is_file() or json.loads(path.read_text()) != metadata():
            raise ValueError('Geometry training-state contract mismatch or missing geometry_state.json; '
                             'use the same geometry configuration, data and extraction weights/code')

    return (accelerator.register_save_state_pre_hook(save),
            accelerator.register_load_state_pre_hook(load))


class GeometryTrainer(Wan22Trainer):
    def _resume_or_load_checkpoint(self):
        # Original __init__ calls this after prepare(), before any state restore.
        model = self.accelerator.unwrap_model(self.model)
        cache = getattr(self.train_dataset, 'geometry_cache', None)
        self._geometry_state_hooks = register_geometry_state_hooks(
            self.accelerator, model, getattr(cache, 'contract', None))
        return super()._resume_or_load_checkpoint()
