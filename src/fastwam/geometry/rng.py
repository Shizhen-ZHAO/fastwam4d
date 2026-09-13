from contextlib import contextmanager
import random
import numpy as np
import torch


@contextmanager
def preserve_rng(device=None):
    """Frozen extraction/new-module initialization must not shift diffusion noise."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # Never touch every visible GPU from every DDP rank.
    device = None if device is None else torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device is not None and device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
