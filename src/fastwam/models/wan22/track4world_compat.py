"""Scoped local-DA3 loading for an unmodified official Track4World checkout."""

from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from types import SimpleNamespace


_DA3_LOAD_LOCK = RLock()


@contextmanager
def local_da3_loading(model_module, directory):
    """Redirect only Track4World's DA3 factory, and restore it on every exit.

    Upstream hard-codes a Hub ID and has no constructor path argument. Do not
    patch the user's checkout or the global Hugging Face loader. A local folder
    plus local_files_only=True also prevents a silent network fallback.
    """
    directory = Path(directory).expanduser().resolve()
    for name in ("config.json", "model.safetensors"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Missing local DA3 file: {directory / name}")
    with _DA3_LOAD_LOCK:
        original = model_module.DepthAnything3

        def from_pretrained(_upstream_source, *args, **kwargs):
            kwargs["local_files_only"] = True
            return original.from_pretrained(str(directory), *args, **kwargs)

        model_module.DepthAnything3 = SimpleNamespace(from_pretrained=from_pretrained)
        try:
            yield
        finally:
            model_module.DepthAnything3 = original
