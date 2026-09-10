"""Tensor contract at the frozen Track4World / trainable tokenizer boundary."""
import torch


def validate_raw_geometry(raw, *, batched=True, views=None, length=None, points=None, fp32=False):
    keys = {"scene", "scene_aux", "scene_valid", "camera", "camera_aux", "camera_valid",
            "track", "track_aux", "track_valid"}
    if not isinstance(raw, dict) or set(raw) != keys or not all(torch.is_tensor(v) for v in raw.values()):
        raise ValueError(f"Raw geometry must contain exactly these tensor keys: {sorted(keys)}")
    scene, track = raw["scene"], raw["track"]
    if scene.ndim != (4 if batched else 3) or track.ndim != (5 if batched else 4):
        raise ValueError("Invalid raw scene/track rank")
    prefix = (scene.shape[0],) if batched else ()
    v, p = scene.shape[-3:-1]
    l = track.shape[-3]
    if min((*prefix, v, p, l)) < 1:
        raise ValueError("Raw geometry must be nonempty")
    if any(expected is not None and actual != expected
           for actual, expected in ((v, views), (l, length), (p, points))):
        raise ValueError("Raw geometry views/history length/grid mismatch")
    shapes = {"scene": (v, p, 1024), "scene_aux": (v, p, 6), "scene_valid": (v, p),
              "camera": (v, 3072), "camera_aux": (v, 13), "camera_valid": (v,),
              "track": (v, l, p, 256), "track_aux": (v, l, p, 11), "track_valid": (v, l, p)}
    for key, shape in shapes.items():
        value = raw[key]
        if tuple(value.shape) != prefix + shape:
            raise ValueError(f"Invalid raw geometry shape for {key}: {tuple(value.shape)} != {prefix + shape}")
        if key.endswith("_valid"):
            if value.dtype != torch.bool:
                raise ValueError(f"{key} must be bool")
        elif not value.is_floating_point() or (fp32 and value.dtype != torch.float32):
            raise ValueError(f"{key} must be {'float32' if fp32 else 'floating point'}")
    # Invalid/unreliable points may contain NaN; masks and the tokenizer's
    # nan_to_num handling intentionally remain identical to online extraction.
