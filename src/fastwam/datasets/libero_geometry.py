"""Canonical two-camera LIBERO history for offline and online geometry.

The tensor boundary is identical in both modes:
``history_images`` is float RGB [B, 2, L, 3, S, S] in [0, 1], camera order
is external then wrist, timestamps are causal seconds relative to the current
frame, and invalid replicated prefix frames are marked by ``history_valid``.
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


CAMERA_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)


def resize_history_rgb(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize [V,L,3,H,W] RGB to float [V,L,3,size,size] in [0,1]."""
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError(f"Expected [V,L,3,H,W], got {tuple(images.shape)}")
    if images.dtype == torch.uint8:
        images = images.float().div_(255.0)
    else:
        images = images.float()
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("History must be finite RGB in [0,1]")
    views, length, channels, height, width = images.shape
    flat = images.reshape(views * length, channels, height, width)
    flat = F.interpolate(
        flat,
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return flat.reshape(views, length, channels, int(size), int(size))


class LiberoHistoryBuffer:
    """Online history populated from orientation-corrected simulator RGB."""

    def __init__(self, length: int = 8, stride: int = 1, image_size: int = 256, fps: float = 20):
        if length < 1 or stride < 1 or image_size < 1 or fps <= 0:
            raise ValueError("Invalid history configuration")
        self.length = int(length)
        self.stride = int(stride)
        self.image_size = int(image_size)
        self.fps = float(fps)
        capacity = (self.length - 1) * self.stride + 1
        self.frames = deque(maxlen=capacity)
        self.steps = deque(maxlen=capacity)

    def reset(self):
        self.frames.clear()
        self.steps.clear()

    def append(self, images: Sequence[np.ndarray], step: int):
        if len(images) != len(CAMERA_KEYS):
            raise ValueError(f"Expected {len(CAMERA_KEYS)} camera images")
        if self.steps and int(step) <= self.steps[-1]:
            raise ValueError("History steps must be strictly increasing")
        arrays = []
        for image in images:
            value = np.asarray(image)
            if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ValueError("Online history expects uint8 HWC RGB")
            arrays.append(torch.from_numpy(np.ascontiguousarray(value)).permute(2, 0, 1))
        stacked = torch.stack(arrays)[:, None]
        self.frames.append(resize_history_rgb(stacked, self.image_size)[:, 0])
        self.steps.append(int(step))

    def inputs(self) -> dict[str, torch.Tensor]:
        if not self.frames:
            raise ValueError("Cannot infer with an empty observation history")
        offsets = torch.arange(-(self.length - 1), 1) * self.stride
        indices = offsets + len(self.frames) - 1
        valid = indices >= 0
        selected = indices.clamp_min(0).tolist()
        images = torch.stack([self.frames[index] for index in selected], dim=1)
        timestamps = torch.tensor(
            [self.steps[index] - self.steps[-1] for index in selected],
            dtype=torch.float32,
        ).div_(self.fps)
        return {
            "history_images": images[None],
            "history_timestamps": timestamps[None],
            "history_valid": valid[None],
        }


class LeRobotGeometryHistoryDataset(Dataset):
    """Read causal history from the same LeRobot indices used by FastWAM."""

    def __init__(
        self,
        dataset_dirs: Sequence[str],
        history_length: int = 8,
        history_stride: int = 1,
        image_size: int = 256,
        camera_keys: Sequence[str] = CAMERA_KEYS,
    ):
        from .lerobot.lerobot.lerobot_dataset import MultiLeRobotDataset

        self.dataset_dirs = [str(Path(path).expanduser().resolve()) for path in dataset_dirs]
        if not self.dataset_dirs or any(not Path(path).is_dir() for path in self.dataset_dirs):
            raise FileNotFoundError("Every LeRobot dataset directory must exist")
        self.history_length = int(history_length)
        self.history_stride = int(history_stride)
        self.image_size = int(image_size)
        self.camera_keys = tuple(camera_keys)
        if len(self.camera_keys) != 2 or self.history_length < 1 or self.history_stride < 1:
            raise ValueError("LIBERO geometry requires two cameras and a positive history")

        # Read FPS without opening the parquet/video datasets twice.
        fps_values = {
            float(json.loads((Path(root) / "meta" / "info.json").read_text())["fps"])
            for root in self.dataset_dirs
        }
        if len(fps_values) != 1:
            raise ValueError(f"All LeRobot roots must share one FPS, got {sorted(fps_values)}")
        self.fps = fps_values.pop()
        offsets = torch.arange(-(self.history_length - 1), 1) * self.history_stride
        delta_timestamps = {
            key: (offsets.double() / self.fps).tolist() for key in self.camera_keys
        }
        self.dataset = MultiLeRobotDataset(
            dataset_dirs=self.dataset_dirs,
            delta_timestamps=delta_timestamps,
            download_videos=False,
        )
        self.dataset.set_during_training(True)
        for dataset in self.dataset._datasets:
            missing = set(self.camera_keys) - set(dataset.meta.video_keys)
            if missing:
                raise KeyError(f"Missing LeRobot camera keys in {dataset.root}: {sorted(missing)}")

    def __len__(self):
        return len(self.dataset)

    def _resolve(self, index: int):
        if not 0 <= int(index) < len(self):
            raise IndexError(index)
        remaining = int(index)
        for dataset_id, dataset in enumerate(self.dataset._datasets):
            if remaining < len(dataset):
                row = dataset.hf_dataset[remaining]
                return dataset_id, dataset, remaining, row
            remaining -= len(dataset)
        raise IndexError(index)

    def metadata(self, index: int) -> dict:
        dataset_id, dataset, local_index, row = self._resolve(index)
        episode = int(row["episode_index"])
        frame = int(row["frame_index"])
        offsets = np.arange(-(self.history_length - 1), 1) * self.history_stride
        clamped = np.maximum(offsets, -frame)
        return {
            "dataset_id": dataset_id,
            "dataset_root": str(Path(dataset.root).resolve()),
            "local_index": local_index,
            "episode_index": episode,
            "frame_index": frame,
            "history_frame_indices": (frame + clamped).astype(int).tolist(),
            "history_valid": (frame + offsets >= 0).tolist(),
            "history_timestamps": (clamped / self.fps).astype(float).tolist(),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[int(index)]
        histories = []
        pad_masks = []
        for key in self.camera_keys:
            value = item[key]
            if value.ndim == 3:
                value = value[None]
            if value.ndim != 4 or value.shape[0] != self.history_length or value.shape[1] != 3:
                raise ValueError(f"Unexpected {key} shape: {tuple(value.shape)}")
            histories.append(value)
            pad_masks.append(item[f"{key}_is_pad"].bool())
        for mask in pad_masks[1:]:
            if not torch.equal(mask, pad_masks[0]):
                raise ValueError("Camera history padding masks differ")
        meta = self.metadata(index)
        valid = ~pad_masks[0]
        expected_valid = torch.tensor(meta["history_valid"], dtype=torch.bool)
        if not torch.equal(valid.cpu(), expected_valid):
            raise ValueError("Decoded history padding does not match frame metadata")
        return {
            "history_images": resize_history_rgb(torch.stack(histories), self.image_size),
            "history_timestamps": torch.tensor(meta["history_timestamps"], dtype=torch.float32),
            "history_valid": valid,
            "sample_index": torch.tensor(int(index), dtype=torch.long),
        }
