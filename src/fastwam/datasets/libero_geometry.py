"""Real LIBERO demonstrations and rollout history for online geometry experiments."""

from collections import deque
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
CAMERA_KEYS = ("agentview_rgb", "eye_in_hand_rgb")


def resize_rgb(images, size):
    """RGB uint8 NHWC -> float NCHW [0,1], shared by history and observation."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float() / 255.0
    return F.interpolate(x, (size, size), mode="bilinear", align_corners=False, antialias=True)


class LiberoHistoryBuffer:
    def __init__(self, length=8, stride=1, image_size=256, fps=20):
        if length < 1 or stride < 1 or fps <= 0:
            raise ValueError("Invalid history length/stride/fps")
        self.length, self.stride, self.image_size, self.fps = length, stride, image_size, fps
        self.frames = deque(maxlen=(length - 1) * stride + 1)
        self.times = deque(maxlen=self.frames.maxlen)

    def reset(self):
        self.frames.clear()
        self.times.clear()

    def append(self, images, step):
        """images: two already orientation-corrected RGB images, external then wrist."""
        if self.times and step <= self.times[-1]:
            raise ValueError("History steps must be strictly increasing; reset on each episode")
        self.frames.append(resize_rgb(np.stack(images), self.image_size))
        self.times.append(step)

    def inputs(self):
        if not self.frames:
            raise ValueError("Cannot infer with an empty observation history")
        offsets = torch.arange(-(self.length - 1), 1) * self.stride
        indices = offsets + len(self.frames) - 1
        valid = indices >= 0
        selected = indices.clamp_min(0).tolist()
        images = torch.stack([self.frames[i] for i in selected], dim=1)[None]
        times = torch.tensor([self.times[i] - self.times[-1] for i in selected], dtype=torch.float32) / self.fps
        return {"history_images": images, "history_timestamps": times[None], "history_valid": valid[None]}


class LiberoHDF5HistoryDataset(Dataset):
    """Read only causal RGB history online; future images are separate supervision.

    This is a small-experiment reader for the original LIBERO HDF5 release, NOT a
    claim to reproduce FastWAM's higher-resolution re-rendered LeRobot dataset.
    It never persists Track4World outputs. Text embeddings are permitted caches.
    """
    def __init__(self, files, stats_path, text_cache_dir, episode_ids=None,
                 horizon=32, video_stride=4, history_length=8, history_stride=1,
                 image_size=224, history_image_size=256, sample_stride=4,
                 rotate_180=True, context_len=128, load_history=True, history_only=False):
        if horizon % video_stride or (horizon // video_stride) % 4:
            raise ValueError("Future video must have 4k+1 frames")
        if not 1 <= history_length <= 8 or history_stride < 1 or sample_stride < 1:
            raise ValueError("Invalid history/sample configuration")
        self.files = [str(Path(p).resolve()) for p in files]
        self.stats_path, self.text_cache_dir = str(stats_path), Path(text_cache_dir)
        self.horizon, self.video_stride = horizon, video_stride
        self.history_length, self.history_stride = history_length, history_stride
        self.image_size, self.history_image_size = image_size, history_image_size
        self.rotate_180, self.context_len = rotate_180, context_len
        self.load_history = load_history
        self.history_only = history_only
        with open(stats_path) as handle:
            self.stats = json.load(handle)
        self.samples, self.instructions, self.fps = [], {}, {}
        self.episode_lengths = {}
        for path in self.files:
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                info = json.loads(data.attrs["problem_info"])
                self.instructions[path] = info["language_instruction"]
                env_args = json.loads(data.attrs["env_args"])
                self.fps[path] = float(env_args["env_kwargs"].get("control_freq", 20))
                names = sorted(data.keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))
                if episode_ids is not None:
                    names = [name for name in names if int(name.rsplit("_", 1)[-1]) in episode_ids]
                for name in names:
                    n = len(data[name]["actions"])
                    self.episode_lengths[(path, name)] = n
                    # Extraction can include episode tails. Training still
                    # requires full future targets, with no cross-episode padding.
                    stop = n if history_only else n - horizon
                    for t in range(0, stop, sample_stride):
                        self.samples.append((path, name, t))
        if not self.samples:
            raise ValueError("No LIBERO windows available for the requested episodes/horizon")

    def __len__(self):
        return len(self.samples)

    @property
    def prompts(self):
        return sorted({PROMPT.format(task=x) for x in self.instructions.values()})

    def context_path(self, prompt):
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return self.text_cache_dir / f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"

    def _normalize(self, x, kind):
        stats = self.stats[kind]["default"]
        low, high = torch.tensor(stats["global_min"]), torch.tensor(stats["global_max"])
        span = high - low
        constant = span < 1e-4
        scale = 2 / torch.where(constant, torch.full_like(span, 2), span)
        offset = torch.where(constant, -low, -1 - scale * low)
        return (x * scale + offset).clamp(-5, 5)

    def history_metadata(self, index):
        """Stable causal window identity; does not read images or future targets."""
        path, episode, t = self.samples[index]
        history_indices = t + np.arange(-(self.history_length - 1), 1) * self.history_stride
        history_valid = history_indices >= 0
        clamped_history = history_indices.clip(min=0)
        return {"source": path, "episode": episode, "decision_frame": int(t),
                "frame_indices": clamped_history.tolist(), "valid": history_valid.tolist(),
                "timestamps": ((clamped_history - t) / self.fps[path]).tolist()}

    def history_item(self, index):
        """Extraction-only reader: no future RGB, actions, proprio or T5 required."""
        meta = self.history_metadata(index)
        t = meta["decision_frame"]
        indices = np.asarray(meta["frame_indices"])
        history = []
        with h5py.File(meta["source"], "r") as handle:
            for key in CAMERA_KEYS:
                first = int(indices[0])
                past = handle[f"data/{meta['episode']}/obs/{key}"][first:t + 1][indices - first]
                if self.rotate_180:
                    past = past[:, ::-1, ::-1]
                history.append(resize_rgb(past, self.history_image_size))
        return {"history_images": torch.stack(history),
                "history_timestamps": torch.tensor(meta["timestamps"], dtype=torch.float32),
                "history_valid": torch.tensor(meta["valid"], dtype=torch.bool)}

    def __getitem__(self, index):
        if self.history_only:
            return self.history_item(index)
        path, episode, t = self.samples[index]
        future_indices = np.arange(t, t + self.horizon + 1, self.video_stride)
        video = []
        with h5py.File(path, "r") as handle:
            demo = handle[f"data/{episode}"]
            for key in CAMERA_KEYS:
                future = demo[f"obs/{key}"][t:t + self.horizon + 1:self.video_stride]
                if self.rotate_180:
                    future = future[:, ::-1, ::-1]
                video.append(resize_rgb(future, self.image_size))
            action = torch.from_numpy(demo["actions"][t:t + self.horizon].astype(np.float32))
            # Match the policy checkpoint convention: 0=closed, 1=open.
            action[:, -1] = (1 - action[:, -1]) / 2
            state = np.concatenate([
                demo["obs/ee_pos"][t:t + self.horizon],
                demo["obs/ee_ori"][t:t + self.horizon],
                demo["obs/gripper_states"][t:t + self.horizon],
            ], axis=-1).astype(np.float32)
        prompt = PROMPT.format(task=self.instructions[path])
        text_path = self.context_path(prompt)
        if not text_path.is_file():
            raise FileNotFoundError(f"Missing T5 text context: {text_path}; run the experiment's --prepare-text step")
        text = torch.load(text_path, map_location="cpu", weights_only=True)
        context, text_mask = text["context"].clone(), text["mask"].bool()
        context[~text_mask] = 0
        # Preserve the original FastWAM convention: zero padding remains visible.
        text_mask = torch.ones_like(text_mask)
        result = {
            "video": (torch.cat(video, dim=-1).permute(1, 0, 2, 3) * 2 - 1),
            "action": self._normalize(action, "action"),
            "proprio": self._normalize(torch.from_numpy(state), "state"),
            "context": context, "context_mask": text_mask, "prompt": prompt,
            "image_is_pad": torch.zeros(len(future_indices), dtype=torch.bool),
            "action_is_pad": torch.zeros(self.horizon, dtype=torch.bool),
            "source_episode": f"{path}:{episode}", "decision_frame": t,
        }
        if self.load_history:
            result.update(self.history_item(index))
        return result
