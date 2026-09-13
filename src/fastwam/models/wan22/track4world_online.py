"""Online, frozen Track4World feature extraction, one independent clip per view.

No feature files, no episode-wide encoding, and no reuse of Track4World eval_dict.
The tracker is deliberately NOT a child nn.Module of FastWAM: it remains frozen,
may live on a separate GPU, and is not copied into policy checkpoints.
"""

import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


def require_module_origin(module, expected_file):
    """Reject a stale/imported Track4World package from a different checkout."""
    actual = getattr(module, "__file__", None)
    expected = Path(expected_file).resolve()
    if actual is None or Path(actual).resolve() != expected:
        raise RuntimeError(
            "Imported Track4World from the wrong checkout: "
            f"expected={expected}, actual={actual}. Start a fresh Python process and "
            "put the configured track4world_repo first on sys.path."
        )


def transform_endpoints_to_current_camera(endpoints, poses):
    """T,P,3 target-camera endpoints -> one fixed current-camera frame."""
    transform = torch.linalg.inv(poses[-1].float())[None] @ poses.float()
    return (torch.einsum("tij,tpj->tpi", transform[:, :3, :3], endpoints.float())
            + transform[:, None, :3, 3])


class OnlineTrack4WorldExtractor:
    def __init__(self, repo_path, checkpoint_path, da3_path, device="cuda:0",
                 image_size=256, grid_size=8, iters=4, confidence_threshold=0.25):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("The upstream Track4World implementation requires CUDA")
        if image_size < 256 or image_size % 64:
            raise ValueError("Track4World image_size must be >=256 and divisible by 64")
        self.image_size, self.grid_size, self.iters = image_size, grid_size, iters
        self.confidence_threshold = confidence_threshold
        self.calls = 0
        self.last_seconds = 0.0
        self.last_quality = {}
        repo = Path(repo_path).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        da3 = Path(da3_path).resolve()
        for path in (repo / "track4world/nets/model.py", checkpoint,
                     da3 / "config.json", da3 / "model.safetensors"):
            if not path.is_file():
                raise FileNotFoundError(f"Online geometry requires local file: {path}")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from track4world.nets import model as track4world_model
        from .track4world_compat import local_da3_loading

        require_module_origin(track4world_model, repo / "track4world/nets/model.py")

        with local_da3_loading(track4world_model, da3):
            model = track4world_model.Track4World(use_model="depthanythingv3", use_3d=True, seqlen=16)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        missing, unexpected = model.load_pretrained_with_remap(state)
        missing = [k for k in missing if not k.startswith("backbone.model.da3_metric.")]
        if missing or unexpected:
            raise RuntimeError(f"Unexpected tracker checkpoint mismatch: missing={missing[:12]}, unexpected={unexpected[:12]}")
        del state
        self.model = model.eval().requires_grad_(False).to(self.device)
        self._captured = {}
        # These hooks see observation-dependent hidden states, not token templates.
        self._hooks = [
            self.model.backbone.model.da3.cam_dec.register_forward_pre_hook(self._camera_hook),
            self.model.backbone.model.register_forward_hook(self._backbone_hook),
            self.model.flow3d_head.register_forward_hook(self._track_hook),
        ]

    def _camera_hook(self, module, args):
        self._captured["camera"] = args[0].detach()

    def _backbone_hook(self, module, args, output):
        self._captured["scene"] = output["feats"][..., -1024:].detach()
        self._captured["intrinsics"] = output["intrinsics"].detach()

    def _track_hook(self, module, args, output):
        self._captured["track"] = output[1].detach()

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    @staticmethod
    def _sample(maps, grid):
        """N,C,H,W -> N,P,C, sampling identical anchor pixels across target times."""
        sampled = F.grid_sample(maps.float(), grid.expand(maps.shape[0], -1, -1, -1),
                                align_corners=False, padding_mode="border")
        return sampled.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def _extract_view(self, images, timestamps):
        length = images.shape[0]
        if not 1 <= length <= 8:
            raise ValueError("The initial online implementation supports 1..8 real history frames")
        size = self.image_size
        rgb = F.interpolate(images.float().to(self.device), (size, size), mode="bilinear",
                            align_corners=False, antialias=True)
        normalized = (rgb - self.model.image_mean[0]) / self.model.image_std[0]
        self._captured.clear()
        with torch.autocast("cuda", dtype=torch.float16):
            fmaps, context, detail, pms, points, quality, _, poses = self.model.forward_point(
                normalized[None], current_batch_size=1, for_flow=True
            )
        # Clone outside upstream inference_mode: trainable projections must be able
        # to save these frozen tensors for their own backward pass.
        scene = self._captured["scene"].float().clone()
        camera = self._captured["camera"][:, -1].float().clone()[0]
        intrinsics = self._captured["intrinsics"][0, -1].float().clone()
        side = round(math.sqrt(scene.shape[-2]))
        if side * side != scene.shape[-2]:
            raise ValueError("Expected square DA3 patch grid for square online input")
        axis = (torch.arange(self.grid_size, device=self.device).float() + 0.5) / self.grid_size * 2 - 1
        gy, gx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack([gx, gy], dim=-1)[None]
        scene_map = scene[0, -1].T.reshape(1, 1024, side, side)
        scene_features = self._sample(scene_map, grid)[0]
        scene_xyz = self._sample(points[-1:], grid)[0]
        scene_conf = self._sample(quality[-1:], grid)[0, :, 0]
        scale = scene_xyz[:, 2][torch.isfinite(scene_xyz[:, 2]) & (scene_xyz[:, 2] > 0)]
        scale = scale.median().clamp_min(1e-4) if scale.numel() else torch.ones((), device=self.device)
        scene_valid = (scene_conf > self.model.mask_threshold) & torch.isfinite(scene_xyz).all(-1) & (scene_xyz[:, 2] > 0)
        scene_aux = torch.cat([scene_xyz / scale, grid.flatten(1, 2)[0], torch.log1p(scene_conf.clamp_min(0))[:, None]], -1)
        poses = poses.float()
        relative = torch.linalg.inv(poses[0]) @ poses[-1]
        # K is in the DA3 resized pixel grid, not the public infer() averaged K.
        da3_size = size // 14 * 14
        k = torch.stack([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]) / da3_size
        camera_aux = torch.cat([k, relative[:3, :2].T.flatten(), relative[:3, 3] / scale])
        num_points = self.grid_size ** 2
        track_hidden = self._sample(detail[:1], grid)
        endpoints = self._sample(pms[:1], grid)
        track_conf = torch.ones(1, num_points, device=self.device)
        if length > 1:
            h8, w8 = fmaps.shape[-2:]
            targets = length - 1
            zeros = lambda channels: torch.zeros(targets, channels, h8, w8, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                result = self.model.forward_window_unified(
                    fmap1_single=fmaps[:1], fmap2=fmaps[None, 1:],
                    cxt1_single=context[:1], cxt2=context[None, 1:],
                    pm1_single=pms[:1], pm2=pms[None, 1:],
                    fmaps3d_detail1_single=detail[:1], fmaps3d_detail2=detail[None, 1:],
                    visconfs8=zeros(2), flow2ds8=zeros(2), flow3ds8=zeros(3),
                    iters=self.iters, is_training=False, tracking3d=True,
                )
            # Result[4] is low-resolution residual. Add source camera points to
            # obtain TARGET-camera endpoints; never call this residual world flow.
            endpoint_maps = result[4].float() + pms[:1].float()
            endpoints = torch.cat([endpoints, self._sample(endpoint_maps, grid)], dim=0)
            track_hidden = torch.cat([track_hidden, self._sample(self._captured["track"], grid)], dim=0)
            scores = self._sample(result[2][-1].float().sigmoid(), grid).prod(dim=-1)
            track_conf = torch.cat([track_conf, scores], dim=0)
            del result
        trajectory = transform_endpoints_to_current_camera(endpoints, poses)
        trajectory = trajectory / scale
        timestamps = timestamps.to(device=self.device, dtype=torch.float32)
        velocity = torch.zeros_like(trajectory)
        if length > 1:
            dt = timestamps[1:] - timestamps[:-1]
            if (dt <= 0).any():
                raise ValueError("Real history timestamps must be strictly increasing")
            velocity[1:] = (trajectory[1:] - trajectory[:-1]) / dt[:, None, None]
        relative_time = (timestamps - timestamps[-1])[:, None, None].expand(-1, num_points, 1)
        track_aux = torch.cat([trajectory, trajectory - trajectory[-1:], velocity,
                               relative_time, track_conf[..., None]], dim=-1)
        anchor_valid = self._sample(quality[:1], grid)[0, :, 0] > self.model.mask_threshold
        track_valid = (track_conf >= self.confidence_threshold) & anchor_valid[None] & torch.isfinite(trajectory).all(-1) & (endpoints[..., 2] > 0)
        if length == 1:
            track_valid.zero_()  # No measured motion/history at episode start.
        output = dict(scene=scene_features, scene_aux=scene_aux, scene_valid=scene_valid,
                      camera=camera, camera_aux=camera_aux,
                      camera_valid=torch.isfinite(camera).all() & torch.isfinite(camera_aux).all(),
                      track=track_hidden, track_aux=track_aux, track_valid=track_valid)
        self._captured.clear()
        return output
    @torch.no_grad()
    def __call__(self, images, timestamps, valid, output_device="cpu"):
        if images.ndim != 6 or images.shape[3] != 3:
            raise ValueError("history_images must be [B,V,L,3,H,W] RGB float in [0,1]")
        b, views, length = images.shape[:3]
        if timestamps.shape != (b, length) or valid.shape != (b, length):
            raise ValueError("history_timestamps/history_valid must be [B,L]")
        if b < 1 or views < 1 or length < 1 or not torch.isfinite(timestamps).all():
            raise ValueError("History must be nonempty with finite timestamps")
        if (timestamps > 1e-6).any() or (timestamps[:, -1].abs() > 1e-6).any() or not valid[:, -1].all():
            raise ValueError("History must end at current observation and contain no future timestamps")
        if (valid[:, :-1].bool() & ~valid[:, 1:].bool()).any():
            raise ValueError("history_valid must be one invalid prefix followed by valid history")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1.001:
            raise ValueError("history_images must be finite RGB in [0,1]")
        start = time.monotonic()
        batch = []
        # An outer trainer autocast must not change camera transforms or motion
        # arithmetic relative to the standalone offline extractor. The upstream
        # network calls explicitly opt into fp16 inside _extract_view.
        with torch.cuda.device(self.device), torch.autocast("cuda", enabled=False):
            for bi in range(b):
                view_results = []
                keep = valid[bi].bool()
                for vi in range(views):
                    result = self._extract_view(images[bi, vi, keep], timestamps[bi, keep])
                    # Left-pad raw history, never expose replicated frames as real observations.
                    pad = length - int(keep.sum())
                    if pad:
                        for key in ("track", "track_aux", "track_valid"):
                            value = result[key]
                            result[key] = torch.cat([value.new_zeros((pad,) + value.shape[1:]), value], dim=0)
                    view_results.append(result)
                batch.append({k: torch.stack([r[k] for r in view_results]) for k in view_results[0]})
            output = {k: torch.stack([r[k] for r in batch]).to(output_device).clone() for k in batch[0]}
        self.calls += 1
        self.last_seconds = time.monotonic() - start
        self.last_quality = {k: float(output[k].float().mean()) for k in ("scene_valid", "camera_valid", "track_valid")}
        return output
