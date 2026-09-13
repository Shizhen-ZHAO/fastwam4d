"""Trainable, three-bank geometry conditioning; no tracker dependency here."""

import torch
from torch import nn
from torch.nn import functional as F


BANKS = ("scene", "camera", "track")


class GeometryCrossAttention(nn.Module):
    def __init__(self, query_dim, memory_dim=512, inner_dim=512, heads=8):
        super().__init__()
        if inner_dim % heads:
            raise ValueError("inner_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = inner_dim // heads
        self.q_norm = nn.LayerNorm(query_dim)
        self.m_norm = nn.LayerNorm(memory_dim)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(memory_dim, 2 * inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)

    def _split(self, x):
        return x.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def project_memory(self, memory):
        k, v = self.to_kv(self.m_norm(memory)).chunk(2, dim=-1)
        return self._split(k), self._split(v)

    def forward(self, x, kv, valid, query_bias=None):
        q = self.to_q(self.q_norm(x.to(self.to_q.weight.dtype)))
        if query_bias is not None:
            q = q + query_bias.to(q)
        q = self._split(q)
        k, v = kv
        if valid.ndim == 2:
            attn_mask = valid[:, None, None, :]
        elif valid.ndim == 3:
            attn_mask = valid[:, None, :, :]
        else:
            raise ValueError("Geometry mask must be [B,M] or [B,Q,M]")
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0
        )
        return self.to_out(y.transpose(1, 2).flatten(2))


class ThreeWayGeometryAdapter(nn.Module):
    def __init__(self, query_dim=1024, memory_dim=512, inner_dim=512, heads=8):
        super().__init__()
        self.branches = nn.ModuleDict({
            name: GeometryCrossAttention(query_dim, memory_dim, inner_dim, heads)
            for name in BANKS
        })
        # Only gates are zero initialized. Zeroing both gates and outputs kills learning.
        self.gates = nn.Parameter(torch.zeros(len(BANKS)))

    def project_memories(self, memories):
        return {name: self.branches[name].project_memory(memories[name]) for name in BANKS}

    def forward(self, x, payload, query_bias=None):
        kv = payload.get("kv")
        if kv is None:
            kv = self.project_memories(payload["memories"])
        delta = torch.zeros_like(x, dtype=self.gates.dtype)
        for i, name in enumerate(BANKS):
            delta = delta + self.gates[i].tanh() * self.branches[name](
                x, kv[name], payload["masks"][name], query_bias=query_bias
            )
        return (x.to(delta.dtype) + delta).to(x.dtype)


class VAELatentGeometryAdapter(nn.Module):
    """One residual fusion on the CURRENT observation latent, before patchify.

    Assumes horizontally concatenated equal-width camera views. Positions are
    local to each view. No self-attention/time mixing, no future latent input.
    """
    def __init__(self, latent_channels=48, memory_dim=512, inner_dim=512, heads=8,
                 num_views=2, same_view_only=True):
        super().__init__()
        if num_views < 1:
            raise ValueError("num_views must be positive")
        self.latent_channels = latent_channels
        self.num_views = num_views
        self.same_view_only = same_view_only
        self.attention = ThreeWayGeometryAdapter(latent_channels, memory_dim, inner_dim, heads)
        self.position = nn.Sequential(nn.Linear(2, inner_dim), nn.SiLU(), nn.Linear(inner_dim, inner_dim))
        self.view_embedding = nn.Embedding(num_views, inner_dim)
        self.calls = 0
        self.last_metrics = {}

    @property
    def gates(self):
        return self.attention.gates

    def _query_layout(self, height, width, device, dtype):
        if width % self.num_views:
            raise ValueError("Latent width must divide into equal horizontal camera views")
        view_width = width // self.num_views
        column = torch.arange(width, device=device)
        view_ids = (column // view_width)[None].expand(height, -1).reshape(-1)
        x = (column.remainder(view_width).to(dtype) + 0.5) / view_width * 2 - 1
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height * 2 - 1
        coordinates = torch.stack([x[None].expand(height, -1), y[:, None].expand(-1, width)], -1).reshape(-1, 2)
        bias = self.position(coordinates) + self.view_embedding(view_ids)
        return view_ids, bias[None]

    def forward(self, latent, payload):
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels or latent.shape[2] != 1:
            raise ValueError("VAE geometry accepts only current [B,C,1,H,W] observation latent")
        b, channels, _, height, width = latent.shape
        tokens = latent[:, :, 0].flatten(2).transpose(1, 2)
        views, bias = self._query_layout(height, width, latent.device, self.view_embedding.weight.dtype)
        masks = {}
        for name in BANKS:
            valid = payload["masks"][name]
            if valid.ndim != 2 or valid.shape[0] != b:
                raise ValueError("Expected batched geometry memory masks")
            count = valid.shape[1] - 1  # index 0 is always-valid null, then view-major tokens
            if count < self.num_views or count % self.num_views:
                raise ValueError("Geometry bank must contain null + equal-size per-view memories")
            if self.same_view_only:
                memory_views = torch.arange(self.num_views, device=latent.device).repeat_interleave(count // self.num_views)
                allowed = torch.cat([torch.ones(height * width, 1, dtype=torch.bool, device=latent.device),
                                     views[:, None] == memory_views[None]], dim=1)
                masks[name] = valid[:, None, :] & allowed[None]
            else:
                masks[name] = valid
        conditioned = self.attention(tokens, {"memories": payload["memories"], "masks": masks}, query_bias=bias)
        output = conditioned.transpose(1, 2).reshape(b, channels, 1, height, width)
        self.calls += 1
        self.last_metrics = {"latent_shape": list(latent.shape), "query_tokens": height * width,
                             "delta_rms": float((output.detach().float() - latent.detach().float()).square().mean().sqrt()),
                             "latent_rms": float(latent.detach().float().square().mean().sqrt())}
        return output


class GeometryTokenizer(nn.Module):
    """Per-view current scene/camera; per-track temporal encoding of causal history.

    Frozen features use B,V,P,D (scene), B,V,D (camera), B,V,L,P,D (track).
    An always-valid null token handles empty/occluded banks without NaNs.
    """
    def __init__(self, memory_dim=512, heads=8, temporal_layers=2, num_views=2):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_views = num_views
        self.scene = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, memory_dim))
        self.camera = nn.Sequential(nn.LayerNorm(3072), nn.Linear(3072, memory_dim))
        self.track = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, memory_dim))
        self.scene_aux = nn.Sequential(nn.Linear(6, memory_dim), nn.SiLU(), nn.Linear(memory_dim, memory_dim))
        self.camera_aux = nn.Sequential(nn.Linear(13, memory_dim), nn.SiLU(), nn.Linear(memory_dim, memory_dim))
        self.track_aux = nn.Sequential(nn.Linear(11, memory_dim), nn.SiLU(), nn.Linear(memory_dim, memory_dim))
        self.view_embedding = nn.Embedding(num_views, memory_dim)
        layer = nn.TransformerEncoderLayer(
            memory_dim, heads, 2 * memory_dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, temporal_layers, enable_nested_tensor=False)
        self.null_tokens = nn.ParameterDict({name: nn.Parameter(torch.zeros(1, 1, memory_dim)) for name in BANKS})

    def forward(self, raw):
        dtype = self.view_embedding.weight.dtype
        device = self.view_embedding.weight.device
        raw = {k: v.to(device=device, dtype=(torch.bool if v.dtype == torch.bool else dtype))
               for k, v in raw.items()}
        b, views, _, _ = raw["scene"].shape
        if views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {views}")
        view = self.view_embedding(torch.arange(views, device=device))
        clean = lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        scene = self.scene(clean(raw["scene"])) + self.scene_aux(clean(raw["scene_aux"])) + view[None, :, None]
        camera = self.camera(clean(raw["camera"])) + self.camera_aux(clean(raw["camera_aux"])) + view[None]
        track = self.track(clean(raw["track"])) + self.track_aux(clean(raw["track_aux"])) + view[None, :, None, None]
        _, _, length, points, dim = track.shape
        track = track.permute(0, 1, 3, 2, 4).reshape(b * views * points, length, dim)
        valid = raw["track_valid"].permute(0, 1, 3, 2).reshape(b * views * points, length)
        any_valid = valid.any(dim=1)
        # Transformer attention must never see an entirely masked row.
        safe_valid = valid.clone()
        safe_valid[~any_valid, 0] = True
        track = self.temporal(track, src_key_padding_mask=~safe_valid)
        track = (track * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        memories = {"scene": scene.flatten(1, 2), "camera": camera,
                    "track": track.reshape(b, views * points, dim)}
        masks = {"scene": raw["scene_valid"].flatten(1), "camera": raw["camera_valid"],
                 "track": any_valid.reshape(b, views * points)}
        for name in BANKS:
            memories[name] = torch.cat([self.null_tokens[name].expand(b, -1, -1), memories[name]], dim=1)
            masks[name] = torch.cat([torch.ones(b, 1, device=device, dtype=torch.bool), masks[name]], dim=1)
        return {"memories": memories, "masks": masks}
