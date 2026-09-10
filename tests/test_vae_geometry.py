"""CPU unit tests use tiny models/fixtures, NOT the real-data experiment path."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fastwam.models.wan22.geometry_adapter import VAELatentGeometryAdapter, GeometryTokenizer
from fastwam.models.wan22.fastwam import FastWAM
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.models.wan22.mot import MoT


def raw_features(b=2, views=2, length=8, points=4):
    return {
        "scene": torch.randn(b, views, points, 1024),
        "scene_aux": torch.randn(b, views, points, 6),
        "scene_valid": torch.ones(b, views, points, dtype=torch.bool),
        "camera": torch.randn(b, views, 3072),
        "camera_aux": torch.randn(b, views, 13),
        "camera_valid": torch.ones(b, views, dtype=torch.bool),
        "track": torch.randn(b, views, length, points, 256),
        "track_aux": torch.randn(b, views, length, points, 11),
        "track_valid": torch.ones(b, views, length, points, dtype=torch.bool),
    }


class TinyVAE(nn.Module):
    z_dim = 4
    upsampling_factor = 4
    temporal_downsample_factor = 4

    def __init__(self):
        super().__init__()
        self.scale = 1.0
        self.model = SimpleNamespace(z_dim=4, encode=lambda images, scale: self.encode(images))

    def encode(self, images, **kwargs):
        if isinstance(images, list):
            return [self.encode(x[None])[0] for x in images]
        x = F.avg_pool3d(images[:, :, ::4], (1, 4, 4))
        return torch.cat([x, x[:, :1]], dim=1)

    def decode(self, latent, **kwargs):
        return F.interpolate(latent[:, :3], size=(1 + (latent.shape[2] - 1) * 4,
                                                  latent.shape[3] * 4, latent.shape[4] * 4), mode="nearest")


class FixtureExtractor:
    def __init__(self):
        self.raw = raw_features(b=1, views=2, length=8)
        self.calls = 0

    def __call__(self, images, timestamps, valid, output_device="cpu"):
        self.calls += 1
        return {k: v.to(output_device) for k, v in self.raw.items()}


def tiny_policy(checkpoint=False):
    torch.manual_seed(321)
    video = WanVideoDiT(has_image_input=False, hidden_dim=64, in_dim=4, ffn_dim=128, out_dim=4, text_dim=48,
                        freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                        attn_head_dim=24, num_layers=2, seperated_timestep=True,
                        video_attention_mask_mode="first_frame_causal", fuse_vae_embedding_in_latents=True)
    action = ActionDiT(action_dim=7, hidden_dim=32, ffn_dim=64, num_heads=2,
                       attn_head_dim=24, num_layers=2, text_dim=48, freq_dim=16, eps=1e-6)
    video.use_gradient_checkpointing = checkpoint
    action.use_gradient_checkpointing = checkpoint
    model = FastWAM(video, action, MoT({"video": video, "action": action}, checkpoint),
                    TinyVAE(), text_dim=48, device="cpu", loss_lambda_action=0.0)
    model._vae_encode_compiled = lambda images, scale: model.vae.encode(images)
    return model


def enable(model):
    model.enable_geometry(dict(target="vae_latent", latent_channels=4, memory_dim=32, inner_dim=32,
                               heads=2, temporal_layers=1, num_views=2, train_mode="adapters", extractor={}))
    model._geometry_extractor = FixtureExtractor()
    model.configure_geometry_train_mode()


def sample():
    return dict(video=torch.randn(1, 3, 9, 16, 32).clamp(-1, 1), action=torch.randn(1, 32, 7),
                context=torch.randn(1, 3, 48), context_mask=torch.ones(1, 3, dtype=torch.bool),
                history_images=torch.zeros(1, 2, 8, 3, 16, 16),
                history_timestamps=torch.arange(-7, 1)[None].float() / 20,
                history_valid=torch.ones(1, 8, dtype=torch.bool),
                image_is_pad=torch.zeros(1, 9, dtype=torch.bool), action_is_pad=torch.zeros(1, 32, dtype=torch.bool))


def inference_inputs(s):
    return dict(prompt=None, input_image=s["video"][:, :, 0], action_horizon=32,
                context=s["context"], context_mask=s["context_mask"],
                history_images=s["history_images"], history_timestamps=s["history_timestamps"],
                history_valid=s["history_valid"], num_inference_steps=3, seed=123)


def test_latent_adapter_zero_gate_exact_and_rejects_future():
    adapter = VAELatentGeometryAdapter(48, 32, 32, 2)
    payload = GeometryTokenizer(32, 2, 1)(raw_features())
    latent = torch.randn(2, 48, 1, 4, 8).bfloat16()
    result = adapter(latent, payload)
    torch.testing.assert_close(latent, result, atol=0, rtol=0)
    result.float().square().mean().backward()
    assert adapter.gates.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="current"):
        adapter(latent.expand(-1, -1, 3, -1, -1), payload)


def test_same_view_mask_blocks_other_camera_memory():
    adapter = VAELatentGeometryAdapter(48, 32, 32, 2)
    adapter.gates.data.fill_(0.1)
    payload = GeometryTokenizer(32, 2, 1)(raw_features())
    latent = torch.randn(2, 48, 1, 4, 8)
    before = adapter(latent, payload)
    changed = {"memories": {k: v.clone() for k, v in payload["memories"].items()}, "masks": payload["masks"]}
    for value in changed["memories"].values():
        start = 1 + (value.shape[1] - 1) // 2
        value[:, start:] = torch.randn_like(value[:, start:]) * 10
    after = adapter(latent, changed)
    torch.testing.assert_close(before[..., :4], after[..., :4], atol=0, rtol=0)
    assert (before[..., 4:] - after[..., 4:]).abs().max() > 1e-5


def test_zero_gate_preserves_original_video_loss():
    model, s = tiny_policy(), sample()
    torch.manual_seed(1)
    before, _ = model.training_loss(s)
    enable(model)
    torch.manual_seed(1)
    after, _ = model.training_loss(s)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("checkpoint", [False])
def test_video_loss_trains_all_branches_through_frozen_world(checkpoint):
    model, s = tiny_policy(checkpoint), sample()
    enable(model)
    adapter = model.mot.geometry_latent_adapter
    adapter.gates.data.fill_(0.1)
    inputs = model.build_inputs(s)
    assert not inputs["input_latents"].requires_grad
    assert inputs["first_frame_condition"].requires_grad
    torch.testing.assert_close(inputs["input_latents"], model.vae.encode(s["video"]))
    assert not torch.equal(inputs["first_frame_condition"], inputs["first_frame_latents"])
    loss, _ = model.training_loss(s)
    loss.backward()
    for name, branch in adapter.attention.branches.items():
        assert branch.to_kv.weight.grad.abs().sum() > 0
        assert getattr(model.mot.geometry_tokenizer, name)[1].weight.grad.abs().sum() > 0
    assert not any(hasattr(block, "geometry_adapter") for block in model.action_expert.blocks)
    assert all(p.grad is None and not p.requires_grad for p in model.video_expert.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.action_expert.parameters())


def test_train_targets_and_noised_future_do_not_change_with_fusion():
    model, s = tiny_policy(), sample()
    enable(model)
    captured = []
    original = model.video_expert.prepare
    def capture(*args, **kwargs):
        captured.append(kwargs["x"].detach().clone())
        return original(*args, **kwargs)
    model.video_expert.prepare = capture
    targets = []
    original_target = model.train_video_scheduler.training_target
    def target(*args):
        result = original_target(*args)
        targets.append(result.clone())
        return result
    model.train_video_scheduler.training_target = target
    torch.manual_seed(99)
    model.training_loss(s)
    model.mot.geometry_latent_adapter.gates.data.fill_(0.2)
    torch.manual_seed(99)
    model.training_loss(s)
    torch.testing.assert_close(captured[0][:, :, 1:], captured[1][:, :, 1:], atol=0, rtol=0)
    torch.testing.assert_close(targets[0], targets[1], atol=0, rtol=0)
    assert not torch.equal(captured[0][:, :, :1], captured[1][:, :, :1])


def test_joint_cached_match_extract_once_and_decode_original_latent():
    model, s = tiny_policy(), sample()
    enable(model)
    model.mot.geometry_latent_adapter.gates.data.fill_(0.1)
    captured = []
    def decode(latents, **kwargs):
        captured.append(latents.clone())
        return []
    model._decode_latents = decode
    joint = model.infer_joint(**inference_inputs(s), num_video_frames=9)
    assert model._geometry_extractor.calls == 1
    assert model.mot.geometry_latent_adapter.calls == 1
    original = model._encode_input_image_latents_tensor(s["video"][:, :, 0])
    torch.testing.assert_close(captured[0][:, :, :1], original, atol=0, rtol=0)
    cached = model.infer_action(**inference_inputs(s))
    assert model._geometry_extractor.calls == 2
    assert model.mot.geometry_latent_adapter.calls == 2
    torch.testing.assert_close(joint["action"], cached["action"], atol=1e-5, rtol=1e-5)


def test_latent_checkpoint_reload_and_wrong_target_rejected(tmp_path):
    first = tiny_policy()
    enable(first)
    first.mot.geometry_latent_adapter.gates.data.fill_(0.15)
    path = tmp_path / "vae.pt"
    first.save_geometry_adapter(path)
    second = tiny_policy()
    payload = second.load_geometry_adapter(path)
    assert second.geometry_target == "vae_latent"
    assert not second.geometry_layers
    assert any(k.startswith("geometry_latent_adapter.") for k in payload["geometry_adapter"])
    for key, value in payload["geometry_adapter"].items():
        torch.testing.assert_close(second.mot.state_dict()[key], value, atol=0, rtol=0)
    wrong = tiny_policy()
    with pytest.raises(ValueError, match="vae_latent"):
        wrong.enable_geometry(dict(memory_dim=32, inner_dim=32, heads=2, temporal_layers=1, layers=[0]))
