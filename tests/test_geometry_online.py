import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from fastwam.datasets.libero_geometry import LiberoHistoryBuffer, LiberoHDF5HistoryDataset
from fastwam.models.wan22.geometry_adapter import GeometryTokenizer, ThreeWayGeometryAdapter
from fastwam.models.wan22.track4world_online import OnlineTrack4WorldExtractor


def raw_features(b=2, views=2, length=4, points=4):
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


def test_zero_gate_exact_baseline_and_first_gradient():
    tokenizer = GeometryTokenizer(memory_dim=32, heads=2, temporal_layers=1)
    adapter = ThreeWayGeometryAdapter(query_dim=48, memory_dim=32, inner_dim=32, heads=2)
    x = torch.randn(2, 7, 48, requires_grad=True)
    payload = tokenizer(raw_features())
    out = adapter(x, payload)
    torch.testing.assert_close(out, x, rtol=0, atol=0)
    out.square().mean().backward()
    assert adapter.gates.grad.abs().sum() > 0
    assert torch.isfinite(adapter.gates.grad).all()


def test_cached_and_uncached_equivalent_with_nonzero_gates():
    torch.manual_seed(10)
    tokenizer = GeometryTokenizer(memory_dim=32, heads=2, temporal_layers=1)
    adapter = ThreeWayGeometryAdapter(query_dim=48, memory_dim=32, inner_dim=32, heads=2)
    adapter.gates.data.fill_(0.1)
    payload = tokenizer(raw_features())
    x = torch.randn(2, 7, 48)
    cached = {"kv": adapter.project_memories(payload["memories"]), "masks": payload["masks"]}
    torch.testing.assert_close(adapter(x, payload), adapter(x, cached))


def test_empty_banks_and_nonfinite_features_are_safe():
    raw = raw_features()
    for key in ("scene_valid", "camera_valid", "track_valid"):
        raw[key].zero_()
    raw["scene"].fill_(float("nan"))
    raw["track_aux"].fill_(float("inf"))
    tokenizer = GeometryTokenizer(memory_dim=32, heads=2, temporal_layers=1)
    adapter = ThreeWayGeometryAdapter(query_dim=48, memory_dim=32, inner_dim=32, heads=2)
    adapter.gates.data.fill_(0.2)
    payload = tokenizer(raw)
    assert all(mask[:, 0].all() for mask in payload["masks"].values())
    out = adapter(torch.randn(2, 5, 48), payload)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in tokenizer.parameters())


def test_all_three_branches_receive_gradients():
    tokenizer = GeometryTokenizer(memory_dim=32, heads=2, temporal_layers=1)
    adapter = ThreeWayGeometryAdapter(query_dim=48, memory_dim=32, inner_dim=32, heads=2)
    adapter.gates.data.fill_(0.1)
    adapter(torch.randn(2, 5, 48), tokenizer(raw_features())).square().mean().backward()
    for name in ("scene", "camera", "track"):
        assert adapter.branches[name].to_kv.weight.grad.abs().sum() > 0
        assert getattr(tokenizer, name)[1].weight.grad.abs().sum() > 0


def test_history_current_alignment_stride_padding_and_reset():
    buffer = LiberoHistoryBuffer(length=4, stride=2, image_size=8, fps=20)
    for step in range(3):
        image = np.full((8, 8, 3), step * 10, dtype=np.uint8)
        buffer.append([image, image], step)
    payload = buffer.inputs()
    assert payload["history_valid"].tolist() == [[False, False, True, True]]
    torch.testing.assert_close(payload["history_timestamps"], torch.tensor([[-0.1, -0.1, -0.1, 0.0]]))
    assert payload["history_images"].shape == (1, 2, 4, 3, 8, 8)
    assert payload["history_images"][0, 0, -1].mean() == pytest.approx(20 / 255)
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append([image, image], 2)
    buffer.reset()
    with pytest.raises(ValueError, match="empty"):
        buffer.inputs()


def test_extractor_rejects_future_before_model_use():
    # Deliberately no model construction/GPU allocation for input validation.
    extractor = object.__new__(OnlineTrack4WorldExtractor)
    with pytest.raises(ValueError, match="future"):
        extractor(torch.zeros(1, 2, 4, 3, 8, 8), torch.tensor([[-2., -1., 0., 1.]]),
                  torch.ones(1, 4, dtype=torch.bool))


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("video_frames", [1, 3])
def test_mot_shared_hook_is_used_by_training_and_cached_action(checkpoint, video_frames):
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.fastwam import FastWAM
    video = WanVideoDiT(has_image_input=False, hidden_dim=64, in_dim=4, ffn_dim=128, out_dim=4, text_dim=48,
                        freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                        attn_head_dim=24, num_layers=2, seperated_timestep=True,
                        video_attention_mask_mode="first_frame_causal", fuse_vae_embedding_in_latents=True)
    action = ActionDiT(action_dim=7, hidden_dim=32, ffn_dim=64, num_heads=2,
                       attn_head_dim=24, num_layers=2, text_dim=48, freq_dim=16, eps=1e-6)
    video.use_gradient_checkpointing = checkpoint
    action.use_gradient_checkpointing = checkpoint
    mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=checkpoint)
    block = action.blocks[0]
    block.geometry_layer = "0"
    block.geometry_adapter = ThreeWayGeometryAdapter(32, 32, 32, 2)
    block.geometry_adapter.gates.data.fill_(0.1)
    tok = GeometryTokenizer(memory_dim=32, heads=2, temporal_layers=1)
    payload = tok(raw_features(b=1))
    geometry = {"0": {"kv": block.geometry_adapter.project_memories(payload["memories"]), "masks": payload["masks"]}}
    context, cmask = torch.randn(1, 3, 48), torch.ones(1, 3, dtype=torch.bool)
    vp = video.pre_dit(x=torch.randn(1, 4, video_frames, 4, 4), timestep=torch.zeros(1),
                       context=context, context_mask=cmask, fuse_vae_embedding_in_latents=True)
    ap = action.pre_dit(torch.randn(1, 5, 7), torch.ones(1), context, cmask)
    nv, na = vp["tokens"].shape[1], ap["tokens"].shape[1]
    nf = vp["meta"]["tokens_per_frame"]
    mask = FastWAM._build_mot_attention_mask(SimpleNamespace(video_expert=video), nv, na, nf, torch.device("cpu"))
    vctx = {"context": vp["context"], "mask": vp["context_mask"]}
    actx = {"context": ap["context"], "mask": ap["context_mask"], "geometry": geometry}
    normal = mot(embeds_all={"video": vp["tokens"], "action": ap["tokens"]}, attention_mask=mask,
                 freqs_all={"video": vp["freqs"], "action": ap["freqs"]},
                 context_all={"video": vctx, "action": actx},
                 t_mod_all={"video": vp["t_mod"], "action": ap["t_mod"]})["action"]
    first_ctx = {"context": vp["context"], "mask": vp["context_mask"][:, :nf]}
    cache = mot.prefill_video_cache(vp["tokens"][:, :nf], vp["freqs"][:nf], vp["t_mod"][:, :nf], first_ctx, mask[:nf, :nf])
    cached_mask = FastWAM._build_mot_attention_mask(SimpleNamespace(video_expert=video), nf, na, nf, torch.device("cpu"))
    cached = mot.forward_action_with_video_cache(ap["tokens"], ap["freqs"], ap["t_mod"], actx, cache, cached_mask, nf)
    torch.testing.assert_close(normal, cached, rtol=2e-5, atol=2e-5)
    if video_frames > 1:
        changed = vp["tokens"].clone()
        changed[:, nf:] = torch.randn_like(changed[:, nf:]) * 100
        altered = mot(embeds_all={"video": changed, "action": ap["tokens"]}, attention_mask=mask,
                      freqs_all={"video": vp["freqs"], "action": ap["freqs"]},
                      context_all={"video": vctx, "action": actx},
                      t_mod_all={"video": vp["t_mod"], "action": ap["t_mod"]})["action"]
        torch.testing.assert_close(normal, altered, rtol=0, atol=0)
    normal.square().mean().backward()
    assert block.geometry_adapter.branches["track"].to_kv.weight.grad.abs().sum() > 0


def test_static_world_points_stay_static_after_wrist_motion_compensation():
    from fastwam.models.wan22.track4world_online import transform_endpoints_to_current_camera
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.arange(3)
    world_points = torch.tensor([[1., 0., 2.], [2., 1., 3.]])
    endpoints = world_points[None] - poses[:, None, :3, 3]
    transformed = transform_endpoints_to_current_camera(endpoints, poses)
    torch.testing.assert_close(transformed, transformed[-1:].expand_as(transformed))


def test_float32_adapter_accepts_bfloat16_policy_hidden():
    adapter = ThreeWayGeometryAdapter(48, 32, 32, 2)
    tokenizer = GeometryTokenizer(memory_dim=32, heads=2, temporal_layers=1)
    adapter.gates.data.fill_(0.01)
    x = torch.randn(2, 5, 48).bfloat16()
    y = adapter(x, tokenizer(raw_features()))
    assert y.dtype == x.dtype
    y.float().square().mean().backward()
    assert adapter.branches["scene"].to_q.weight.grad.abs().sum() > 0


def test_hdf5_history_cannot_see_future_or_other_episode(tmp_path):
    # A synthetic unit-test fixture only; experiments use the real LIBERO release.
    path, stats_path = tmp_path / "unit.hdf5", tmp_path / "stats.json"
    stats_path.write_text(json.dumps({kind: {"default": {"global_min": [-1.] * dim, "global_max": [1.] * dim}}
                                      for kind, dim in (("action", 7), ("state", 8))}))
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "unit fixture"})
        data.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 20}})
        for episode in range(2):
            demo = data.create_group(f"demo_{episode}")
            demo.create_dataset("actions", data=np.zeros((48, 7), np.float32))
            obs = demo.create_group("obs")
            for name in ("agentview_rgb", "eye_in_hand_rgb"):
                frames = np.broadcast_to(np.arange(48, dtype=np.uint8)[:, None, None, None], (48, 8, 8, 3)) + episode * 100
                obs.create_dataset(name, data=frames)
            for name, dim in (("ee_pos", 3), ("ee_ori", 3), ("gripper_states", 2)):
                obs.create_dataset(name, data=np.zeros((48, dim), np.float32))
    dataset = LiberoHDF5HistoryDataset([path], stats_path, tmp_path, image_size=8, history_image_size=8)
    torch.save({"context": torch.zeros(128, 4), "mask": torch.ones(128, dtype=torch.bool)},
               dataset.context_path(dataset.prompts[0]))
    index = next(i for i, (_, episode, t) in enumerate(dataset.samples) if episode == "demo_0" and t == 8)
    before = dataset[index]
    with h5py.File(path, "a") as f:
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            f[f"data/demo_0/obs/{key}"][9:] = 255
            f[f"data/demo_1/obs/{key}"][:] = 255
    after = dataset[index]
    torch.testing.assert_close(before["history_images"], after["history_images"], rtol=0, atol=0)
    assert not torch.equal(before["video"], after["video"])
    assert after["history_timestamps"].max() == 0
    first = dataset[0]
    assert first["history_valid"].tolist() == [False] * 7 + [True]
    assert first["history_images"].max() == 0


def test_adapter_checkpoint_round_trip_and_full_training_guard(tmp_path):
    from fastwam.models.wan22.fastwam import FastWAM
    def bare_policy():
        model = object.__new__(FastWAM)
        torch.nn.Module.__init__(model)
        model.device = torch.device("cpu")
        model.geometry_config = None
        model.action_expert = torch.nn.Module()
        model.action_expert.hidden_dim = 48
        model.action_expert.blocks = torch.nn.ModuleList([torch.nn.Linear(48, 48)])
        model.mot = torch.nn.ModuleDict({"action": model.action_expert})
        return model
    first = bare_policy()
    first.enable_geometry(dict(memory_dim=32, inner_dim=32, heads=2, temporal_layers=1, layers=[0], train_mode="adapters"))
    first.action_expert.blocks[0].geometry_adapter.gates.data.fill_(0.1)
    path = tmp_path / "adapter.pt"
    first.save_geometry_adapter(path, unit_test=True)
    second = bare_policy()
    payload = second.load_geometry_adapter(path)
    for key, value in payload["geometry_adapter"].items():
        torch.testing.assert_close(second.mot.state_dict()[key], value, rtol=0, atol=0)
        assert "geometry" in key
    first.geometry_config["train_mode"] = "action"
    with pytest.raises(ValueError, match="save_checkpoint"):
        first.save_geometry_adapter(tmp_path / "not_allowed.pt")
