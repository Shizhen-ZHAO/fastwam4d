"""Regression coverage for the real trainer entry point and portable extraction."""

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from accelerate.data_loader import prepare_data_loader

from fastwam.models.wan22.track4world_compat import local_da3_loading
from fastwam.models.wan22.track4world_online import (
    OnlineTrack4WorldExtractor,
    require_module_origin,
)
from fastwam.trainer import Wan22Trainer
from fastwam.utils.samplers import ResumableEpochSampler
from test_vae_geometry import tiny_policy, enable, sample, inference_inputs


def test_public_infer_forwards_online_history_once():
    model = tiny_policy()
    enable(model)
    inputs = inference_inputs(sample())
    public = model.infer(**inputs, num_frames=9)
    assert model._geometry_extractor.calls == 1
    joint = model.infer_joint(**inputs, num_video_frames=9)
    torch.testing.assert_close(public["action"], joint["action"], rtol=0, atol=0)


def test_offline_raw_and_online_history_have_identical_loss_and_gradients():
    model = tiny_policy()
    enable(model)
    model.mot.geometry_latent_adapter.gates.data.fill_(0.1)
    online = sample()
    offline = {key: value for key, value in online.items() if not key.startswith("history_")}
    offline["geometry_raw"] = {key: value.clone() for key, value in model._geometry_extractor.raw.items()}
    outputs = []
    for item in (online, offline):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(125)
        loss, _ = model(item)
        loss.backward()
        gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
        outputs.append((loss.detach(), gradients))
    assert model._geometry_extractor.calls == 1
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_validation_restores_mixed_training_modes(fail):
    model = tiny_policy()
    enable(model)
    before = [m.training for m in model.modules()]
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = model
    trainer.val_dataset = [object()]

    def evaluate(_model):
        assert not any(m.training for m in _model.modules())
        if fail:
            raise RuntimeError("fixture validation failure")
        return {"checked": True}

    trainer._evaluate = evaluate
    if fail:
        with pytest.raises(RuntimeError, match="fixture validation failure"):
            trainer.evaluate()
    else:
        assert trainer.evaluate() == {"checked": True}
    assert [m.training for m in model.modules()] == before


def test_trainer_online_validation_runs_through_public_infer(tmp_path, monkeypatch):
    from accelerate import Accelerator
    model = tiny_policy()
    enable(model)
    item = {key: value[0] for key, value in sample().items()}
    item["prompt"] = "fixture"
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model, trainer.val_dataset = model, [item]
    trainer.accelerator = Accelerator(cpu=True)
    trainer.global_step, trainer.eval_num_inference_steps = 200, 2
    trainer.eval_dir = str(tmp_path)
    # Exercise actual video inference/metrics. Action denormalization requires
    # a real LeRobot processor and is covered separately by the LIBERO smoke.
    original_infer = model.infer

    def video_only(**kwargs):
        result = original_infer(**kwargs)
        result["action"] = None
        return result

    monkeypatch.setattr(model, "infer", video_only)
    monkeypatch.setattr("fastwam.trainer.save_mp4", lambda *args, **kwargs: None)
    before = [m.training for m in model.modules()]
    metrics = trainer.evaluate()
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
    assert model._geometry_extractor.calls == 2  # validation loss + one rollout
    assert [m.training for m in model.modules()] == before


def test_local_da3_redirect_and_restore_even_on_error(tmp_path):
    (tmp_path / "config.json").write_text('{}')
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    calls = []

    class OriginalDA3:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return "local model"

    module = SimpleNamespace(DepthAnything3=OriginalDA3)
    with pytest.raises(RuntimeError, match="constructor failed"):
        with local_da3_loading(module, tmp_path):
            result = module.DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
            assert result == "local model"
            raise RuntimeError("constructor failed")
    assert module.DepthAnything3 is OriginalDA3
    assert calls == [(str(tmp_path.resolve()), {"local_files_only": True})]


def test_track4world_import_must_come_from_configured_checkout(tmp_path):
    expected = tmp_path / "configured/track4world/nets/model.py"
    expected.parent.mkdir(parents=True)
    expected.write_text("# configured fixture\n")
    require_module_origin(SimpleNamespace(__file__=str(expected)), expected)
    with pytest.raises(RuntimeError, match="wrong checkout"):
        require_module_origin(
            SimpleNamespace(__file__=str(tmp_path / "stale/track4world/nets/model.py")),
            expected,
        )


def test_online_history_rejects_a_validity_hole_before_cuda():
    extractor = OnlineTrack4WorldExtractor.__new__(OnlineTrack4WorldExtractor)
    images = torch.zeros(1, 2, 3, 3, 2, 2)
    timestamps = torch.tensor([[-0.1, -0.05, 0.0]])
    valid = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="invalid prefix"):
        extractor(images, timestamps, valid)


def test_adapter_contract_allows_moved_paths_and_rejects_semantic_mismatch(tmp_path):
    first = tiny_policy()
    enable(first)
    first.geometry_config["extractor"].update(
        repo_path="/training/track4world",
        checkpoint_path="/training/track4world.pth",
        da3_path="/training/da3",
        device="cuda:7",
    )
    provenance = {"fixture_sha256": "abc123"}
    first.set_geometry_provenance(provenance)
    first.set_base_checkpoint_provenance("base-abc123")
    path = tmp_path / "adapter.pt"
    first.save_geometry_adapter(path)

    moved = tiny_policy()
    enable(moved)
    moved.geometry_config["extractor"].update(
        repo_path="/cluster/track4world",
        checkpoint_path="/cluster/track4world.pth",
        da3_path="/cluster/da3",
        device="cuda:0",
    )
    moved.set_geometry_provenance(provenance)
    moved.set_base_checkpoint_provenance("base-abc123")
    moved.load_geometry_adapter(path)

    incompatible = tiny_policy()
    enable(incompatible)
    incompatible.geometry_config["history_length"] = 4
    with pytest.raises(ValueError, match="configuration mismatch"):
        incompatible.load_geometry_adapter(path)

    wrong_producer = tiny_policy()
    enable(wrong_producer)
    wrong_producer.geometry_config["extractor"].update(
        repo_path="/cluster/track4world",
        checkpoint_path="/cluster/track4world.pth",
        da3_path="/cluster/da3",
    )
    wrong_producer.set_geometry_provenance({"fixture_sha256": "different"})
    wrong_producer.set_base_checkpoint_provenance("base-abc123")
    with pytest.raises(ValueError, match="training producer"):
        wrong_producer.load_geometry_adapter(path)

    wrong_base = tiny_policy()
    enable(wrong_base)
    wrong_base.geometry_config["extractor"].update(
        repo_path="/cluster/track4world",
        checkpoint_path="/cluster/track4world.pth",
        da3_path="/cluster/da3",
    )
    wrong_base.set_geometry_provenance(provenance)
    wrong_base.set_base_checkpoint_provenance("different-base")
    with pytest.raises(ValueError, match="base FastWAM checkpoint differs"):
        wrong_base.load_geometry_adapter(path)


def test_base_checkpoint_is_strict_except_for_new_geometry_keys(tmp_path):
    base = tiny_policy()
    good_path = tmp_path / "base.pt"
    base.save_checkpoint(good_path)

    geometry_model = tiny_policy()
    enable(geometry_model)
    geometry_model.load_checkpoint(good_path)

    payload = torch.load(good_path, map_location="cpu", weights_only=True)
    missing_key = next(iter(payload["mot"]))
    payload["mot"].pop(missing_key)
    bad_path = tmp_path / "bad.pt"
    torch.save(payload, bad_path)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        geometry_model.load_checkpoint(bad_path)


class _FiveIndices(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return index


def _sharded_indices(rank, resume_batch_offset=0):
    dataset = _FiveIndices()
    sampler = ResumableEpochSampler(dataset, seed=7, batch_size=1, num_processes=2)
    sampler.set_resume_batch_offset(resume_batch_offset)
    loader = prepare_data_loader(
        DataLoader(dataset, batch_size=1, sampler=sampler),
        num_processes=2,
        process_index=rank,
        put_on_device=False,
    )
    return [int(batch.item()) for batch in loader]


def test_distributed_resume_preserves_original_uneven_tail_padding():
    for rank in range(2):
        uninterrupted = _sharded_indices(rank)
        resumed = _sharded_indices(rank, resume_batch_offset=2)
        assert resumed == uninterrupted[2:]


def test_resume_contract_rejects_changed_training_horizon(tmp_path):
    trainer = Wan22Trainer.__new__(Wan22Trainer)
    trainer.model = tiny_policy()
    trainer.train_dataset = list(range(5))
    trainer.seed = 42
    trainer.batch_size = 1
    trainer.gradient_accumulation_steps = 1
    trainer.max_steps = 100
    trainer.mixed_precision = "bf16"
    trainer.learning_rate = 1e-4
    trainer.weight_decay = 1e-2
    trainer.max_grad_norm = 1.0
    trainer.accelerator = SimpleNamespace(num_processes=2)
    trainer.cfg = SimpleNamespace(lr_scheduler_type="cosine")
    payload = {"training_contract": trainer._training_resume_contract()}
    trainer.max_steps = 200
    with pytest.raises(ValueError, match="max_steps"):
        trainer._validate_training_resume_contract(payload, tmp_path / "trainer_state.json")


class _TrainToy(nn.Module):
    geometry_config = {"target": "vae_latent"}

    def __init__(self):
        super().__init__()
        self.dit = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.dit.weight, 0.25)
        self.frozen = nn.Parameter(torch.tensor(17.0), requires_grad=False)

    def configure_geometry_train_mode(self):
        self.eval().requires_grad_(False)
        self.dit.train().requires_grad_(True)

    def training_loss(self, sample):
        loss = (self.dit(sample["x"]) - sample["target"]).square().mean()
        return loss, {"loss_video": float(loss.detach())}

    def forward(self, sample):
        return self.training_loss(sample)

    def save_geometry_adapter(self, path, **metadata):
        torch.save({"adapter": self.dit.state_dict(), "metadata": metadata}, path)

    def load_geometry_adapter(self, path):
        self.dit.load_state_dict(torch.load(path, weights_only=True)["adapter"])


class _RankDataset:
    def __init__(self, rank):
        self.rank = rank

    def __len__(self):
        return 16

    def __getitem__(self, index):
        return {"x": torch.ones(1), "target": torch.tensor([2.0 * self.rank - 1.0])}


def _trainer_config(directory, **overrides):
    config = dict(
        output_dir=str(directory), learning_rate=0.01, weight_decay=0.0,
        batch_size=1, num_workers=0, num_epochs=1, max_steps=3,
        log_every=1, save_every=1, eval_every=0, eval_num_inference_steps=2,
        gradient_accumulation_steps=2, max_grad_norm=100.0, seed=42,
        resume=None, initial_checkpoint=None, mixed_precision="no",
        lr_scheduler_type="constant", wandb={"enabled": False},
    )
    config.update(overrides)
    return OmegaConf.create(config)


def _ddp_worker(rank, rendezvous, directory):
    os.environ.update(ACCELERATE_USE_CPU="true", RANK=str(rank), LOCAL_RANK=str(rank),
                      WORLD_SIZE="2", LOCAL_WORLD_SIZE="2", OMP_NUM_THREADS="1")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    # Some local environments have an optional, CUDA-only DeepSpeed install.
    # This test is specifically regular CPU DDP, not a DeepSpeed smoke.
    with patch("accelerate.utils.other.is_deepspeed_available", return_value=False):
        trainer = Wan22Trainer(_TrainToy(), _RankDataset(rank), cfg=_trainer_config(directory))
        trainer.train()  # calls the actual production loop, including saves
        assert len(trainer.accelerator._models) == 1
        assert trainer.model.module.frozen.item() == 17.0
        value = trainer.model.module.dit.weight.detach().clone()
        gathered = [torch.zeros_like(value) for _ in range(2)]
        dist.all_gather(gathered, value)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        state = Path(directory) / "checkpoints/state/step_000003"
        assert (state / "geometry_adapter.pt").is_file()
        assert not (state / "model.safetensors").exists()
        # Restore twice in the SAME Accelerator, then save again. Neither
        # hook is allowed to unregister its prepared model.
        for _ in range(2):
            with torch.no_grad():
                trainer.model.module.dit.weight.add_(10)
            trainer.load_training_state(str(state))
            torch.testing.assert_close(trainer.model.module.dit.weight, value, rtol=0, atol=0)
            assert len(trainer.accelerator._models) == 1
            assert trainer.global_step == 3
        trainer.max_steps = 4
        trainer.train()
        assert len(trainer.accelerator._models) == 1
        assert (Path(directory) / "checkpoints/state/step_000004/geometry_adapter.pt").is_file()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch distributed unavailable")
def test_real_trainer_ddp_sync_repeated_save_and_resume(tmp_path):
    rendezvous = (tmp_path / "gloo_init").as_uri()
    mp.spawn(_ddp_worker, args=(rendezvous, str(tmp_path / "training")), nprocs=2, join=True)
