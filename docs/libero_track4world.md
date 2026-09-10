# FastWAM + Track4World on LIBERO

Step-by-step Chinese runbooks:

1. [离线提取 Track4World 特征](libero_geometry_01_offline_extraction_zh.md)
2. [使用离线特征训练 FastWAM](libero_geometry_02_offline_training_zh.md)
3. [LIBERO 测试时在线提取特征](libero_geometry_03_online_evaluation_zh.md)

This branch is the official FastWAM model plus one geometry residual at the
current VAE latent. Track4World remains frozen. Scene, camera and recent track
history are tokenized into three banks and independently cross-attended by the
current two-camera VAE latent. Training can read frozen raw geometry from disk;
LIBERO evaluation always extracts it online from simulator RGB history.

## 1. Paths to edit on a new machine

Copy `configs/paths/libero_track4world_local.yaml` to a new name inside the
same directory and edit that file only. The current server resolves to:

| Parameter | Current path |
| --- | --- |
| Four LeRobot 2.1 training roots | `/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2/{libero_spatial,libero_object,libero_goal,libero_10}_no_noops_lerobot` |
| T5 text cache | `/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero` |
| Initial FastWAM checkpoint | `/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224.pt` |
| Dataset statistics | `/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json` |
| Wan model base | `/mnt/homes/zhaoshizhen/checkpoints/fastwam/model_base` |
| Track4World repository | `/mnt/homes/zhaoshizhen/lf/Track4World` |
| Track4World checkpoint | `/mnt/homes/zhaoshizhen/lf/Track4World/checkpoints/track4world_da3.pth` |
| DA3 checkpoint directory | `/mnt/homes/zhaoshizhen/lf/Track4World/checkpoints/DA3NESTED-GIANT-LARGE-1.1` |
| Additional Track4World Python packages | `/mnt/homes/zhaoshizhen/lf/.deps/fastwam_track4world` |
| LIBERO repository | `/home/zhaoshizhen/lf/third_party/libero_base/repos/LIBERO` |
| Offline geometry cache | `/home/zhaoshizhen/lf/repos_new/FastWAM/feature_cache/libero_track4world_v2` |
| Training/evaluation output root | `/home/zhaoshizhen/lf/repos_new/FastWAM/outputs/libero_track4world` |
| Hugging Face endpoint | `https://hf-mirror.com` |

`geometry_device` is `cuda:1` on this two-GPU server: FastWAM uses GPU 0 and
online Track4World uses GPU 1. For multi-GPU online training, set it to `same`
so each rank runs Track4World on that rank's FastWAM device. Offline training
does not instantiate Track4World until periodic online validation.

The Wan files expected below `model_base` are:

```text
DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

Run the doctor before extraction or training. Every wrapper prints the same
resolved path table before it changes state.

```bash
conda activate fastwam
cd /path/to/FastWAM
python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml doctor
```

Use the official FastWAM environment (Python 3.10, PyTorch 2.7.1, torchvision
0.22.1 and TorchCodec 0.4.0) and install FastWAM with `pip install -e .`.
Track4World/DA3 dependencies and upstream submodules must also be installed or
made available through `track4world_extra_pythonpath`. DA3 is required by this
configuration. FastWAM now redirects the upstream DA3 factory to `da3_model`
with `local_files_only=True` during construction; no local modification of
Track4World's `model.py` or `TRACK4WORLD_DA3_PATH` support is required. The
local smoke environment uses PyTorch 2.5.1, so video decoding warns and falls
back to PyAV; the recommended versions avoid that fallback.

If a checkpoint must be downloaded on the cluster, use the mirror without an
HTTP proxy, for example:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  TencentARC/Track4World track4world_da3.pth \
  --local-dir /path/to/Track4World/checkpoints
```

## 2. Extract all LIBERO geometry

One process can extract everything, and interrupted runs safely skip committed
frames:

```bash
python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  extract --device cuda:0
```

For 16 GPUs, use 16 contiguous shards. Each process writes to the same cache;
per-episode file locks and commit bits make the writes restart-safe.

```bash
mkdir -p logs/geometry_extract
for shard in $(seq 0 15); do
  CUDA_VISIBLE_DEVICES=${shard} python scripts/libero_track4world.py \
    --paths configs/paths/libero_track4world_CLUSTER.yaml \
    extract --num-shards 16 --shard-id ${shard} --device cuda:0 \
    > logs/geometry_extract/shard_${shard}.log 2>&1 &
done
wait

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  coverage --require-complete
```

After extraction, re-run Track4World online on representative cached windows.
Boolean validity masks must match exactly; floating tensors use relative RMSE
and cosine thresholds that tolerate only CUDA FP16 kernel noise:

```bash
python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  parity --device cuda:0 --count 32
```

The current dataset contains 277,713 windows. Measured hot extraction is about
1.4--3.0 seconds/window on A100, so a rough 16-GPU estimate is 7--15 hours plus
startup and I/O. Raw float32 tensors occupy about 0.416 TiB before HDF5 LZF
compression; reserve roughly 0.35--0.45 TiB. Measure several thousand windows
on the target filesystem before scheduling the complete run.

Cache schema v2 binds dataset metadata, parquet and video **contents**, camera
order, frame rate, history length, resize rule, extractor settings, Track4World
source, extraction/preprocessing source, and both Track4World and DA3 weights
using SHA-256. Individual windows also carry identity and tensor checksums.
Device IDs and absolute machine paths do not participate in these identities.

You can copy the data, models, source and v2 cache to different directories on
another machine and update the paths YAML without re-extracting. Keep dataset
order and file contents unchanged. Editing producer code or replacing weights
at the same path correctly rejects the old cache. Each cache opening performs
a full sequential content read before training/extraction (including video
files); budget startup I/O, especially with 16 ranks. Hashes are not memoized
using timestamps because some filesystems coalesce them.

**Legacy v1 caches are deliberately rejected, not silently converted.** They
have no producer-weight fingerprints, so their provenance cannot be recovered
from the manifest. Preserve the old directory and re-extract into the new v2
directory. The local path config now selects v2 and leaves v1 untouched.

## 3. Train from offline features

The wrapper prints and validates all paths before Hydra starts. The dataset
also checks that all 277,713 cache entries exist. On one GPU:

```bash
python scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  max_steps=10000 batch_size=1
```

On 16 GPUs using regular distributed data parallelism:

```bash
accelerate launch --multi_gpu --num_processes 16 \
  scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  max_steps=10000 batch_size=1 paths.geometry_device=same
```

Only `geometry_tokenizer` and `geometry_latent_adapter` are optimized (9.24M
parameters). The 6B FastWAM backbone is loaded from `initial_checkpoint` and
frozen. Checkpoints are adapter-only: about 37 MB for inference weights and
about 104 MB for resumable adapter/optimizer/scheduler/RNG state.

Every production adapter records a path-independent geometry contract, exact
Track4World/DA3 weights and producer-source fingerprints, and the SHA-256 of
the immutable FastWAM base checkpoint. Evaluation refuses a semantically
different geometry configuration, producer, or base checkpoint. Therefore,
enable geometry from the destination machine's path config first; checkpoint
payloads are never allowed to reactivate stale absolute paths.

Use `paths.geometry_device=same` on every multi-GPU training launch, including
offline training: periodic validation extracts geometry online on every rank.
Do not map all ranks' validation extractors to a single literal `cuda:1`.
Training enters the prepared model through `forward`, so regular DDP can
synchronize gradients. The public validation `infer` entry also forwards the
causal history, and adapter training modes are restored after validation.

Resume from a state directory; the trainer first reloads the immutable FastWAM
checkpoint and then restores the adapter state and dataloader cursor:

```bash
python scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  resume=/path/to/output/checkpoints/state/step_006000 \
  max_steps=10000
```

`max_steps` is the final schedule length, not an increment. Set the intended
total on the first launch and keep it unchanged on every resume. For example,
if the intended run is 20,000 steps, use `max_steps=20000` both initially and
when resuming. The strict resume contract rejects changes to this value, world
size, per-rank batch size, accumulation, seed, optimizer schedule, precision,
dataset length, or base checkpoint; changing them would invalidate exact
dataloader and cosine-scheduler continuity.

## 4. Optional online training

Online training consumes the same LeRobot supervision but decodes the causal
RGB history and runs frozen Track4World in each forward pass. It is intended
for parity/debug experiments; offline training is much faster.

```bash
python scripts/run_libero_geometry_train.py \
  --mode online \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  max_steps=100 batch_size=1
```

For distributed online training set `geometry_device: same` in the selected
path YAML. Do not leave all ranks pointing at one literal `cuda:1`.

## 5. Online LIBERO evaluation

Evaluation never reads the offline geometry cache. The wrapper generates the
LIBERO runtime path file from the same path YAML, maintains one causal history
buffer per environment, and calls Track4World exactly once per policy replan.

```bash
python scripts/run_libero_geometry_eval.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  task=libero_geometry_offline_2cam224 \
  ckpt=/path/to/libero_uncond_2cam224.pt \
  EVALUATION.geometry_adapter=/path/to/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=/path/to/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50
```

Keep `ckpt` pointed to the immutable base FastWAM checkpoint and
`EVALUATION.geometry_adapter` pointed to the trained adapter. The simulator
provides two uint8 RGB images; after LIBERO orientation correction they become
`history_images [B,2,8,3,256,256]` in external-then-wrist order, with causal
timestamps ending at zero and a valid mask. This is the same tensor boundary
used to produce the offline training cache.

The geometry evaluation wrapper defaults `EVALUATION.compile_action_infer` to
`false`, matching the validated online path. It can be explicitly enabled only
after a separate compiled-inference parity check on the target environment.

`EVALUATION.max_steps` should remain `null` for reported benchmark results.
Setting it to a small integer only performs an integration smoke test.

## 6. Regression checks after migration

```bash
PYTHONPATH=src:tests python -m pytest -q \
  tests/test_vae_geometry.py \
  tests/test_geometry_data_contract.py \
  tests/test_geometry_regressions.py
```

The tests cover zero-gate baseline equivalence, causal/current-latent fusion,
three-bank gradients, the public online inference and trainer validation
paths, validation mode restoration, local-only DA3 loading and module-origin
checks, cache relocation/content changes, adapter/base/producer contracts,
history validity, uneven distributed sampler tails, and a CPU two-rank run of
the actual trainer with gradient accumulation and checkpoint restore. They use
tiny models and fixtures and do not establish full LIBERO success rates or
training convergence. Run real-data extraction, train/validation and simulator
smoke checks on the target machine before a full experiment. Never disable
complete cache validation for a reported full-dataset training run.

Local revalidation on 2026-09-10 (A100, existing PyTorch 2.5.1 environment):

- 31 regression tests passed, including the CPU two-rank trainer test.
- Six real LeRobot windows were extracted into v2 with HF networking disabled;
  the old v1 cache was preserved. This is not a complete dataset cache.
- Online re-extraction of cached samples passed exact-mask and numerical parity;
  the observed worst track relative RMSE was `3.67e-4` and minimum cosine was
  `0.99999994`.
- The full 6B model completed an offline update plus online validation with
  finite losses. Its final adapter contains 79 tensors (about 37 MB), base and
  producer identities, while resumable state contains the strict training
  contract. No full-model checkpoint was written. The smoke explicitly disabled
  the complete-cache guard; production configs still require it.
- Reloading that state restored adapter, optimizer, scheduler, RNG and
  dataloader offset under the same strict contract. A real fixed-sample,
  fixed-noise optimization check reduced loss from `0.0980076` to `0.0963697`
  in six adapter updates (about 1.67%); this proves the gradient/optimizer path,
  not dataset-level convergence.
- A real LIBERO simulator smoke loaded that final adapter, consumed causal
  external+wrist history, made one online Track4World extraction for one policy
  replan, and returned finite actions over two truncated environment steps. The
  episode did not succeed and is not a benchmark result.

Local final artifacts are under
`outputs/libero_track4world/audit_final_bound/` and
`outputs/libero_track4world/audit_final_sim/`. These checks establish
integration behavior, not sustained loss reduction, 16-GPU validation,
compiled inference compatibility, or full-suite success rates.
