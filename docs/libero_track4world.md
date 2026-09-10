# FastWAM + Track4World on LIBERO

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
| Offline geometry cache | `/home/zhaoshizhen/lf/repos_new/FastWAM/feature_cache/libero_track4world_v1` |
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
Track4World/DA3 dependencies must also be installed or made available through
`track4world_extra_pythonpath`. DA3 is required by this configuration. The
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

The current dataset contains 277,713 windows. Measured hot extraction is about
1.4--3.0 seconds/window on A100, so a rough 16-GPU estimate is 7--15 hours plus
startup and I/O. Raw float32 tensors occupy about 0.416 TiB before HDF5 LZF
compression; reserve roughly 0.35--0.45 TiB. Measure several thousand windows
on the target filesystem before scheduling the complete run.

The cache manifest binds dataset metadata, camera order, frame rate, history
length, resize rule and extractor settings. Training refuses a mismatched or
incomplete cache. Individual windows also carry identity and tensor SHA-256
checksums.

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
  max_steps=10000 batch_size=1
```

Only `geometry_tokenizer` and `geometry_latent_adapter` are optimized (9.24M
parameters). The 6B FastWAM backbone is loaded from `initial_checkpoint` and
frozen. Checkpoints are adapter-only: about 37 MB for inference weights and
about 104 MB for resumable adapter/optimizer/scheduler/RNG state.

Resume from a state directory; the trainer first reloads the immutable FastWAM
checkpoint and then restores the adapter state and dataloader cursor:

```bash
python scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  resume=/path/to/output/checkpoints/state/step_010000 \
  max_steps=20000
```

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

`EVALUATION.max_steps` should remain `null` for reported benchmark results.
Setting it to a small integer only performs an integration smoke test.
