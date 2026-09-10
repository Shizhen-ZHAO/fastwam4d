# LIBERO 离线 Track4World 特征提取

本文只介绍如何从四套 LeRobot LIBERO 数据生成离线几何缓存。后续步骤见：

- [使用离线特征训练](libero_geometry_02_offline_training_zh.md)
- [测试时在线提取特征](libero_geometry_03_online_evaluation_zh.md)

## 1. 输入与输出

`train_datasets` 必须按以下顺序配置，不能在提取后改变：

1. `libero_spatial_no_noops_lerobot`
2. `libero_object_no_noops_lerobot`
3. `libero_goal_no_noops_lerobot`
4. `libero_10_no_noops_lerobot`

每个 LeRobot 根目录至少应包含 `meta/info.json`、`meta/episodes.jsonl`、
`data/` 和 `videos/`。

每个样本读取最近 8 个时刻的 external 和 wrist RGB：

```text
history_images      [2,8,3,256,256]
history_timestamps  [8]
history_valid       [8]
```

默认网格为 `8 x 8 = 64` 个点，HDF5 中保存以下 Track4World 原始输出：

| 名称 | 单样本形状 |
| --- | --- |
| `scene / scene_aux / scene_valid` | `[2,64,1024] / [2,64,6] / [2,64]` |
| `camera / camera_aux / camera_valid` | `[2,3072] / [2,13] / [2]` |
| `track / track_aux / track_valid` | `[2,8,64,256] / [2,8,64,11] / [2,8,64]` |

浮点张量以 `float32` 保存，mask 以 `bool` 保存。缓存采用 per-episode HDF5、
LZF 压缩、文件锁和逐 window commit。保存的是冻结 Track4World 与可学习
GeometryTokenizer 之间的原始边界，不是训练后的 token。

## 2. 配置集群路径

```bash
cp configs/paths/libero_track4world_local.yaml \
   configs/paths/libero_track4world_CLUSTER.yaml
```

编辑新文件：

```yaml
train_datasets:
  - /cluster/data/libero_spatial_no_noops_lerobot
  - /cluster/data/libero_object_no_noops_lerobot
  - /cluster/data/libero_goal_no_noops_lerobot
  - /cluster/data/libero_10_no_noops_lerobot

text_embedding_cache: /cluster/data/text_embeds_cache/libero
fastwam_checkpoint: /cluster/checkpoints/fastwam/libero_uncond_2cam224.pt
dataset_stats: /cluster/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json
model_base: /cluster/checkpoints/fastwam/model_base

track4world_repo: /cluster/repos/Track4World
track4world_checkpoint: /cluster/checkpoints/track4world/track4world_da3.pth
da3_model: /cluster/checkpoints/track4world/DA3NESTED-GIANT-LARGE-1.1
track4world_extra_pythonpath: /cluster/deps/fastwam_track4world

libero_repo: /cluster/repos/LIBERO
geometry_cache: /cluster/cache/libero_track4world_v2
output_root: /cluster/outputs/fastwam_track4world
geometry_device: same
hf_endpoint: https://hf-mirror.com
```

`geometry_cache` 是本文的特征保存路径，建议位于共享 POSIX 文件系统。当前统一
preflight 会检查 YAML 中的所有路径，即使部分 FastWAM 路径不直接用于提取。

改变数据内容、数据顺序、Track4World/DA3 权重、生产代码、历史长度、FPS 或
网格后必须使用新缓存目录，不能复用旧 manifest。v1 缓存也必须重新提取为 v2。

## 3. checkpoint 与离线模式

提取脚本本身不会下载文件。应提前准备 Track4World 源码、Pi3/DA3 子模块、
`track4world_da3.pth`、DA3 的 `config.json` 和 `model.safetensors`，以及
`utils3d` 等依赖。

如需从 HF 镜像下载 Track4World checkpoint，应在提取任务前执行：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  TencentARC/Track4World track4world_da3.pth \
  --local-dir /cluster/checkpoints/track4world
```

正式提取时可强制离线：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 4. 提取前检查与小规模 smoke

```bash
conda activate fastwam
cd /cluster/repos/FastWAM
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml doctor

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  extract --device cuda:0 --start 0 --count 32 --log-every 1
```

`doctor` 应打印实际导入的 Track4World、`utils3d` 路径、四套数据的 frame 数和
20 FPS。当前完整数据预期共 277,713 个 window。

## 5. 单卡与 16 卡全量提取

单卡：

```bash
python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  extract --device cuda:0
```

16 卡：

```bash
mkdir -p logs/geometry_extract
for shard in $(seq 0 15); do
  CUDA_VISIBLE_DEVICES=${shard} python scripts/libero_track4world.py \
    --paths configs/paths/libero_track4world_CLUSTER.yaml \
    extract --num-shards 16 --shard-id ${shard} --device cuda:0 \
    --log-every 100 > logs/geometry_extract/shard_${shard}.log 2>&1 &
done
wait
```

每个进程只看到一张卡，因此进程内统一写 `cuda:0`。所有 shard 可以安全写入
同一 `geometry_cache`。中断后重复相同命令即可续抽，已 commit 的 window 会
自动跳过；不要手工删除 HDF5 或 lock 文件。

本机 A100 hot extraction 约 1.4--3.0 秒/window。16 卡全量粗略估计 7--15
小时，实际取决于共享存储和视频解码。原始 float32 数据约 0.416 TiB，建议
预留 0.35--0.45 TiB。

## 6. 完整性与 parity 验收

```bash
python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  coverage --require-complete

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  verify --count 1000

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  parity --device cuda:0 --count 32
```

完整缓存应报告 `expected_frames=written_frames=277713`、`missing_frames=0`。
`parity` 会在线重提取同一批 RGB：所有 bool mask 必须完全一致，每路最大
relative RMSE 不超过 `5e-3`，最小 cosine 不低于 `0.999`。

## 7. 常见错误

- `Path preflight failed`：检查 YAML 中所有文件和目录。
- `module was imported from ...`：清理错误的 `PYTHONPATH` 并重启进程。
- `Geometry cache contract mismatch`：使用新目录按当前数据、权重和代码重提取。
- `Legacy geometry cache`：旧缓存没有可靠 provenance，不能自动迁移。
- `coverage` 不完整：重启对应 shard；正式训练不要关闭完整性保护。
