# 使用离线 Track4World 特征训练 FastWAM

本文介绍如何使用完整的 HDF5 几何缓存训练 FastWAM。前后步骤见：

- [离线提取 Track4World 特征](libero_geometry_01_offline_extraction_zh.md)
- [测试时在线提取特征](libero_geometry_03_online_evaluation_zh.md)

## 1. 训练数据流

训练监督仍来自官方 FastWAM 使用的四套 LeRobot 数据，视频、action、proprio
和文本 embedding 没有换数据源。新增 dataset 根据同一个 `sample_index` 从
`geometry_cache` 读取对应的 Track4World 原始特征：

```text
LeRobot sample_index
    ├── FastWAM video/action/proprio/context
    └── HDF5 scene/camera/track raw features
                    │
                    ▼
            trainable GeometryTokenizer
                    │
当前 VAE latent ─── 三路 residual cross-attention
                    │
                    ▼
          frozen FastWAM world/action backbone
                    │
              video + action loss
```

只优化 `mot.geometry_tokenizer` 和 `mot.geometry_latent_adapter`，共
9,240,355 个参数。约 6B 的 FastWAM 主干、VAE 和 Track4World 都被冻结。
三路 residual gate 从零初始化，因此初始行为与原始 FastWAM 一致。

## 2. 训练前确认路径

训练读取 `configs/paths/libero_track4world_CLUSTER.yaml`。执行命令前逐项确认：

| 字段 | 用途 |
| --- | --- |
| `train_datasets` | 四套 LeRobot 数据，顺序与提取时完全相同 |
| `text_embedding_cache` | FastWAM 的 T5 文本 embedding |
| `geometry_cache` | 完整的 v2 HDF5 缓存 |
| `fastwam_checkpoint` | 不可变的初始 FastWAM checkpoint |
| `dataset_stats` | 与基础 checkpoint 配套的归一化统计 |
| `model_base` | Wan VAE/T5 本地目录 |
| `track4world_repo/checkpoint` | 在线 validation 使用的 Track4World |
| `da3_model` | 在线 validation 使用的本地 DA3 |
| `output_root` | adapter、状态、日志和验证视频输出根目录 |

离线训练 batch 不运行 Track4World，但周期性 validation 故意在线提取，以持续
检查测试阶段的真实路径，所以 Track4World 和 DA3 仍必须正确配置。

## 3. 训练前验收

```bash
conda activate fastwam
cd /cluster/repos/FastWAM
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  doctor --require-cache

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  coverage --require-complete

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  parity --device cuda:0 --count 32
```

正式训练不要添加 `data.train.geometry_require_complete_cache=false`。这个参数只
用于少量样本的开发 smoke；生产配置默认拒绝不完整缓存。

## 4. 单卡离线训练

```bash
python scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  max_steps=10000 \
  batch_size=1
```

两卡机器可以让 FastWAM 使用 GPU 0、在线 validation 的 Track4World 使用 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  paths.geometry_device=cuda:1 \
  max_steps=10000 \
  batch_size=1
```

## 5. 16 卡 DDP 训练

```bash
accelerate launch --multi_gpu --num_processes 16 \
  scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  paths.geometry_device=same \
  max_steps=10000 \
  batch_size=1
```

多卡训练必须使用 `paths.geometry_device=same`，否则在线 validation 的所有 rank
可能争用同一个字面设备。`batch_size` 是每 rank batch；上述 global batch 为
16。若再设置 `gradient_accumulation_steps=2`，有效 global batch 为 32。

默认 task 配置在
[`libero_geometry_offline_2cam224.yaml`](../configs/task/libero_geometry_offline_2cam224.yaml)，
主要参数为：

```yaml
learning_rate: 1.0e-4
weight_decay: 1.0e-2
save_every: 2000
eval_every: 200
eval_num_inference_steps: 10
gradient_accumulation_steps: 1
```

## 6. 输出与 checkpoint

假设 `output_root=/cluster/outputs/fastwam_track4world`，默认输出为：

```text
/cluster/outputs/fastwam_track4world/offline/
├── config.yaml
├── dataset_stats.json
├── eval/step_XXXXXX_rank_XXX.mp4
└── checkpoints/
    ├── weights/step_XXXXXX.pt
    └── state/step_XXXXXX/
        ├── geometry_adapter.pt
        ├── optimizer.bin
        ├── scheduler.bin
        ├── random_states_*.pkl
        ├── sampler.bin          # 是否生成取决于 Accelerate 版本
        └── trainer_state.json
```

`weights/step_XXXXXX.pt` 是测试使用的 adapter-only 权重，约 37 MB，包含 79
个几何 tensor、几何语义配置、Track4World/DA3/生产源码指纹以及基础 FastWAM
checkpoint SHA。不会重复保存 6B 主干。

`state/step_XXXXXX/` 额外包含 optimizer、scheduler、RNG 和 dataloader 位置，
用于继续训练。

## 7. 恢复训练

```bash
accelerate launch --multi_gpu --num_processes 16 \
  scripts/run_libero_geometry_train.py \
  --mode offline \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  paths.geometry_device=same \
  resume=/cluster/outputs/fastwam_track4world/offline/checkpoints/state/step_006000 \
  max_steps=10000 \
  batch_size=1
```

`max_steps` 是完整 cosine schedule 的最终长度，不是本次增加的步数。若目标是
20,000 步，首次启动和每次恢复都必须使用 `max_steps=20000`。

严格恢复要求以下内容保持一致：数据长度和顺序、world size、每 rank batch、
gradient accumulation、seed、precision、learning rate、weight decay、scheduler、
gradient clipping、最终 `max_steps` 和基础 FastWAM checkpoint。修改后代码会在
载入 Accelerate 状态之前报出具体差异。

## 8. 正常运行的判断标准

日志中应出现：

```text
Bound geometry adapter to offline cache producer
Loading immutable initial checkpoint
Setting geometry adapters to train mode and freezing FastWAM
loss_video=...
loss_action=...
```

到达 `eval_every` 后，还应看到 Track4World 从本地加载，以及有限的 `val_loss`、
`action_l1`、`action_l2`。单 batch diffusion loss 会随样本、timestep 和噪声
波动，不要求逐步单调下降。正式结论应比较 train/validation 曲线，并对有无
geometry 进行 LIBERO success-rate ablation。

## 9. 常见错误

- `Offline geometry cache is incomplete`：继续全量提取，不要关闭保护。
- `Geometry cache contract mismatch`：训练的数据、权重、源码或配置与提取时不同。
- `Training resume contract mismatch`：恢复时改变了总步数、卡数、batch 或 scheduler。
- `base FastWAM checkpoint differs`：adapter 与当前基础 checkpoint 不匹配。
- 多卡 validation OOM：确认 `paths.geometry_device=same`，并适当降低验证频率。
