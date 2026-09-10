# LIBERO 测试时在线提取 Track4World 特征

本文介绍如何加载“基础 FastWAM checkpoint + geometry adapter”，并在 LIBERO
simulator rollout 中在线提取 Track4World 特征。测试阶段不会读取离线缓存。

## 1. 在线测试数据流

```text
LIBERO observation
    ├── external RGB
    └── wrist RGB
          │
          ▼
LiberoHistoryBuffer [B,2,8,3,256,256]
          │
          ▼
frozen Track4World + DA3（每次 policy replan 调用一次）
    ├── scene：当前时刻
    ├── camera：当前时刻
    └── track：最近 8 帧历史
          │
          ▼
GeometryTokenizer + 当前 VAE latent 三路 cross-attention
          │
          ▼
FastWAM world model / ActionDiT → action chunk → simulator
```

`geometry_cache` 不会被打开。测试机器必须能够访问 Track4World、DA3 和对应
权重，因为特征来自 simulator 当前观测。

## 2. 路径与两种 checkpoint

继续使用 `configs/paths/libero_track4world_CLUSTER.yaml`，重点检查：

| 字段/参数 | 用途 |
| --- | --- |
| `ckpt` | 训练 adapter 时使用的基础 6B FastWAM checkpoint |
| `EVALUATION.geometry_adapter` | 离线训练产生的约 37 MB adapter |
| `paths.dataset_stats` | action/proprio 归一化统计 |
| `paths.model_base` | Wan VAE/T5 本地目录 |
| `paths.track4world_repo/checkpoint` | 在线 extractor 源码和权重 |
| `paths.da3_model` | 本地 DA3 权重 |
| `paths.libero_repo` | LIBERO、BDDL、init states 和 assets |
| `paths.output_root` | rollout 视频和结果 JSON 的输出根目录 |

两种权重不能混用：

```text
ckpt                         = immutable FastWAM checkpoint
EVALUATION.geometry_adapter  = trained geometry adapter
```

adapter 会校验基础 checkpoint SHA、几何结构、Track4World/DA3 权重和生产代码；
不匹配时直接报错。

## 3. 环境检查

```bash
conda activate fastwam
cd /cluster/repos/FastWAM
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml doctor
```

推荐使用官方环境：Python 3.10、PyTorch 2.7.1、torchvision 0.22.1、
TorchCodec 0.4.0。TorchCodec 不兼容时会回退 PyAV，但正式 benchmark 应先解决
版本匹配。

## 4. 两步在线 smoke

先跑一个任务、一次 trial、两个环境 action step：

```bash
python scripts/run_libero_geometry_eval.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  task=libero_geometry_offline_2cam224 \
  ckpt=/cluster/checkpoints/fastwam/libero_uncond_2cam224.pt \
  EVALUATION.geometry_adapter=/cluster/outputs/fastwam_track4world/offline/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=/cluster/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  EVALUATION.max_steps=2 \
  EVALUATION.num_inference_steps=2 \
  EVALUATION.output_dir=/cluster/outputs/fastwam_track4world/smoke_online
```

这里选择 `task=libero_geometry_offline_2cam224` 是为了加载与离线训练相同的
模型、processor 和 geometry 配置。评测入口不会实例化训练 dataset，因此
不会访问 HDF5。

专用 wrapper 默认添加 `EVALUATION.compile_action_infer=false`，这是当前真实
验证过的在线路径。只有在目标集群单独完成 compiled/eager parity 后才建议
显式启用编译。

## 5. 完整 LIBERO benchmark

完整评测保持 `EVALUATION.max_steps=null`，让代码使用对应 suite 的 episode
上限：

```bash
python scripts/run_libero_geometry_eval.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  task=libero_geometry_offline_2cam224 \
  ckpt=/cluster/checkpoints/fastwam/libero_uncond_2cam224.pt \
  EVALUATION.geometry_adapter=/cluster/outputs/fastwam_track4world/offline/checkpoints/weights/step_010000.pt \
  EVALUATION.dataset_stats_path=/cluster/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50 \
  EVALUATION.max_steps=null \
  EVALUATION.output_dir=/cluster/outputs/fastwam_track4world/eval_online
```

之后对 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 的全部
task_id 分发运行。报告结果时还应运行原始 FastWAM baseline，保持 checkpoint、
trial、seed、action horizon 和 inference steps 一致。

## 6. 训练/测试输入一致性

离线训练缓存与 simulator 在线测试遵循同一边界约定：

- 相机顺序为 external、wrist；
- 时间顺序为最早帧到当前帧；
- 当前帧必须是历史最后一帧；
- episode 开始处复制最早帧补齐，但 mask 标记为无效前缀；
- mask 必须是“无效前缀 + 连续有效后缀”，不允许中间空洞；
- history length 8、stride 1、FPS 20；
- Track4World 输入每相机 256×256，FastWAM VAE 输入每相机 224×224。

两个分支来自同一原始 RGB 和同一时间索引，只是目标分辨率不同。不要将
`512→256→224` 和 `512→224` 的像素差异误认为数据错位。

## 7. 确认在线提取发生

每次 policy replan 应看到：

```text
Online geometry replan: denoising_steps=... extractor_calls=1 ...
```

结果 JSON 应包含：

```json
{
  "online_geometry": {
    "features_precomputed": false,
    "episodes": [
      {
        "extractor_calls": 1,
        "last_quality": {
          "scene_valid": 0.95,
          "camera_valid": 1.0,
          "track_valid": 0.80
        }
      }
    ]
  }
}
```

quality 数值只是结构示例，不是固定阈值。关键是
`features_precomputed=false`，且 extractor 调用次数与 replan 次数一致。

如需直接比较在线 extractor 与离线缓存：

```bash
python scripts/libero_track4world.py \
  --paths configs/paths/libero_track4world_CLUSTER.yaml \
  parity --device cuda:0 --count 32
```

## 8. 性能与常见错误

本机 A100 smoke 中首次 replan（含模型初始化）约 29--31 秒；warm Track4World
提取约 0.5--2.3 秒。完整测试时间需要在集群实测。

- `Online Track4World producer differs`：在线源码/权重与训练缓存的 producer 不同。
- `base FastWAM checkpoint differs`：adapter 与 `ckpt` 不是同一个基础模型。
- `Online geometry requires history...`：底层调用未传完整的三种 history tensor。
- 单卡 OOM：两卡时可让 FastWAM 用 GPU 0、Track4World 用 GPU 1；单卡可设置
  `paths.geometry_device=same`，但需要确认显存。
- 两步 smoke 未成功：这只验证闭环；正式结果必须使用完整 episode 上限。
