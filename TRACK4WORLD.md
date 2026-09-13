# FastWAM_3_track：LIBERO Plus 基线 + Track4World 几何条件

基线是同级 FastWAM_3 的 `31770396e672bfa283f4283e59775057ab69ec89`。
几何实现迁自 FastWAM_2；不修改 FastWAM_3 或 FastWAM_2。
现在支持 **uncond 和 joint 两种几何版本**，默认入口仍为 uncond。
joint 的完整命令见 [JOINT_TRACK4WORLD.md](JOINT_TRACK4WORLD.md)。原 IDM 保留，但未添加几何层。

## 数据和模型

- 训练：同一套 LIBERO LeRobot 数据，按实际 `sample_index` 读取逐 episode HDF5 缓存。
- 离线保存的是冻结 Track4World 的 raw features，不是可训练 tokenizer 的输出。
- 当前 scene、当前 camera 和最近 8 帧 track，通过一次三路残差 cross-attention
  注入当前帧 VAE latent。未来 latent target、原损失和 scheduler 不变。
- VAE/T5/Track4World 冻结；原 WAM 主干、proprio、几何 tokenizer/adapter 参与训练。
- 训练期 validation 使用离线缓存；LIBERO/Plus 仿真每个 replan 在线提取一次。
- RGB 历史为外部相机、腕部相机顺序，每环境步记录；只含当前及过去帧。
- Plus 的 seed 递增和清理规则保留。几何开启时不跳过渲染，避免非 replan 帧变成黑图；
  几何关闭时完全保留原 render gate。比较成功率时保留相同 seed/replan/去噪设置。

## 资产路径与环境

```bash
cd /home/zhaoshizhen/lf/repos_new_1/FastWAM_3_track
export FASTWAM_ENV=/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam
export CUDA_HOME="$FASTWAM_ENV"
export GEOMETRY_PATHS="$PWD/configs/paths/libero_track4world_local.yaml"
source scripts/geometry_paths.sh
```

在集群复制 `configs/paths/libero_track4world_local.yaml` 为同目录下的新 YAML，修改路径并
设置 `GEOMETRY_PATHS` 即可。不要把路径配置放在 `configs/paths` 之外。

| 参数 | 用途 |
| --- | --- |
| `train_datasets` | 四个 LeRobot 根目录，包含 meta/data/videos |
| `text_embedding_cache`、`dataset_stats` | 训练 T5 缓存、动作/proprio 归一化统计 |
| `model_base`、`model_id`、`tokenizer_model_id` | Wan/VAE/T5/tokenizer 位置；model_id 可绝对路径 |
| `redirect_common_files` | 是否使用 DiffSynth 共享 safetensors 布局 |
| `action_dit_checkpoint` | ActionDiT 初始化 checkpoint |
| `track4world_repo`、`track4world_checkpoint` | 已配置的官方 Track4World 源码和权重 |
| `da3_model` | DA3 本地目录，含 config.json、model.safetensors |
| `track4world_extra_pythonpath` | DA3/Pi3/utils3d 等额外依赖目录 |
| `libero_repo` | 普通 LIBERO 环境代码和资产 |
| `libero_plus_repo`、`libero_plus_assets` | Plus 专用源码/资产；默认 null，需明确填写 |
| `geometry_cache`、`output_root` | 离线特征和训练/评测输出位置 |

`fastwam_checkpoint` 是兼容旧工具的展示字段，不作为训练初始化权重。
训练仍从 Wan + ActionDiT 初始化；测试必须指定本次训练的 policy `.pt`。
代码没有打包训练数据或模型权重，没有自动安装依赖或下载。
本机已有环境是 torch2.5.1/CUDA12.1；新集群需验证其 torch/CUDA/解码依赖。
保持主干原环境约束，补充 h5py 和已配置的 Track4World/DA3/Pi3/utils3d 依赖。

## 1. 离线提取

```bash
# 先抽两个样本，直接 Python 入口默认 cuda:0
CUDA_VISIBLE_DEVICES=0 python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" extract --indices 0,7
CUDA_VISIBLE_DEVICES=0 python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" parity --indices 0,7

# 全量单卡，或全部16个分片分别启动：SHARD_ID=0..15（不要只启动一个）
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_geometry.sh
# CUDA_VISIBLE_DEVICES=5 NUM_SHARDS=16 SHARD_ID=5 bash scripts/extract_geometry.sh

python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" coverage --require-complete
```

完成完整覆盖后才可正式训练。文件内容/producer 指纹防止错用旧缓存；路径移动不改变身份。
FastWAM_2 的同配置 raw 缓存可通过指纹验证后复用，本版本默认输出到独立新目录。
不要把仅抽两个样本的 smoke 当成全量提取完成。

## 2. 16卡离线特征训练

```bash
# 对照组
GEOMETRY_ENABLED=false OUTPUT_DIR=/cluster/runs/control_seed42 \
  bash scripts/train_ablation_16gpu.sh
# 几何组，必须已完整提取
GEOMETRY_ENABLED=true OUTPUT_DIR=/cluster/runs/track_seed42 \
  bash scripts/train_ablation_16gpu.sh
```

默认 B4/GAS2、16卡 global batch=128、10 epoch、seed42、bf16、ZeRO-1。
上述为 uncond；joint 使用 `train_joint_ablation_16gpu.sh`，默认 B8/GAS1、global batch 同为128。
allocator 和 FastWAM_3 一样默认 `expandable_segments:False`，允许环境变量覆盖。
两机各8卡使用 `NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0/1 MASTER_ADDR=主节点IP`，
同时设置一致的 `RUN_ID` 和共享 `OUTPUT_DIR`。其他参数两组保持一致。

完整续训：`RESUME_STATE=/.../checkpoints/state/step_NNNNNN`。
几何组新增 `geometry_state.json` pre-save/pre-load hook，校验模型几何配置、producer 和数据合同，
不改 optimizer/sampler/训练循环。旧 FastWAM_2 state 没有此文件会拒绝自动恢复；
不要伪造 sidecar。相容 `.pt` 可以作为 weights-only 加载，但不等价于完整续训。

## 3. 在线提取测试

```bash
GEOMETRY_ENABLED=true CKPT=/cluster/runs/track_seed42/checkpoints/weights/step_002000.pt \
  CUDA_VISIBLE_DEVICES=0 NUM_TRIALS=1 bash scripts/eval_ablation.sh \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0
```

迁移 policy 必须一起复制 `.pt`、相邻 `.pt.geometry.json` 和训练 `dataset_stats.json`。
stats 不在 checkpoint 上级目录时显式传 `EVALUATION.dataset_stats_path=/path/stats.json`。
普通 LIBERO 全任务可用 `EVAL_MODE=manager NUM_EVAL_GPUS=16`，仍走原 manager。

Plus 在填写专用路径后运行：

```bash
GEOMETRY_ENABLED=true CKPT=/cluster/runs/track_seed42/checkpoints/weights/step_002000.pt \
  NUM_GPUS=1 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 \
  bash scripts/eval_geometry_plus.sh
# 先检查配置（无需 Plus 资产，不运行评测）
# DRY_RUN=1 CKPT=/path/policy.pt bash scripts/eval_geometry_plus.sh
```

Plus wrapper 调用 FastWAM_3 的原 Plus launcher/manager/worker；默认每卡一个 worker，
避免每卡多个冻结几何模型挤占显存。全量前测显存再设 NUM_GPUS=16；任务清单仍为10030项。
`LIBERO_PLUS_ROOT` / `LIBERO_PLUS_ASSETS_DIR` 环境变量也可覆盖 Plus 路径；缺失时明确报错，
不会退回普通 LIBERO。此包装器默认关闭远端结果自动备份，避免隐式外部写入。

## 改动范围与本地检查

仅四个已有文件有几何接线：dataset 的实际索引、trainer 的验证特征传递、共享 LIBERO
episode 历史与渲染处理、Plus factory 许可。其他基线文件字节不变，包括原 runtime、训练
循环、模型核心、原训练脚本和历史验证报告。新增 GeometryTrainer 仅注册几何 state hooks。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider \
  geometry_tests experiments/libero_plus/test -q -rs
python scripts/check_ablation_gpu.py --skip-online --output outputs/check_small_train
python scripts/check_ablation_sim.py \
  --checkpoint outputs/check_small_train/checkpoints/weights/step_000004.pt \
  --output outputs/check_small_sim
```

GPU检查是小尺寸 WAM/VAE + 真实数据/几何，不是完整5B验证。Plus 单测使用环境替身验证
历史/seed/worker 配置，不代表真实 Plus 成功率。`experiments/libero_plus/validation.json`
保留的是 FastWAM_3 历史记录，不能作为本次几何版本的验收或全仓库 hash。
