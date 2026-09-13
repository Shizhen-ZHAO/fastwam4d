# FastWAM joint + Track4World

交给集群 Agent 的完整执行说明见 [JOINT_CLUSTER_AGENT_PROMPT.md](JOINT_CLUSTER_AGENT_PROMPT.md)，
包含源码版本检查、全量缓存校验、集群 smoke 和在线测试验收。

这是实际 `FastWAMJoint` 的几何子类，不是把 uncond 的配置改名。
原 joint 的 attention mask、视频/动作联合去噪、loss、scheduler、训练循环均未修改。
三路几何仍只增强当前帧 VAE latent；joint 的 action 可关注当前和预测未来视频 tokens。
训练读离线 raw features，训练期 validation 也读离线特征；仿真每个 replan 在线提取一次。

## 先设置数据和 checkpoint 路径

```bash
cd /home/zhaoshizhen/lf/repos_new_1/FastWAM_3_track
export FASTWAM_ENV=/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam
export CUDA_HOME="$FASTWAM_ENV"
export GEOMETRY_PATHS="$PWD/configs/paths/libero_track4world_local.yaml"
source scripts/geometry_paths.sh
```

集群上把路径 YAML 复制到 `configs/paths/cluster.yaml` 并修改，然后设置
`GEOMETRY_PATHS="$PWD/configs/paths/cluster.yaml"`。主要配置项：

| 路径参数 | 内容 |
| --- | --- |
| `train_datasets` | 四套 LIBERO LeRobot 数据，含 meta/data/videos |
| `text_embedding_cache`、`dataset_stats` | 训练文本缓存及匹配数据的归一化统计 |
| `model_base`、`model_id`、`tokenizer_model_id` | Wan DiT、VAE、T5、tokenizer 的本地位置 |
| `action_dit_checkpoint` | ActionDiT 初始化权重 |
| `track4world_repo`、`track4world_checkpoint`、`da3_model` | Track4World/DA3 源码与权重 |
| `track4world_extra_pythonpath` | 已安装的几何依赖位置 |
| `geometry_cache` | 离线特征保存/读取位置 |
| `output_root` | 训练/评测输出根目录 |
| `libero_repo` | 普通 LIBERO 源码和仿真资产 |
| `libero_plus_repo`、`libero_plus_assets` | 仅 Plus 评测需要 |

joint 与 uncond 均从 **Wan + ActionDiT** 初始化；路径表中兼容字段
`fastwam_checkpoint` 虽含 uncond 文件名，但**不会用于初始化训练**。
`dataset_stats` 文件名可带 uncond，关键是统计对应相同数据/处理流程，不能仅凭文件名换统计。
依赖、权重、数据需在目标机器准备好；不是只 clone 源码就具备全部运行条件。

## 1. 离线提取（与 uncond 共用）

模型 variant 不参与 raw features 的定义。相同数据、顺序、几何配置、生产者权重的
既有完整缓存可以直接复用；缓存指纹不匹配时会拒绝，不要伪造 manifest。

```bash
# 新缓存先做两个样本的提取与离线/在线一致性检查
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_geometry.sh --indices 0,7
CUDA_VISIBLE_DEVICES=0 python scripts/libero_track4world.py \
  --paths "$GEOMETRY_PATHS" parity --indices 0,7

# 全量单卡；也可以按下面的16分片方法抽取
CUDA_VISIBLE_DEVICES=0 bash scripts/extract_geometry.sh

# 正式训练前必须通过完整覆盖检查
python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" coverage --require-complete
```

如果是单机16卡，可替代上面的全量单卡命令：

```bash
# 必须启动全部16个分片，且使用同一个共享 geometry_cache。
# 不要和全量单卡抽取同时启动。
pids=()
for gpu in {0..15}; do
  CUDA_VISIBLE_DEVICES="$gpu" NUM_SHARDS=16 SHARD_ID="$gpu" \
    bash scripts/extract_geometry.sh &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then echo 'Extraction failed; inspect worker output before training.'; else
  python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" coverage --require-complete
fi
```

多机抽取时 `SHARD_ID` 是全局编号，不是每机重复从0开始；所有机器的数据排列必须相同。

## 2. 16卡 joint 训练

```bash
# 几何组：只读取离线特征，不运行 Track4World 前向
GEOMETRY_ENABLED=true OUTPUT_DIR=/cluster/runs/joint_track_seed42 \
  bash scripts/train_joint_ablation_16gpu.sh

# 对照组：原 joint，无几何模块、无需离线缓存
GEOMETRY_ENABLED=false OUTPUT_DIR=/cluster/runs/joint_control_seed42 \
  bash scripts/train_joint_ablation_16gpu.sh
```

两组是分别提交的训练任务，不要无意中同时占用同一组16卡。
默认与仓库原 `train_libero_joint_16gpu.sh` 一致：B8、GAS1、global batch=128、
bf16、ZeRO-1、lr=1e-4、10 epoch、seed42。参数可用环境变量/末尾 Hydra 参数覆盖。
先检查命令：`DRY_RUN=1 bash scripts/train_joint_ablation_16gpu.sh`。
首次全尺寸运行建议另设输出目录，以 `MAX_STEPS=20` 做集群 smoke；小模型测试不证明全尺寸显存足够。

两机各8卡：每机执行同一训练命令，分别设置 `NODE_RANK=0/1`，并共同设置
`NNODES=2 GPUS_PER_NODE=8 MASTER_ADDR=主节点IP RUN_ID=同一标识 OUTPUT_DIR=同一共享目录`。
完整续训指定 `RESUME_STATE=/.../checkpoints/state/step_NNNNNN`。

joint/uncond checkpoint 形状可能相同但语义不同：joint sidecar 增加 `variant: joint`，
加载时拒绝跨 variant 混用。不要给 joint 测试传入旧的 uncond checkpoint，也不要删除/改写 sidecar。

## 3. joint 测试：在线提取

```bash
export CKPT=/cluster/runs/joint_track_seed42/checkpoints/weights/step_002000.pt

# 先一个普通 LIBERO task/trial
GEOMETRY_ENABLED=true CUDA_VISIBLE_DEVICES=0 NUM_TRIALS=1 \
  bash scripts/eval_joint_ablation.sh \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0

# 普通 LIBERO manager，多GPU评测
GEOMETRY_ENABLED=true EVAL_MODE=manager NUM_EVAL_GPUS=16 NUM_TRIALS=50 \
  bash scripts/eval_joint_ablation.sh

# Plus：提前配置专用源码/资产；默认完整任务清单，先用既有入口的任务筛选做小规模检查
FASTWAM_VARIANT=joint GEOMETRY_ENABLED=true NUM_GPUS=16 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 \
  bash scripts/eval_geometry_plus.sh
```

迁移时同时复制 `.pt`、相邻 `.pt.geometry.json`、训练输出的 `dataset_stats.json`。
统计未处于入口约定的位置时，传 `EVALUATION.dataset_stats_path=/path/dataset_stats.json`。
对照组测试用对应 **joint control** checkpoint，并设置 `GEOMETRY_ENABLED=false`。
joint 需要视频长度；wrapper 保留原 joint 方法签名，LIBERO 自动传入 `num_video_frames`。
即使不输出未来视频可视化，joint 也仍进行视频/动作联合去噪。

## 本机检查与范围

```bash
python -m pytest geometry_tests experiments/libero_plus/test -q
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 2 --main_process_port 29771 scripts/check_ablation_gpu.py \
  --variant joint --skip-online --paths "$GEOMETRY_PATHS" --output outputs/check_joint_train
python scripts/check_ablation_sim.py --variant joint --paths "$GEOMETRY_PATHS" \
  --checkpoint outputs/check_joint_train/checkpoints/weights/step_000004.pt \
  --output outputs/check_joint_sim
```

检查脚本只用两个真实样本、小尺寸实际 WAM/VAE 和真实 Track4World。
不代表完整5B/16卡、长期loss收敛或LIBERO/Plus成功率已经验证。
原核心 `fastwam.py`、`fastwam_joint.py`、`runtime.py` 和训练循环没有新增改动。
joint 的新增接线集中在 geometry 子类/factory、配置、启动脚本、Plus factory 识别与测试。
