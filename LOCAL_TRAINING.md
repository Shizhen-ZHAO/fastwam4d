# 官方 FastWAM：本机训练入口

本目录重新克隆自 `https://github.com/yuantianyuan01/FastWAM.git`，
上游 commit 为 `7faa71108368fbb3b6885649f112af607427a2d4`。
没有复制旧仓库的 geometry / Track4World 修改，没有改动官方 Python、YAML、DeepSpeed 或原启动脚本。
新增的 `scripts/train_libero_local.sh` 只集中设置本机路径和启动参数，最后调用官方 `scripts/train_zero1.sh`。
默认任务是 `libero_uncond_2cam224_1e-4`，即 FastWAM，不是 joint 或 IDM。

## 迁移前需要配置的路径

编辑 `scripts/train_libero_local.sh` 开头，或通过同名环境变量覆盖：

| 变量 | 含义 |
| --- | --- |
| `FASTWAM_ENV` | 安装好官方依赖的 Python 环境目录 |
| `CUDA_HOME` | CUDA 编译工具链目录，包含 `bin/nvcc` |
| `DATA_ROOT` | 包含四个 `libero_*_no_noops_lerobot` 子目录的 LeRobot 数据根目录 |
| `TEXT_CACHE` | 官方 T5 文本缓存目录，不是几何特征缓存 |
| `MODEL_BASE` | Wan 预训练权重根目录，保留下面列出的模型 ID 子目录结构 |
| `ACTION_DIT_CHECKPOINT` | 官方插值生成的 ActionDiT backbone `.pt` |
| `DATASET_STATS` | 与该数据集及官方 processor 配套的归一化统计 JSON |
| `OUTPUT_DIR` | 本次训练输出目录，建议每次新训练使用不同目录 |
| `RESUME_STATE` | 首次训练为 `null`；恢复训练时为完整 state 目录 |

`MODEL_BASE` 中用于本次训练的文件：

```text
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
```

路径通过官方支持的 `DIFFSYNTH_MODEL_BASE_PATH` 设置。脚本禁止自动下载；迁移时应先准备齐文件。
训练默认使用文本缓存，不加载 T5 编码器。上述路径不是已训练好的 `libero_uncond_2cam224.pt`；
本方案按官方流程从 Wan2.2 与 ActionDiT backbone 初始化。

## 双 A100 短流程检查

```bash
cd /home/zhaoshizhen/lf/repos_new_1/FastWAM
NUM_GPUS=2 BATCH_SIZE=1 MAX_STEPS=4 \
OUTPUT_DIR=/your/output/official_smoke \
bash scripts/train_libero_local.sh log_every=1 save_every=3 eval_every=0
```

这是短训练/保存测试，不是完整训练，更不是 LIBERO 仿真成功率评估。
不设置 `MAX_STEPS` 时使用官方任务的 10 个 epoch。
当前上游 FastWAM 不支持 `model.mot_checkpoint_mixed_attn=true`，不要添加该参数。

## 单机 16 卡

路径和环境准备完成后：

```bash
NUM_GPUS=16 BATCH_SIZE=1 \
OUTPUT_DIR=/your/output/official_train \
bash scripts/train_libero_local.sh
```

每卡 batch size 先保持 1，再依据集群实测显存增加。这不等于官方 README 的 8 卡、每卡 16 的全局 batch；
改变卡数或 batch 会改变每个 epoch 的更新次数。不要将双卡短流程验证当作 16 卡已经实测。

**如果是两台机器各 8 卡，不要直接执行上面的命令。** 本版本官方 `train_zero1.sh` 最后的
`accelerate launch` 没有传入多机参数，且其 YAML 默认 `num_machines: 1`。
需要根据集群的 SLURM/多机拓扑另外设置分布式启动参数；只改路径和 `NUM_GPUS=16` 不够。

## 恢复训练

使用官方完整状态目录，保持 GPU 数量、batch、数据路径和训练总步数等一致：

```bash
NUM_GPUS=2 BATCH_SIZE=1 MAX_STEPS=4 \
RESUME_STATE=/your/output/validated_run/checkpoints/state/step_NNNNNN \
OUTPUT_DIR=/your/output/official_resume \
bash scripts/train_libero_local.sh log_every=1 save_every=0 eval_every=0
```

`checkpoints/weights/step_*.pt` 是模型权重；`checkpoints/state/step_*` 才包含优化器、调度器、
随机状态和数据读取进度。不要把单独权重文件作为本流程的训练恢复目录。

## 环境与验证记录

独立环境已安装完成：Python 3.10、PyTorch 2.7.1+cu128、Accelerate 1.12.0、DeepSpeed 0.18.7，
依赖检查通过。但这不代表训练已稳定跑通：

- 旧环境（PyTorch 2.5.1、CUDA 12.1）的双卡预检完成了 2 个有限 loss 的优化步骤和完整状态保存。
  记录在 `runs/official_precheck_defaults_20260911`；仅两步，不能证明长期稳定或收敛。
- 独立官方依赖环境首先在优化器更新时 OOM；设置 `expandable_segments:True` 后，
  第 1 步 loss 正常，第 2 步开始 NaN，原因尚未确定，未实施模型或优化器源码修复。
- 当前训练任务已停止。不要恢复产生 NaN 的 checkpoint；上面的恢复路径仅为占位示例，
  完整状态恢复后的继续训练尚未验证。
- 新增的 16 卡 joint/unconditional 启动入口见 `TRAINING_16GPU.md`。
  已检查 shell 语法及配置解析，未进行 16 卡实际训练或多机通信验证。

运行日志和 checkpoint 不包含在 Git 上传内容中。
