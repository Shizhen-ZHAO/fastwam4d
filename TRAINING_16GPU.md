# LIBERO：官方 FastWAM joint / unconditional / IDM，16 卡启动

三个入口（均不含 geometry）：

- `scripts/train_libero_joint_16gpu.sh` → 官方 `libero_joint_2cam224_1e-4`。
- `scripts/train_libero_uncond_16gpu.sh` → 官方 `libero_uncond_2cam224_1e-4`。
- `scripts/train_libero_idm_16gpu.sh` → 官方 `libero_idm_2cam224_1e-4`，标准 IDM，不是 optional-IDM。

三个入口依赖 `scripts/train_libero_16gpu_common.sh`，迁移时四个文件都要保留。
共用启动器参考官方 `scripts/train_zero1.sh`，使用相同的训练入口、Accelerate YAML 和
DeepSpeed ZeRO-1 JSON，显式补充多机参数和 `standard` launcher（每节点启动一次，不使用 pdsh/SSH 自动启动）。
没有修改官方 Python、模型结构、loss 或配置文件。

## 先修改路径

每个入口脚本开头均可编辑，也可使用同名环境变量覆盖：

- `FASTWAM_ENV`：已安装依赖的 Python 环境目录。
- `CUDA_HOME`：包含 `bin/nvcc` 的 CUDA 工具链目录。
- `DATA_ROOT`：包含四个 `libero_*_no_noops_lerobot` 子目录的数据根目录。
- `TEXT_CACHE`：官方 LIBERO T5 文本缓存目录，不是几何缓存。
- `MODEL_BASE`：Wan 初始化权重根目录。
- `ACTION_DIT_CHECKPOINT`：插值后的 ActionDiT backbone 初始化权重。
- `DATASET_STATS`：与训练数据和 processor 匹配的归一化 JSON；三种模型可共用。
- `OUTPUT_DIR`：训练输出目录；多机必须共享且一致，不同模型的实验必须分开。
- `RESUME_STATE`：首次训练保持 `null`；恢复时填写对应模型的完整训练 state 目录。

`MODEL_BASE` 保持如下结构（与本机现有权重一致）：

```text
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

默认禁止自动下载，迁移前准备齐数据、文本缓存和权重。支持的数据格式取决于对应官方 data 配置，
本脚本沿用本机已检查的 LeRobot 数据配置，不会自动转换数据布局。

## 单机 16 卡：分别执行，不要在同一组 GPU 上同时启动

在仓库根目录，修改好路径后：

```bash
bash scripts/train_libero_joint_16gpu.sh
# 上一个实验结束后，再运行另一个：
bash scripts/train_libero_uncond_16gpu.sh
# 上一个实验结束后，再运行 IDM：
bash scripts/train_libero_idm_16gpu.sh
```

默认每卡 batch=1，梯度累积=1，全局 batch=16；workers=2/进程。
这不是官方 task 的每卡 batch=16。学习率仍为官方 1e-4，训练 10 epoch，其他参数沿用官方配置。
可通过 `BATCH_SIZE`、`GRAD_ACCUM`、`NUM_WORKERS`、`MAX_STEPS` 环境变量覆盖；
改变 batch 后不保证同样的收敛表现，不会自动缩放学习率。

只打印命令（不启动训练）：

```bash
DRY_RUN=1 bash scripts/train_libero_joint_16gpu.sh
DRY_RUN=1 bash scripts/train_libero_uncond_16gpu.sh
DRY_RUN=1 bash scripts/train_libero_idm_16gpu.sh
```

先进行短测试的示例（三种模型都应分别测）：

```bash
MAX_STEPS=20 OUTPUT_DIR=/your/output/joint_smoke \
bash scripts/train_libero_joint_16gpu.sh log_every=1 eval_every=0 save_every=0
```

短测试会改变学习率调度长度，不是正式训练的前 20 步等价复现；正式训练使用新输出目录，
不设置 `MAX_STEPS`，不要将非有限 loss 的 checkpoint 用于恢复。

## 两台机器各 8 卡

需要调度器先分配两台机器各 8 张可见 GPU。两台机器使用相同代码和依赖；数据、权重路径均需有效，
输出目录必须是共享存储。以下两条命令分别在两个节点上近乎同时执行，不是在同一机器顺序执行。

节点 0（将 IP、输出目录替换成真实值）：

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
RUN_ID=joint_run01 OUTPUT_DIR=/shared/outputs/joint_run01 \
bash scripts/train_libero_joint_16gpu.sh
```

节点 1：

```bash
NNODES=2 GPUS_PER_NODE=8 NODE_RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
RUN_ID=joint_run01 OUTPUT_DIR=/shared/outputs/joint_run01 \
bash scripts/train_libero_joint_16gpu.sh
```

unconditional 实验：两个节点均换成 `train_libero_uncond_16gpu.sh`，并使用新的、两端一致的
`RUN_ID` 和 `OUTPUT_DIR`。IDM 实验同理，两个节点均换成 `train_libero_idm_16gpu.sh`，
并使用新的、两端一致的 `RUN_ID=idm_run01` 和 `OUTPUT_DIR=/shared/outputs/idm_run01`。
端口需允许节点间通信，集群网络/NCCL 设置按当地配置。
脚本不负责申请 SLURM 资源。

## 验证边界

已检查 shell 语法、三种官方 Hydra 模型目标、路径参数传递、单机 16 卡和两机各 8 卡参数，
以及不合法 GPU 数量的拒绝逻辑。未进行 16 卡实际训练或多机通信测试。
此前双卡官方依赖环境出现第 2 步 NaN，原因尚未确定；不能承诺增至 16 卡后消失。
