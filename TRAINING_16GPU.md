# LIBERO：FastWAM joint / unconditional / IDM，16 卡训练

三个入口（均不含 geometry）：

- `scripts/train_libero_joint_16gpu.sh` → 官方 `libero_joint_2cam224_1e-4`。
- `scripts/train_libero_uncond_16gpu.sh` → 官方 `libero_uncond_2cam224_1e-4`。
- `scripts/train_libero_idm_16gpu.sh` → 官方 `libero_idm_2cam224_1e-4`，标准 IDM，不是 optional-IDM。

三个入口依赖 `scripts/train_libero_16gpu_common.sh` 与 `scripts/libero_cluster_paths.sh`，迁移时五个文件都要保留。
共用启动器参考官方 `scripts/train_zero1.sh`，使用相同的训练入口、Accelerate YAML 和
DeepSpeed ZeRO-1 JSON，显式补充多机参数和 `standard` launcher（每节点启动一次，不使用 pdsh/SSH 自动启动）。
当前版本已按 `fastwam4d_pp` 对齐三个模型的1d训练配置、初始化、VAE编码和trainer行为。
逐microbatch精确对照、依赖环境及集群验收见 [TRAINING_ALIGNMENT.md](TRAINING_ALIGNMENT.md)。

## 先修改路径

所有路径统一在 `scripts/libero_cluster_paths.sh` 编辑，也可使用同名环境变量覆盖。环境目录留空时使用当前已激活环境：

- `FASTWAM_ENV`：已安装依赖的 Python 环境目录。
- `CUDA_HOME`：包含 `bin/nvcc` 的 CUDA 工具链目录。
- `DATA_ROOT`：包含四个 `libero_*_no_noops_lerobot` 子目录的数据根目录。
- `TEXT_CACHE`：官方 LIBERO T5 文本缓存目录，不是几何缓存。
- `MODEL_BASE`：Wan 初始化权重根目录。
- `ACTION_DIT_CHECKPOINT`：插值后的 ActionDiT backbone 初始化权重。
- `DATASET_STATS`：默认null，与参考一样在训练时计算，保存OUTPUT_DIR/dataset_stats.json；指定JSON则复用。严格run_pair对照先用prepare_stats.py生成一次。
- `OUTPUT_DIR`：训练输出目录；多机必须共享且一致，不同模型的实验必须分开。
- `RESUME_STATE`：首次训练保持 `null`；恢复时填写对应模型的完整训练 state 目录。
- `MODEL_ID` / `TOKENIZER_MODEL_ID` / `REDIRECT_COMMON_FILES`：可选，覆盖模型目录解析方式，便于使用平铺权重目录。

Wan默认采用用户提供的平铺目录，`MODEL_BASE`是其父目录：

```text
/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/Wan2.2-TI2V-5B/
  diffusion_pytorch_model*.safetensors
  Wan2.2_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth   # 评测/预计算文本需要

REPO_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
REPO_ROOT/data/text_embeds_cache/libero/
OUTPUT_DIR/dataset_stats.json     # 默认训练时自动生成
/new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P/google/umt5-xxl/
```

ActionDiT/cache/tokenizer均按参考启动脚本及其model/data配置确定，Python/CUDA沿用当前已激活环境。
`MODEL_ID=Wan2.2-TI2V-5B`、`REDIRECT_COMMON_FILES=false`；旧布局仍可用环境变量覆盖。
详见 [ALIGNMENT_AGENT_HANDOFF.md](ALIGNMENT_AGENT_HANDOFF.md) 的路径来源与一次性stats准备命令。

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

默认uncond每卡batch=4、梯度累积=2；joint和IDM每卡batch=8、梯度累积=1。
三者全局batch均为128，workers=8/进程；action RoPE均为1d，action/video shift均为5。
学习率1e-4，训练10 epoch。这里以参考脚本实际解析值为准，不采用其过时注释。
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

已检查三个模型的Hydra配置、路径/拓扑传递，并完成CPU小模型的实际VAE、loss、梯度和多步AdamW对照。
这些证据不代替真实16卡训练：集群必须按TRAINING_ALIGNMENT.md验证每个rank、每个microbatch和参数更新。
此前记录的双卡第2步NaN也需要在实际环境重新验证，不能仅凭本地对照认为已经解决。
