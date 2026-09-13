# 完整 Agent Prompt：FastWAM joint + Track4World

请配置并分阶段运行本项目：LIBERO LeRobot 离线提取 Track4World 特征 → FastWAM joint 读取离线特征训练 → LIBERO 仿真在线提取特征测试。
以下是完整任务说明，不要把所有命令不加检查地一次运行。

## 0. 目标、源码版本与权限

- 必须使用真正的 `GeometryFastWAMJoint`，不能把 uncond task 改名来冒充 joint。
- 几何条件是当前 scene、当前 camera、最近8帧 track；经可训练 tokenizer 和一次三路残差 cross-attention 增强当前 VAE latent。
- 保留原 joint 的 full-video attention、视频/动作联合去噪、未来 target、loss、scheduler、优化器和训练循环；不要做不相关代码修改。
- 训练及训练期 validation 读取离线 raw features，冻结 Track4World 不做前向；仿真测试每个 replan 在线提取一次。
- 默认只配置路径、检查依赖和运行已有入口。发现需要改实现时先报告具体问题，不要绕过校验、伪造 sidecar 或悄悄更换模型。
- 只使用调度器分配/用户授权的 GPU，不停掉他人任务；大规模提取/训练必须在已分配资源中提交，不在登录节点直接启动。Mac 仅作源码同步，不用于 CUDA 实验。

源码来自 `FastWAM_3_track`，仓库为 `Shizhen-ZHAO/fastwam4d`，分支 `fastwam-3-track`。
必须获取包含本 prompt 和 joint 入口的新版本；旧提交 `f424d09` 只有 uncond 几何版本。
以下 clone 命令可在 Mac 和 Linux 使用。目标目录若已存在，先检查，不要删除或覆盖已有代码。

```bash
git clone --branch fastwam-3-track --single-branch \
  https://github.com/Shizhen-ZHAO/fastwam4d.git FastWAM_3_track
cd FastWAM_3_track
git log -1 --format='%H %s'
git status --short
```

如果已 clone **同一分支**且工作树干净，更新命令是：

```bash
cd /你的代码目录/FastWAM_3_track
git branch --show-current
git remote -v
# 确认分支为 fastwam-3-track，origin 指向上述 GitHub 仓库，再运行：
git pull --ff-only origin fastwam-3-track
```

若有本地修改、分支不符或快进失败，停止并报告，不 reset、不强推。本说明中的 origin 指新 clone 的远端；
源服务器的同名仓库可能另用 github 作为远端名，不能混淆。
私有仓库使用已授权的 HTTPS 凭据或 SSH；不将密码/token 写进命令或提交。
Mac 上可用 `pbcopy < JOINT_CLUSTER_AGENT_PROMPT.md` 复制本文全文，交给 Linux 集群上的 Agent。
权重/数据/环境不包含在 git clone 中；克隆不会自动创建已配置的 Track4World/DA3/LIBERO 安装。
不要自行重写几何模块，先检查以下文件：

```bash
cd /你的代码目录/FastWAM_3_track
git status --short
git rev-parse HEAD
test -f scripts/train_joint_ablation_16gpu.sh
test -f scripts/eval_joint_ablation.sh
test -f configs/task/libero_joint_geometry_ablation.yaml
test -f configs/model/fastwam_joint_geometry.yaml
rg -n 'class GeometryFastWAMJoint' src/fastwam/geometry/model.py
rg -n 'def create_joint_model' src/fastwam/geometry/runtime.py
```

任一检查失败就停止并请求正确源码。记录实际 commit 和本地修改；未提交的工作树不能只记录 HEAD 当作完整版本。
阅读 `TRACK4WORLD.md`、`JOINT_TRACK4WORLD.md`、`TRACK4WORLD_VERIFICATION.md`。

## 1. 先列出路径，再运行命令

所有占位路径必须替换为这台机器真实路径。保持 `configs/paths/libero_track4world_local.yaml` 不动，
在同目录新增 `cluster.yaml`，其内容如下。不得把 YAML 放到 `configs/paths` 之外。

```yaml
train_datasets:
  - /cluster/data/libero/libero_spatial_no_noops_lerobot
  - /cluster/data/libero/libero_object_no_noops_lerobot
  - /cluster/data/libero/libero_goal_no_noops_lerobot
  - /cluster/data/libero/libero_10_no_noops_lerobot
text_embedding_cache: /cluster/data/text_embeds_cache/libero
dataset_stats: /cluster/checkpoints/libero_dataset_stats.json

# 展示兼容字段，不是训练初始化，也不是测试实际加载的policy。
fastwam_checkpoint: null
model_base: /cluster/checkpoints/model_base
model_id: /cluster/checkpoints/Wan-AI/Wan2.2-TI2V-5B
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
redirect_common_files: true
action_dit_checkpoint: /cluster/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt

track4world_repo: /cluster/repos/Track4World
track4world_checkpoint: /cluster/checkpoints/track4world_da3.pth
da3_model: /cluster/checkpoints/DA3NESTED-GIANT-LARGE-1.1
track4world_extra_pythonpath: /cluster/deps/fastwam_track4world

libero_repo: /cluster/repos/LIBERO
libero_plus_repo: null
libero_plus_assets: null
geometry_cache: /cluster/data/geometry_cache_joint_v2
output_root: /cluster/runs/fastwam_joint_track
geometry_device: same
hf_endpoint: https://hf-mirror.com
```

路径含义和约束：

- 数据必须是当前 loader 可读的 LeRobot 格式，含 `meta/info.json`、`meta/episodes.jsonl`、`data/`、`videos/`；不是 raw HDF5 demonstrations。相机键为 `observation.images.image` 与 `observation.images.wrist_image`，默认20 FPS。
- 训练初始权重是 Wan DiT + ActionDiT，另需 VAE；训练文本来自预先准备好的 T5 cache。不能用旧 uncond policy 当作 joint 初始化。
- `redirect_common_files: true` 时，`model_base/DiffSynth-Studio/Wan-Series-Converted-Safetensors/` 下须有 `Wan2.2_VAE.safetensors` 和测试需要的 `models_t5_umt5-xxl-enc-bf16.safetensors`；tokenizer 位于 `model_base/<tokenizer_model_id>/google/umt5-xxl/`。若用其他权重布局，按已有 loader 规则设置，不要假装目录相同。
- DA3 目录包含 `config.json`、`model.safetensors`。Track4World 必须包含本版本要求的 DA3/Pi3/utils3d 源码与依赖；不要只拿一个未配置完整的上游目录。
- offline raw 缓存可在 joint/uncond 间共用，前提是数据内容、根目录排列、窗口/预处理、producer 代码与权重相同。路径移动可以，修改内容不行。
- 复制缓存必须整体复制，包含 `manifest.json`、逐 episode `.h5` 与 `.lock`。共享盘须支持文件锁/硬链接，不要边训练边写同一缓存。
- 训练虽然不运行 tracker，但仍需本地 Track4World/DA3 权重和源码用于 producer 指纹校验。
- 当前普通 LIBERO 在线测试入口也会预检训练数据/文本缓存等路径，测试机上需保留这些有效路径或挂载；它不读取离线 raw 缓存。不要擅自删掉检查。
- `dataset_stats` 必须匹配本次训练数据与处理器。文件名带 uncond 不等于统计不可共用；内容才是依据。测试优先显式指定本次训练输出的统计文件。

激活目标 Linux CUDA 环境并设置路径：

```bash
export REPO_ROOT=/cluster/repos/FastWAM_3_track
export FASTWAM_ENV=/cluster/envs/lf_fastwam
export CUDA_HOME=/cluster/cuda/toolkit
export GEOMETRY_PATHS="$REPO_ROOT/configs/paths/cluster.yaml"
export PATH="$CUDA_HOME/bin:$PATH"
cd "$REPO_ROOT"
source scripts/geometry_paths.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export TOKENIZERS_PARALLELISM=false

python -c 'import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda); print(torch.cuda.device_count())'
nvcc --version
nvidia-smi
python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" doctor
```

`CUDA_HOME` 指向实际 toolkit（须有 nvcc），不一定等于 Python 环境。
本机参考环境是 torch2.5.1/CUDA12.1，不要盲目升级共享环境；已有 PyAV fallback 可工作，TorchCodec 告警不等于必须改版本。
上述脚本使用离线模式，不会因设置 hf_endpoint 自动下载。若缺权重，先报告缺少的文件，按授权通过 HF 镜像准备后再执行；不要在16个 rank 内同时下载。
打印最终数据、初始化权重、缓存、统计和输出路径，确认没有遗留旧机器 `/home/...` 路径。

## 2. 先做单测、两样本提取和 parity

```bash
python -m pytest geometry_tests experiments/libero_plus/test -q -rs

# 在已分配的一张GPU上；CUDA_VISIBLE_DEVICES=0仅适用于该GPU确实已获分配。
CUDA_VISIBLE_DEVICES=0 NUM_SHARDS=1 SHARD_ID=0 \
  bash scripts/extract_geometry.sh --indices 0,7
CUDA_VISIBLE_DEVICES=0 python scripts/libero_track4world.py \
  --paths "$GEOMETRY_PATHS" parity --indices 0,7
python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" verify --indices 0,7
```

检查 mask 完全一致、浮点误差满足默认阈值（relative RMSE≤0.005，cosine≥0.999）。不要要求 GPU 浮点计算逐 bit 完全相同，也不要放宽阈值掩盖问题。
原实现按内容计算数据/视频/权重指纹，启动时可能较慢；不要把 I/O 等待当作死锁而删掉指纹。
当前提取流程保存的是冻结模型的 raw scene/camera/track，不是可训练 tokenizer 的输出。

可以在两张获分配的GPU上做小模型端到端接线检查：

```bash
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 2 --main_process_port 29773 scripts/check_ablation_gpu.py \
  --variant joint --skip-online --paths "$GEOMETRY_PATHS" \
  --output "$REPO_ROOT/outputs/check_joint_train"
python scripts/check_ablation_sim.py --variant joint --paths "$GEOMETRY_PATHS" \
  --checkpoint "$REPO_ROOT/outputs/check_joint_train/checkpoints/weights/step_000004.pt" \
  --output "$REPO_ROOT/outputs/check_joint_sim"
```

使用新的检查输出目录；不要把小模型 checkpoint 传给正式5B评测。这一步只验证小模型接线，不代表全尺寸生产训练通过。

## 3. 全量离线提取

有相同合同的完整缓存则直接验证并复用；不完整则继续提取。不要删除已有缓存。
单卡与16分片方案二选一，不能同时运行。

单卡：

```bash
NUM_SHARDS=1 SHARD_ID=0 bash scripts/extract_geometry.sh
```

单机16卡：将下面代码作为独立 Bash 作业脚本，在分配到整机16卡的资源内运行。
如果调度器使用 UUID/MIG 或重映射 GPU，先核对映射，不要机械覆盖 `CUDA_VISIBLE_DEVICES`。

```bash
set -euo pipefail
source "$REPO_ROOT/scripts/geometry_paths.sh"
export EXTRACT_LOG_DIR=/cluster/logs/joint_extract_run1
mkdir -p "$EXTRACT_LOG_DIR"
pids=()
for gpu in {0..15}; do
  CUDA_VISIBLE_DEVICES="$gpu" NUM_SHARDS=16 SHARD_ID="$gpu" \
    bash scripts/extract_geometry.sh >"$EXTRACT_LOG_DIR/shard_${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if (( failed )); then
  echo '至少一个提取进程失败，查看各分片日志；不要开始训练。' >&2
  exit 1
fi
python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" coverage --require-complete
```

两机各8卡时，每机只启动8个提取进程；总 `NUM_SHARDS=16`，节点0用 `SHARD_ID=0..7`，节点1用 `SHARD_ID=8..15`，各机可见卡编号按实际分配设置。共享一个缓存目录，且数据根目录排列一致。

无论哪种提取方案，全部进程结束后必须执行：

```bash
python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" coverage --require-complete
python scripts/libero_track4world.py --paths "$GEOMETRY_PATHS" verify
```

coverage 检查全量提交标记，verify 全量读取并校验窗口身份、tensor checksum、shape；后者可能有较大 I/O。
只有完整覆盖、verify 成功才进入正式训练。不能用 `--indices 0,7` 的成功冒充完整数据集。
先估计缓存空间/配额与抽取时间；每个时刻保存高维多路特征，全量不是几十 MB 的小文件。

## 4. 对照原 joint 配方并做16卡全尺寸 smoke

参考 `scripts/train_libero_joint_16gpu.sh`、`scripts/train_libero_16gpu_common.sh`、
`configs/task/libero_joint_2cam224_1e-4.yaml`，只使用新增的 joint geometry 包装入口：

```bash
DRY_RUN=1 GEOMETRY_ENABLED=true BATCH_SIZE=8 GRAD_ACCUM=1 \
  bash scripts/train_joint_ablation_16gpu.sh
```

应为真正 joint task/factory、16 rank、B8/GAS1/global batch128、bf16、ZeRO-1、lr1e-4、seed42。
DRY_RUN 只展示命令，不证明数据/显存或模型已通过。

先用完整缓存和正式模型尺寸跑20步，单机16卡命令：

```bash
GEOMETRY_ENABLED=true NNODES=1 GPUS_PER_NODE=16 NODE_RANK=0 \
BATCH_SIZE=8 GRAD_ACCUM=1 MAX_STEPS=20 RESUME_STATE=null \
OUTPUT_DIR=/cluster/runs/joint_track_smoke20 \
bash scripts/train_joint_ablation_16gpu.sh \
  save_every=10 eval_every=10 log_every=1 wandb.enabled=false
```

检查所有 rank 正常、loss/梯度有限、主干与几何参数确实更新、训练时没有 tracker 前向，
以及 `.pt.geometry.json`/`geometry_state.json` 已保存。统计实际显存、step耗时和loss趋势。
20步loss不要求单调下降；不能据此宣称几何有效。OOM/NaN 时先记录日志和定位，不擅自改 loss/去噪逻辑。
若必须调 batch/GAS，保持总 batch 并对 control/geometry 两组采用相同参数，明确记录偏离原配方之处。

## 5. 正式 joint 离线特征训练

正式从相同初始化重新训练，使用与 smoke 不同的输出目录；不要拿改变了 max_steps 的 smoke state 无说明地续训。

```bash
GEOMETRY_ENABLED=true NNODES=1 GPUS_PER_NODE=16 NODE_RANK=0 \
BATCH_SIZE=8 GRAD_ACCUM=1 MAX_STEPS=null RESUME_STATE=null SEED=42 \
OUTPUT_DIR=/cluster/runs/joint_track_seed42 \
bash scripts/train_joint_ablation_16gpu.sh
```

可选无几何对照组，必须另行排期，不能同时抢占同一组GPU：

```bash
GEOMETRY_ENABLED=false NNODES=1 GPUS_PER_NODE=16 NODE_RANK=0 \
BATCH_SIZE=8 GRAD_ACCUM=1 MAX_STEPS=null RESUME_STATE=null SEED=42 \
OUTPUT_DIR=/cluster/runs/joint_control_seed42 \
bash scripts/train_joint_ablation_16gpu.sh
```

两机各8卡：在**两台节点各执行一次**对应训练命令，把拓扑替换为：

```bash
export NNODES=2 GPUS_PER_NODE=8
export NODE_RANK=0                 # 另一台为1
export MASTER_ADDR=主节点实际IP
export MASTER_PORT=29500
export RUN_ID=joint_track_seed42
export OUTPUT_DIR=/cluster/runs/joint_track_seed42
GEOMETRY_ENABLED=true BATCH_SIZE=8 GRAD_ACCUM=1 MAX_STEPS=null RESUME_STATE=null \
  bash scripts/train_joint_ablation_16gpu.sh
```

各节点的 RUN_ID/OUTPUT_DIR/参数相同，OUTPUT_DIR 是共享目录，master端口空闲可达；
不要使用上面的单机 `NNODES=1` 行覆盖多机设置。脚本的 `--num_processes 16` 是全局进程数。

同配置完整续训时，使用上一训练的完整 state 目录，而非 policy `.pt`：

```bash
GEOMETRY_ENABLED=true \
RESUME_STATE=/cluster/runs/joint_track_seed42/checkpoints/state/step_002000 \
OUTPUT_DIR=/cluster/runs/joint_track_seed42 \
bash scripts/train_joint_ablation_16gpu.sh
```

此例 step_002000 必须确实存在，且拓扑等参数与实际资源一致。
几何state要求合同一致；不伪造 sidecar，不把旧 uncond checkpoint/state 混入 joint。

## 6. 在线提取特征的正式 joint 测试

选取实际存在的本次 joint geometry 训练权重，明确列出：

```bash
export CKPT=/cluster/runs/joint_track_seed42/checkpoints/weights/step_002000.pt
export EVAL_STATS=/cluster/runs/joint_track_seed42/dataset_stats.json
test -f "$CKPT"
test -f "$CKPT.geometry.json"
test -f "$EVAL_STATS"
```

训练默认仅保留最近两组 checkpoint，step_002000 可能已经清理；请从输出目录选择实际保留的 step，不要机械照抄。
检查 sidecar 的 `variant` 是 `joint`，几何配置/producer匹配。迁移同时携带 policy、sidecar、stats；正式测试需本地 T5 encoder 和 tokenizer，不是小模型检查用的缓存T5替代。

先单个 LIBERO task/trial：

```bash
GEOMETRY_ENABLED=true EVAL_MODE=single NUM_TRIALS=1 \
EVAL_OUTPUT_DIR=/cluster/eval/joint_track_smoke \
bash scripts/eval_joint_ablation.sh \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 \
  EVALUATION.dataset_stats_path="$EVAL_STATS"
```

应确认：使用真实外部视角/腕部相机、每环境步记录历史且每episode重置、仅使用当前及过去、每个replan提取一次，动作有限；
joint 接口正确收到 `num_video_frames`，进行视频/动作联合去噪。即使不保存视频可视化，也不是 uncond action-only 推理。

通过后普通 LIBERO 全四套评测（单机16张已分配GPU）：

```bash
GEOMETRY_ENABLED=true EVAL_MODE=manager NUM_EVAL_GPUS=16 NUM_TRIALS=50 \
EVAL_OUTPUT_DIR=/cluster/eval/joint_track_all \
bash scripts/eval_joint_ablation.sh \
  EVALUATION.dataset_stats_path="$EVAL_STATS"
```

普通 manager 是单机多GPU，不接受训练的 NNODES 配置来自动变成多机评测。
不要继承只含一张卡的 `CUDA_VISIBLE_DEVICES` 去跑16GPU manager。输出目录用新的目录，防止已有结果被当作已完成任务跳过。
control测试改用对应 joint control checkpoint，设置 `GEOMETRY_ENABLED=false`，其余 seed/replan/denoise/trial 参数一致。

## 7. 可选 LIBERO Plus

先配置 Plus 专用 repo/assets，不得把普通LIBERO资产冒充Plus。Plus入口已做配置/worker测试，但提供机器未做真实Plus完整rollout。
先准备一个只含一个有效 `suite,task_id` 的任务清单（从已验证的Plus清单选择），例如路径 `/cluster/eval/plus_one_task.txt`；不要把它当作完整任务清单覆盖原文件。

```bash
FASTWAM_VARIANT=joint GEOMETRY_ENABLED=true \
TASK_FILE=/cluster/eval/plus_one_task.txt NUM_GPUS=1 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 \
DATASET_STATS_PATH="$EVAL_STATS" OUTPUT_DIR=/cluster/eval/joint_plus_smoke \
bash scripts/eval_geometry_plus.sh
```

验证真实Plus小规模rollout、显存和日志后，才可运行完整10030项清单：

```bash
FASTWAM_VARIANT=joint GEOMETRY_ENABLED=true \
TASK_FILE="$REPO_ROOT/experiments/libero_plus/full_10030_lpt.txt" \
NUM_GPUS=16 MAX_TASKS_PER_GPU=1 NUM_TRIALS=1 \
DATASET_STATS_PATH="$EVAL_STATS" OUTPUT_DIR=/cluster/eval/joint_plus_all \
bash scripts/eval_geometry_plus.sh
```

它是另外一个评测任务，不与普通LIBERO共用运行目录。先报告预计耗时再提交大规模评测，不要误把1 trial理解为只跑1个task。

## 8. 必须交付的验收报告

分别报告，不要只写“全部跑通”：

1. 实际代码版本/本地diff、环境、GPU拓扑、数据/checkpoint/cache/stats/输出路径。
2. 提取 expected/written/missing 数、全量verify结果、parity误差、耗时和空间占用。
3. 正式训练解析出的 task/factory、B/GAS/global batch、训练步数、loss走势、显存/step耗时、是否存在NaN/Inf、保存与恢复结果。
4. 测试用哪个joint policy及sidecar、哪个stats，真实在线提取是否运行；task/trial数量、成功率和异常任务分别列出。
5. 哪些只做了单测/小模型测试，哪些做了全尺寸真实运行，哪些尚未执行或失败。

已知提供机器上通过94项测试、2项因缺参考仓库跳过，做过双A100小尺寸joint四步训练和真实LIBERO两步在线测试。
这些证据不等于你这台机器已通过完整5B/16卡，不证明长期收敛或加入几何提高成功率。
发现上述边界之外的问题，保留日志并解释，不得用小模型结果替代完整生产验收。
