# 三个 FastWAM 模型的1d训练对齐与集群验收

目标代码基线：`acaa958`。参考：`fastwam4d_pp` 的 `19d989111193ec2b69ec00493ef250dff531ad03`。
本次直接修改目标仓库；参考仓库保持不变。用户确认 **uncond、joint、IDM 的 action RoPE 都使用1d**。

## 当前代码做了什么

| 模型 | 本仓库入口 | 参考入口 | 每GPU batch | GAS | 全局batch |
|---|---|---|---:|---:|---:|
| uncond | `scripts/train_libero_uncond_16gpu.sh` | `train_libero_offline_16gpu.sh` | 4 | 2 | 128 |
| joint | `scripts/train_libero_joint_16gpu.sh` | `train_libero_joint_offline_16gpu.sh` | 8 | 1 | 128 |
| IDM | `scripts/train_libero_idm_16gpu.sh` | `train_libero_idm_offline_16gpu.sh` | 8 | 1 | 128 |

三者均16GPU、workers8、action/video train/infer shift5、LR1e-4、weight decay1e-2、AdamW betas=(0.9,0.95)、10epoch、seed42。
task覆盖仍关闭mixed-attention checkpoint和denoise compile；基础YAML的batch字面量由launcher覆盖。不要绕过三个launcher后假定仍是同一配方。

具体修改：

- 模型初始化前设seed，trainer仍为数据流再次设seed。
- 三个模型共用的VAE视频/首帧编码改为参考逐样本eager，恢复VAE.encode入口。
- ActionDiT显式接受且只允许`action_rope_mode=1d`，错误模式直接拒绝。
- IDM直接构造默认`video_cond_noise_prob=0.5`，与参考版及原预训练工厂一致。
- trainer采用参考的梯度累积插件、DDP调用方式、日志通信、checkpoint重定向/保留数及验证视频容错。
- 三个任务的action scheduler覆盖5；保留最近2组checkpoint。`FASTWAM_CKPT_DIR`可重定向，多个run不得共用该目录。
- `pyproject.toml`中的DeepSpeed、datasets、numpy、av、pyarrow、torchcodec、tqdm固定到参考声明版本；共有训练依赖版本已对照。没有引入仅用于画图的seaborn。
- 原有所有机器路径环境变量、多机拓扑继续可用；新增可选MODEL_ID/TOKENIZER_MODEL_ID/REDIRECT_COMMON_FILES覆盖。

参考脚本的注释有过时值：joint实际B8/GAS1，IDM实际B8/GAS1；IDM的实际training_loss同时包含video/action loss，未按注释删掉video loss。

为逐步复现现有参考行为，本次**没有单侧修正**参考也存在的DeepSpeed forward/GAS缩放、DS裁剪配置及prepare后weights-only加载行为。默认初始训练使用`RESUME_STATE=null`。文件权重热启动和full-state恢复是不同操作，不能互换。后续若修正共有训练语义，需要两侧同步修改、另建实验基线。

## 本地已验证，集群仍需验证

本地CPU float32、小随机模型：三种模型B1/B2，每例4次AdamW更新；IDM另外覆盖cond_noise_prob=0与1。合计8个案例、32次更新、40个microbatch。使用实际VAE、proprio、padding和各模型原始training_loss，未把loss替换成测试公式。

所有案例初始参数、每个microbatch的loss/loss分量、139个非空梯度张量、随机流，以及每次更新后的参数均完全一致。没有发现需要整体替换tensor版MoT/VideoDiT的证据，因此保留现有核心接口。

此外，已验证三个模型在单机16/双机2×8的Hydra有效配置；对照记录器在CPU toy trainer中不改变loss、参数和随机流，能检出1e-10的loss差异及不完整记录。

本地结果随仓库保存在 [scripts/alignment/evidence](scripts/alignment/evidence)：model_parity、config_parity、trace_check及启动计划/静态检查JSON。启动计划检查使用假的训练子进程，未运行GPU。

**这些测试没有真实权重、真实数据解码、CUDA/bf16、ZeRO、多机通信或LIBERO。当前只能说代码修改和本地对照完成，不能说集群每步loss已经相同。** 后续以本节的零容差工具实测为验收标准。

## 准备同一运行环境与资产

两个repo必须在同一组GPU、同一Python/依赖/CUDA/驱动环境中顺序运行。不要分别安装两份editable fastwam后依赖安装顺序选包；对照入口显式把所选repo/src置于首位，并验证`fastwam.__file__`。

在训练环境安装本仓库声明的依赖。torch/torchvision使用与项目一致的cu128源，并确认DeepSpeed编译工具链可用；现有集群环境若要保留，可以另建环境。安装后先执行`python -m pip check`并保存`python -m pip freeze`、`nvidia-smi`。对照器会检查两次运行的主要包版本、GPU型号、CUDA/cuDNN、确定性与SDPA设置。

在两侧共用的 shell 中设置路径；目标仓库现在只有一份公共路径配置：

```bash
export TARGET_REPO=/shared/code/FastWAM
export REFERENCE_REPO=/shared/code/fastwam4d_pp
cd "$TARGET_REPO"
source scripts/libero_cluster_paths.sh
# 已写入公共脚本的确认路径：
# DATA_ROOT=/new-interaction/share/user_folder/jiahao.ljh/data/LIBERO-fastwam
# MODEL_BASE=/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints
# MODEL_ID=Wan2.2-TI2V-5B
# REDIRECT_COMMON_FILES=false

# ActionDiT和文本cache沿参考仓库相对路径，环境沿用当前已激活shell。
# 默认DATASET_STATS=null：普通训练自动计算，写入本次run目录。
mkdir -p /shared/review
python scripts/alignment/check_assets.py --mode train --output /shared/review/assets.json

# 严格对照前，在一个节点用普通python执行一次；无需另找stats路径。
python scripts/alignment/prepare_stats.py \
  --reference-repo "$REFERENCE_REPO" \
  --output /shared/review/libero_reference_dataset_stats.json
export DATASET_STATS=/shared/review/libero_reference_dataset_stats.json
```

prepare_stats直接调用参考repo的数据构造函数，不加载模型、不修改源码。文件已存在时拒绝覆盖，同一数据配置下可直接复用。两个节点必须export同一共享文件。run_pair拒绝null stats，以避免各次线程化统计中mean/std字段的舍入差异影响严格hash对照。

`FASTWAM_ENV`、`CUDA_HOME` 留空时启动器和参考一样使用当前环境；若显式设置，需先将它们的bin/lib加到当前shell的PATH/LD_LIBRARY_PATH，再执行run_pair。参考README使用`conda activate fastwam`，脚本自身不切换环境。

ActionDiT位于`$TARGET_REPO/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt`，cache位于`$TARGET_REPO/data/text_embeds_cache/libero`；也可直接覆盖为参考repo中的对应路径，让两边共用同一文件。数据按spatial、object、goal、10顺序包含四个子集；标准v2像素路径、两相机横拼、无latent cache。

Wan使用用户指定目录；tokenizer沿参考完整位置`/new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P/google/umt5-xxl/`。T5和VAE在Wan目录。更详细的路径来源与评测stats自动查找见 [ALIGNMENT_AGENT_HANDOFF.md](ALIGNMENT_AGENT_HANDOFF.md)。

## 先运行不访问GPU的检查

```bash
cd "$TARGET_REPO"
python scripts/alignment/check_configs.py --reference-repo "$REFERENCE_REPO" \
  --output /shared/review/config_parity.json
python scripts/alignment/check_model_parity.py --reference-repo "$REFERENCE_REPO" \
  --output /shared/review/cpu_model_parity.json
python scripts/alignment/check_trace.py --output /shared/review/trace_check.json
```

CPU模型检查只需相应torch/numpy/einops/Pillow/OmegaConf等Python依赖，无需checkpoint。输出目录可由脚本创建。三份脚本都是检查，不会修改参考源码。

## 单机16卡：逐步对照

先分配独占的16张同型号GPU，在同一个环境执行：

```bash
export NNODES=1
export GPUS_PER_NODE=16
export NODE_RANK=0

python scripts/alignment/run_pair.py \
  --reference-repo "$REFERENCE_REPO" \
  --output-root /shared/review/alignment_run01 \
  --variant all --steps 8 --workers 0 --dry-run
```

DRY_RUN会从三个实际目标launcher取参数，然后分别用两repo的Hydra配置解析。参考模型组分别为`fastwam_3d`、`fastwam_joint_ppu`、`fastwam_idm`，显式覆盖action1d；batch/GAS采用已核对的参考脚本实际值。实际训练调用两侧自己的`runtime.run_training`、trainer、dataset、model和optimizer，不使用另写的训练循环。

检查六条展开命令的路径和拓扑后，去掉`--dry-run`：

```bash
python scripts/alignment/run_pair.py \
  --reference-repo "$REFERENCE_REPO" \
  --output-root /shared/review/alignment_run01 \
  --variant all --steps 8 --workers 0
```

每个模型先跑reference，再跑target，随后严格比较；全部模型完成后得到：

```text
alignment_run01/
  uncond/
    reference/plan_node_0.json
    reference/train/...
    reference/trace/rank_00000.jsonl ... rank_00015.jsonl
    target/plan_node_0.json
    target/train/...
    target/trace/rank_00000.jsonl ... rank_00015.jsonl
    comparison.json
  joint/...
  idm/...
```

每次实验使用新output-root。对照模式将FASTWAM_CKPT_DIR及研究环境变量清除，避免两个run混写；原始repo内容不修改。
`--variant uncond|joint|idm`可单独运行。`--role reference|target`可分开调度，必须使用同一个output-root并手工执行下面的比较命令。

第一次用workers0排除prefetch干扰；通过后用**新目录**加`--workers 8`，按目标训练数据加载设置再次验证。batch/GAS仍保持各模型正式配方，不因短跑而减小。

对照模式固定`resume=null`，默认eval_every0、save_every0、log_every1、keep_last0。**trainer结束仍会保存最终weights和完整state**，需要预留磁盘空间。8步改变了warmup/余弦调度总长，比较的是两份同样的8步实验，不等于正式10epoch的前8步。

## 两机各8卡

两台机器都准备相同代码、同一Python环境和可访问的共享输出/资产路径；在两个节点上近乎同时运行。

```bash
# 两个节点共同设置；MASTER_ADDR替换为rank0节点真实IP。
export NNODES=2
export GPUS_PER_NODE=8
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
export RUN_ID=alignment_run02

# 节点0使用0；节点1使用1。
export NODE_RANK=0

python scripts/alignment/run_pair.py \
  --reference-repo "$REFERENCE_REPO" \
  --output-root /shared/review/alignment_run02 \
  --variant all --steps 8 --workers 0
```

每节点启动一次同一命令；global world size仍为16。每节点写自己的plan，每个global rank写自己的trace，末尾barrier确保所有trace写完后再比较。工具不申请SLURM资源，也不通过SSH自动启动其他节点。

## 什么才算通过

每个rank、每个microbatch记录以下信息：

- 完整样本各tensor的shape/dtype/SHA256、prompt及pad mask；坏样本直接报错，不随机换样本。
- VAE latent、带proprio的context和模型输入hash。
- 每次scheduler.add_noise的噪声、timestep和加噪结果hash；覆盖IDM额外视频条件分支。
- Python、NumPy、CPU torch及本rank CUDA的前后随机状态。
- 未四舍五入的loss及loss_video/loss_action、LR、更新边界、epoch/batch编号。
- 默认每次optimizer更新后全部参数hash，以及起始/结束参数hash；记录实际DS配置与主要环境版本。

比较器无容差参数，任何非零差异或缺失rank/step/结束标记都返回非零退出码，并指出第一个不同的字段。loss相同但参数、输入、噪声或随机状态不同也不通过。

```bash
python scripts/alignment/compare_traces.py \
  /shared/review/alignment_run01/uncond/reference/trace \
  /shared/review/alignment_run01/uncond/target/trace \
  --output /shared/review/alignment_run01/uncond/comparison.json
```

默认固定deterministic algorithms、关闭TF32/cuDNN benchmark，并在两侧使用SDPA math后端，以减少不同kernel选择造成的差异。math模式可能增加显存和耗时；不能悄悄减batch、改GAS或开checkpoint后仍宣称同配方通过。

math模式通过后，用新目录加`--attention-backend default`验证实际GPU所选SDPA路径；此模式仍要求确定性、零容差，遇到不支持确定性的kernel会明确失败。之后再按计划检查正常训练环境的吞吐。

参数hash需要GPU到CPU传输，适合短验收，不用于性能测量。`--parameter-hash-every 0`可降低开销，但不再逐次核验参数更新；默认验收保持1。`comparison.json`的mean_local_microbatch_loss是各rank/microbatch的报告均值，不改变训练loss或梯度缩放。

首个不同字段的定位顺序：环境/contract → 初始参数/stats → sample → inputs → noise/RNG → loss → 更新后参数。不要先调学习率或放宽容差来掩盖问题。

## 完成短跑以后

1. 三模型分别通过workers0、workers8、实际SDPA路径的逐步对照。
2. 把短跑步数扩大到覆盖epoch边界；需要时用`--eval-every 2`覆盖训练内验证，确认验证后下一步随机流仍一致。
3. 对完整state恢复做独立实验：相同总max_steps，从同一步state恢复，比较后续样本、LR、参数。当前run_pair专用于初始训练，明确拒绝resume；恢复验收不要改成只加载weights文件。
4. 正式训练回到三个launcher，清除短跑参数，用新RUN_ID/OUTPUT_DIR、MAX_STEPS=null、workers8、task默认eval/save频率。若要声称完整训练逐步一致，需要覆盖相应完整区间的记录，8步通过不是全程通过。
5. 保存配置、代码hash、环境、资产manifest及原始trace；最终报告注明已验证的模型、GPU拓扑、精度、backend和步数。

后续已补充三种模型的独立 LIBERO eager 推理检查、统一评测入口和默认编译配置。完整差异见 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)。此前 uncond 交接包是旧基线方案，**不要向当前仓库重复应用旧补丁**；迁移以新 ALIGNMENT_AGENT_HANDOFF.md 和对应补丁为准。
