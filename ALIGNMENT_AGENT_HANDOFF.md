# FastWAM 三模型对齐：给执行 agent 的完整迁移说明

## 1. 任务与不可混淆的基线

把原始 `git clone --branch official-libero-launchers --single-branch https://github.com/Shizhen-ZHAO/fastwam4d.git FastWAM` 得到的目标源码，变成本交付的训练与评测版本。

- 补丁基线：`acaa9581ff6488e26e1b60ef9d0b21c4230b89af`（Add official LIBERO IDM 16-GPU training launcher）。
- 对照源码：fastwam4d_pp，commit `19d989111193ec2b69ec00493ef250dff531ad03`。参考仓库源码保持不变。
- 模型：FastWAM uncond、joint、标准IDM。三个均action RoPE=1d，video RoPE为原始3d。
- 数据：标准 LIBERO v2 像素输入，双相机水平拼接；不迁移latent cache、4d、corruption或optional-IDM。
- 目标是同环境/资产/数据顺序下的逐microbatch loss及参数更新一致，不能以曲线看起来接近或日志四位小数相同作为通过。
- 本地训练CPU对照和eager推理CPU对照已通过；真实16卡、真实资产的验证尚未执行。不要把本文件理解为已经证明GPU复现。

旧 `fastwam_uncond_review/handoff` 的patch基于更早commit，且有“进一步修正训练语义”的独立建议。**不得把旧patch再叠加到此版本。** 本次唯一完整迁移来源是伴随交接包的 `baseline_to_aligned.patch` 和 `snapshot_manifest.json`。

## 2. 推荐迁移方式：使用完整补丁，再验证每个文件

交接包 `FastWAM_alignment_handoff.zip` 包含：

```text
FastWAM_alignment_handoff/
  ALIGNMENT_AGENT_HANDOFF.md       # 本文的副本
  EVALUATION_ALIGNMENT.md         # 测试代码逐路径审查
  TRAINING_ALIGNMENT.md           # 严格训练对照操作说明
  baseline_to_aligned.patch       # 所有修改文件 + 所有新增文件的完整补丁
  snapshot_manifest.json         # 基线/参考commit、完整代码文件SHA256和可执行位
  verify_snapshot.py             # 检查迁移结果与本次交付的文件逐一一致
  validation.json                # 从干净基线应用并验证的结果
```

无需agent根据文字重新猜实现。先应用补丁，再阅读第3节与文末代码核对。补丁包含已有训练修复、新评测修复、统一脚本、检查工具、文档和本地证据；不是只含本轮增量。

下面在**新的干净clone**中执行。若已有工作区有修改，保留它，另建clone/worktree；不要reset/覆盖用户修改。

```bash
git clone --branch official-libero-launchers --single-branch \
  https://github.com/Shizhen-ZHAO/fastwam4d.git FastWAM
cd FastWAM
git status --short
git switch -c align-libero-three-models acaa9581ff6488e26e1b60ef9d0b21c4230b89af

# HANDOFF_DIR填写解压交接包的绝对路径。
export HANDOFF_DIR=/absolute/path/FastWAM_alignment_handoff
git apply --check "$HANDOFF_DIR/baseline_to_aligned.patch"
git apply --binary "$HANDOFF_DIR/baseline_to_aligned.patch"
python "$HANDOFF_DIR/verify_snapshot.py" --repo "$PWD"
git diff --check
git status --short
```

如果当前branch已经有新commit，先检查日志，使用上述固定commit新建分支。不能在另一基线上机械解决patch失败，再宣称结果与交付一致。验证器会核对全部交付源码文件内容和可执行位；不会修改文件。

另一个交付 `FastWAM_training_aligned.zip` 是可直接上传的完整代码目录快照，不含.git、数据、权重或Python环境。它与补丁重建的源码应一致。不要把zip解压覆盖已有训练输出或参考仓库。

## 3. 必须迁移的生产代码与影响

### 3.1 三个训练入口与公共配置

| 文件 | 原始版本 | 本交付 | 为什么 |
|---|---|---|---|
| scripts/train_libero_uncond_16gpu.sh | 各自重复路径、B1/GAS1/workers2 | 统一5行入口，交给common选择B4/GAS2/workers8 | 复现参考实际配方 |
| scripts/train_libero_joint_16gpu.sh | 各自重复路径、B1/GAS1/workers2 | 相同入口模板，B8/GAS1/workers8 | 同上 |
| scripts/train_libero_idm_16gpu.sh | 各自重复路径、B1/GAS1/workers2 | 相同入口模板，B8/GAS1/workers8 | 标准IDM，不是optional-IDM |
| scripts/train_libero_16gpu_common.sh | 依赖三个入口提供全部变量 | 集中task→B/GAS映射、拓扑、输出、环境；最后附加用户Hydra参数 | 消除三个入口逐渐不一致；保留单机16/双机2×8 |
| scripts/libero_cluster_paths.sh | 不存在 | 新增统一训练/评测路径文件 | 一处配置全部模型 |

三个入口只允许task名不同。**脚本结构统一不等于三个模型batch/GAS强行相同。**

| 模型 | 每卡batch | GAS | GPU数 | 全局batch |
|---|---:|---:|---:|---:|
| uncond | 4 | 2 | 16 | 128 |
| joint | 8 | 1 | 16 | 128 |
| IDM | 8 | 1 | 16 | 128 |

参考joint/IDM脚本中的部分注释过时；按实际变量取值对齐。参考IDM训练实际包含video和action loss，不能按照脚本“仅action”注释删掉video loss。

common保留：Accelerate ZeRO-1、global num_processes=16、standard多机启动、参数数值与rank检查、各节点相同RUN_ID/共享输出、DRY_RUN、末尾Hydra覆盖最高优先级。新增MODEL_ID/TOKENIZER_MODEL_ID/REDIRECT_COMMON_FILES传递。FASTWAM_ENV/CUDA_HOME为空时使用当前环境；有值时加入PATH/LD_LIBRARY_PATH，不再默认绑定旧机器路径。

### 3.2 六份模型/task配置

| 文件 | 必须改的配置 |
|---|---|
| configs/model/fastwam.yaml | action_dit_config下新增action_rope_mode: 1d |
| configs/model/fastwam_joint.yaml | 同上 |
| configs/model/fastwam_idm.yaml | 同上 |
| configs/task/libero_uncond_2cam224_1e-4.yaml | model.mot_checkpoint_mixed_attn=false；model.compile_training_denoise=false；action_scheduler train_shift/infer_shift=5.0；keep_last_ckpts=2 |
| configs/task/libero_joint_2cam224_1e-4.yaml | 同上 |
| configs/task/libero_idm_2cam224_1e-4.yaml | 同上 |

字段位置必须正确：`keep_last_ckpts`是task顶层；`action_scheduler`在model下；RoPE在action_dit_config下。基础model YAML中的旧shift=1仍可服务其他task，本轮三个task显式覆盖为5。不要只改基础model而遗漏task覆盖链。

三个task本身的batch_size与入口最终值可以不同；正式配方以实际脚本解析的配置为准。手工运行 `scripts/train.py task=...` 不带batch/GAS覆盖，不等于本轮16卡脚本。

### 3.3 src/fastwam/models/wan22/action_dit.py

`ActionDiT.__init__`接收 `action_rope_mode: str = "1d"`，只接受1d，保存属性。原始目标本来就是1d公式；这项不是更换公式，而是使配置显式可检查，防止误传参考的3d/4d参数。

不要把参考整个ActionDiT文件复制进来。参考有多种RoPE研究扩展，本轮不需要。

### 3.4 src/fastwam/models/wan22/fastwam.py

替换 `_encode_video_latents` 与 `_encode_input_image_latents_tensor`，恢复参考eager VAE编码路径。原始目标会设置compile/cudagraph并把video batch交给VAE；参考按video逐个编码。本次两函数均调用 `self.vae.encode(..., device=self.device, tiled=...)`，明确转成模型device/dtype，首帧增加时间维。

影响：VAE运算路径、显存/速度及可能的数值舍入变化；它同时影响训练输入和评测首帧编码。保留目标tensor版MoT和推理cache接口，数值对照已覆盖。

### 3.5 src/fastwam/models/wan22/wan_video_vae.py

`WanVideoVAE.encode`恢复逐video的eager循环：每条video增加batch维 → `single_encode` → squeeze → stack。禁止训练验收时一侧使用整batch VAE、另一侧逐条VAE。

保留目标mean/inv_std buffer和scale property；它们并非训练参数/持久checkpoint字段。底层encode/decode按运算dtype/device转换scale。没有为了文本相同而替换整个VAE。

### 3.6 src/fastwam/models/wan22/fastwam_idm.py

增加类默认 `video_cond_noise_prob: float = 0.5`，与参考直接构造的默认语义一致。factory原本已经提供0.5；该修改让直接构造测试也一致。不要改成推理时的corruption sigma。

IDM推理优化保留：先生成video，再cache条件video去噪action；action-only不decode视频。详见评测审查。

### 3.7 src/fastwam/runtime.py

导入 `set_global_seed`，在 `run_training` 的 `instantiate(cfg.model, ...)` **之前**调用 `set_global_seed(int(cfg.get("seed", 42)))`。trainer里原有seed设置继续保留。

影响：随机新层/未完全覆盖参数的初始化在两repo间一致；仅在构造模型后设seed不足以重现初始参数。

### 3.8 src/fastwam/trainer.py

本文件必须完整应用补丁，不能只挑日志或batch行：

1. 导入shutil、DistributedType、GradientAccumulationPlugin。
2. Accelerator改用 `GradientAccumulationPlugin(num_steps=GAS, sync_with_dataloader=False, sync_each_batch=True)`。这会影响数据loader末尾与每batch的累积/同步语义，是数值对齐关键。
3. 非DeepSpeed环境下zero_stage日志判空，避免CPU/DDP检查崩溃。
4. 保存配置 `keep_last_ckpts`；checkpoint根可由FASTWAM_CKPT_DIR覆盖。
5. 训练内evaluate仅捕获写MP4异常；模型推理和数值计算异常仍抛出。
6. 保存weights/full-state完成后，main rank按数值step清理老checkpoint；前后barrier；task默认保留2组。不同run不可共享FASTWAM_CKPT_DIR。
7. DDP通过包装后的 `self.model(sample)` forward；其他类型包括DeepSpeed保留参考training_loss调用方式。
8. DeepSpeed读取engine.get_global_grad_norm；其他后端clip_grad_norm_。**读norm不代表DS实际做了梯度裁剪。**
9. 只在log step收集loss/components/norm；所有rank参与collective，只有main rank输出。grad_norm=None转为日志NaN，防止单侧日志崩溃。
10. scheduler、optimizer.step、zero_grad和global_step按参考顺序保留。终止训练仍会保存最后weights/full-state。

**共有训练问题没有单侧修复：** 参考同样绕过DS engine forward调用training_loss，现有GAS loss缩放/forward hook语义与DS裁剪配置需要另立共同修复基线；本轮先复现这份参考行为。prepare后weights-only加载与full-state恢复不是同一种resume。初始对照固定resume=null。

### 3.9 pyproject.toml

共有训练依赖对齐，修改7项：

```text
av==16.0.1
datasets==3.6.0
deepspeed==0.18.5
numpy==1.26.4
pyarrow==23.0.0
torchcodec==0.5
tqdm==4.66.5
```

torch2.7.1+cu128、torchvision0.22.1+cu128、accelerate1.12.0等已有匹配值不变。30项共有声明匹配；不引入参考仅绘图所需seaborn。依赖声明一致不证明集群实际安装一致，两个repo要使用同一个Python环境。

### 3.10 独立评测修改

| 文件 | 精确修改 | 影响 |
|---|---|---|
| configs/sim_libero.yaml | compile_action_infer从true改false | eager对照，不依赖compile优化 |
| experiments/libero/eval_libero_single.py | 当前repo/src放在sys.path首位 | 避免运行另一个editable安装的fastwam |
| experiments/libero/run_libero_manager.py | 同上 | manager与worker路径约定一致 |
| experiments/libero/libero_utils.py | task_bddl_file转str | 与参考输入类型一致，兼容字符串wrapper |
| scripts/eval_libero.sh | 新文件 | 三模型共用路径与eager入口；single/manager切换 |

模型eager核心、RGB/state处理、gripper变换已对照；调度器、研究扩展与summary不是逐文件相同。不要把“数值路径一致”写成“两个repo所有测试代码完全相同”。完整细节、未修复的共有/调度问题见 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)。

## 4. 新增验证工具：全部随补丁迁移

| 文件 | 用途 | 是否真正运行trainer/模型 |
|---|---|---|
| scripts/alignment/check_configs.py | 实际训练launcher+Hydra；3模型×2拓扑 | 不运行GPU |
| scripts/alignment/check_model_parity.py | CPU小模型loss/梯度/AdamW轨迹精确对照 | 实际training_loss；独立CPU循环，非DS |
| scripts/alignment/_trace.py | 原trainer被动记录；样本、RNG、noise、loss、参数 | 不替换loss或optimizer语义 |
| scripts/alignment/train_with_trace.py | 在指定repo实际runtime/trainer上安装trace | 是；GPU环境由集群提供 |
| scripts/alignment/run_pair.py | 顺序运行reference/target并比较 | 是；--dry-run仅配置解析 |
| scripts/alignment/compare_traces.py | 所有rank、全部microbatch、参数精确比较 | 零容差；缺失/非有限/不完整一律失败 |
| scripts/alignment/check_trace.py | CPU toy trainer验证记录器被动性和比较器 | 不等于DS测试 |
| scripts/alignment/check_inference_parity.py | 24例真实小模型eager推理 | 无真实T5/预训练权重/模拟器 |
| scripts/alignment/check_eval_configs.py | 实际评测脚本+Hydra 6例 | 不运行GPU |
| scripts/alignment/check_eval_pipeline.py | AST、导入、假环境rollout和问题探针 | 不等于真实LIBERO成功率 |
| scripts/alignment/check_assets.py | 文件路径、min/max stats结构、文本缓存覆盖；训练null stats允许自动计算，评测null按checkpoint查找 | 不加载权重，不证明文件语义或所有数据可解码 |
| scripts/alignment/prepare_stats.py | 调用参考repo原始dataset构造流程，单进程计算一次stats供严格对照复用 | 无模型/优化器/视频推理；实际数据计算在集群执行 |
| scripts/alignment/evidence/*.json | 本地检查证据 | 报告明确各自验证范围 |

`TRAINING_16GPU.md`已更新新路径入口；`TRAINING_ALIGNMENT.md`提供完整训练验收；本文记录全部迁移内容；`EVALUATION_ALIGNMENT.md`记录测试代码差异。不要遗漏文档与新增工具，否则另一台机器不能执行同一套验收。

## 5. 集群路径：按参考启动脚本及其加载的配置确定

以下命令在bash执行。路径来源是参考 `scripts/train_libero_offline_16gpu.sh` → `scripts/train_zero1.sh` → `configs/model/fastwam_3d.yaml` / `configs/data/libero_2cam.yaml` → `RobotVideoDataset.__init__`。无需另外猜ActionDiT、stats或环境绝对路径。

| 资产 | 本交付默认 | 来源/行为 |
|---|---|---|
| Wan模型 | /mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/Wan2.2-TI2V-5B | 用户明确提供，保留此覆盖 |
| DATA_ROOT | /new-interaction/share/user_folder/jiahao.ljh/data/LIBERO-fastwam | 用户与参考脚本一致 |
| ActionDiT | REPO_ROOT/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt | 参考脚本检查的仓库相对文件；不是Wan目录的同级文件 |
| 文本缓存 | REPO_ROOT/data/text_embeds_cache/libero | 参考data配置，context_len=128 |
| DATASET_STATS | null | 参考脚本没有指定现成stats；dataset从训练数据计算并写OUTPUT_DIR/dataset_stats.json |
| tokenizer | /new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P/google/umt5-xxl/ | 参考MODEL_BASE + tokenizer_model_id；使用绝对TOKENIZER_MODEL_ID保留这个位置 |
| Python/CUDA | 当前shell的python/accelerate/CUDA | 参考脚本不执行conda activate，也不设置环境绝对路径；参考README环境名称为fastwam |

Wan目录仍按 `MODEL_BASE=/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints`、`MODEL_ID=Wan2.2-TI2V-5B`、`REDIRECT_COMMON_FILES=false` 解析。不能重复拼接Wan目录。

tokenizer使用绝对目录作为model_id，loader的路径拼接会保留这个绝对位置；不会因为Wan权重换了根目录，就去新的Wan目录猜tokenizer。T5和VAE仍从用户指定Wan目录读取原始.pth。普通训练使用文本cache，不加载T5/tokenizer；评测和预计算文本需要它们。

ActionDiT和cache与参考一样按仓库相对位置放置；代码zip不包含大资产。若已在参考文件夹准备好资产，既可保持同样布局放到目标文件夹，也可用ACTION_DIT_CHECKPOINT/TEXT_CACHE环境变量直接指向参考文件夹中的对应路径。

### 普通训练：保持参考的自动stats行为

```bash
export TARGET_REPO=/actual/upload/location/FastWAM
export REFERENCE_REPO=/actual/upload/location/fastwam4d_pp
cd "$TARGET_REPO"
# 使用参考训练环境；如果已经激活，无需再次激活。
conda activate fastwam
source scripts/libero_cluster_paths.sh
mkdir -p /shared/review
python -m pip check
python -m pip freeze > /shared/review/python_packages.txt
nvidia-smi > /shared/review/nvidia_smi.txt
python scripts/alignment/check_assets.py --mode train --output /shared/review/assets_train.json
```

此时DATASET_STATS=null，普通三个训练launcher传 `pretrained_norm_stats=null`，参考数据构造流程会生成本次run的dataset_stats.json。它不是缺少一个必须用户提供的固定文件。已显式提供DATASET_STATS时仍可复用，错误的显式路径会由读取器报错。

### 严格两repo对照：在rank0节点用普通python先计算一次

为避免每次重复扫描数据，以及线程完成顺序造成未使用的mean/std字段舍入差异，严格对照使用同一份统计文件。下面调用**参考repo原始数据构造流程**生成它，不需要用户另找stats路径；不加载Wan/ActionDiT，不运行训练。

```bash
cd "$TARGET_REPO"
python scripts/alignment/prepare_stats.py \
  --reference-repo "$REFERENCE_REPO" \
  --output /shared/review/libero_reference_dataset_stats.json
export DATASET_STATS=/shared/review/libero_reference_dataset_stats.json
python scripts/alignment/check_assets.py --mode train
```

文件已存在时prepare_stats拒绝覆盖；同一数据/processor下直接复用即可。两机训练仅在rank0节点准备一次，两个节点都export同一个共享文件后再启动run_pair。run_pair要求这个已生成的JSON；传null会给出准备说明，不会静默对两边各算一份。三模型共用它，实际stats hash仍由trace严格检查。

文本缓存若还没按参考布局放好，可以用同一个T5/tokenizer生成一次、两边共用：

```bash
python scripts/alignment/check_assets.py --mode text
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TARGET_REPO/src:$TARGET_REPO"
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 \
  "model.model_id=$MODEL_ID" "model.tokenizer_model_id=$TOKENIZER_MODEL_ID" \
  "model.redirect_common_files=$REDIRECT_COMMON_FILES" \
  "data.train.dataset_dirs=[$DATA_ROOT/libero_spatial_no_noops_lerobot,$DATA_ROOT/libero_object_no_noops_lerobot,$DATA_ROOT/libero_goal_no_noops_lerobot,$DATA_ROOT/libero_10_no_noops_lerobot]" \
  "data.train.text_embedding_cache_dir=$TEXT_CACHE"
```

本地没有这些集群挂载，路径配置已按参考来源确定，资产存在性与真实数据计算由上述集群命令检验；本交付不宣称已运行这些数据处理。

## 6. 本地/上传后的无GPU验证

```bash
cd "$TARGET_REPO"
DRY_RUN=1 bash scripts/train_libero_uncond_16gpu.sh
DRY_RUN=1 bash scripts/train_libero_joint_16gpu.sh
DRY_RUN=1 bash scripts/train_libero_idm_16gpu.sh
python scripts/alignment/check_configs.py --reference-repo "$REFERENCE_REPO"
python scripts/alignment/check_model_parity.py --reference-repo "$REFERENCE_REPO"
python scripts/alignment/check_trace.py
python scripts/alignment/check_eval_configs.py --reference-repo "$REFERENCE_REPO"
python scripts/alignment/check_inference_parity.py --reference-repo "$REFERENCE_REPO"
python scripts/alignment/check_eval_pipeline.py --reference-repo "$REFERENCE_REPO"
```

本地已有结果：训练8例、32次AdamW更新、40microbatch，所有loss/components/139个非空梯度tensor和更新参数相同；推理24例全部相同。模型检查使用真实小VAE，不下载5B权重。CPU float32不能替代真实bf16/ZeRO。

`check_eval_pipeline`会报告仍存在的stats fallback、旧结果复用、孤儿锁行为；这些是审查发现的负面证据，**不是宣称这些问题已修复**。首次真实评测走single入口并使用新目录。

## 7. 真实16卡逐步loss验收

先申请同一组独占GPU，同一Python环境、同一资产，顺序跑两repo。不要把两个普通训练日志中的四位小数直接diff。

```bash
cd "$TARGET_REPO"
export NNODES=1 GPUS_PER_NODE=16 NODE_RANK=0
python scripts/alignment/run_pair.py \
  --reference-repo "$REFERENCE_REPO" \
  --output-root /shared/review/align_run01 \
  --variant all --steps 8 --workers 0 --dry-run

# 确认六条实际命令后，去掉--dry-run；使用同一组资源顺序运行。
python scripts/alignment/run_pair.py \
  --reference-repo "$REFERENCE_REPO" \
  --output-root /shared/review/align_run01 \
  --variant all --steps 8 --workers 0
```

reference分别用fastwam_3d/fastwam_joint_ppu/fastwam_idm，并显式action1d、latent_cache=null。target用三个实际launcher。两边实际执行各自runtime/trainer/dataset/model，不是独立另写的训练公式。

检查所有 `uncond|joint|idm/comparison.json`，比较项包括：

- rank覆盖、初始参数、参数顺序、stats hash、实际环境与有效DS配置；
- 每个microbatch原始样本、VAE/context输入、scheduler噪声/timestep/加噪结果、RNG；
- 未四舍五入loss/loss_video/loss_action、LR、update边界；
- 默认每次更新后全部参数hash，以及完整结束记录。

任何非零差异、NaN、缺rank、缺microbatch、缺结束标记均失败。不得改atol、四舍五入、只挑rank0或删掉失败step。先定位第一个不同字段，不先调学习率。

工具默认固定deterministic algorithms、关闭TF32并使用SDPA math。GPU math可能显存更高；如OOM须记录，不应单侧降batch仍称同配方。通过后用**新output-root**依次跑workers8、`--attention-backend default`，最后覆盖epoch边界与训练内evaluate。

两机各8卡：每个节点运行相同run_pair命令，共同设置NNODES=2/GPUS_PER_NODE=8/MASTER_ADDR/MASTER_PORT/RUN_ID/共享output-root；NODE_RANK分别0和1。完整命令见 [TRAINING_ALIGNMENT.md](TRAINING_ALIGNMENT.md)。脚本不负责SLURM资源分配，也不自动SSH到另一节点。

8步验收会改变scheduler总长，因此只是两份同样短实验的一致性，不是完整10epoch训练的前8步。即使save_every=0，trainer结束仍写最终weights与完整state。磁盘要能存下两repo×三模型的输出。

## 8. 短跑通过后正式训练与评测

每次训练一个模型，GPU资源相同、输出目录独立。正式训练清除短跑override，使用MAX_STEPS=null、workers8、默认eval/save频率。不要把对照模式的trace参数hash开销当作吞吐结果。

```bash
cd "$TARGET_REPO"
unset MAX_STEPS BATCH_SIZE GRAD_ACCUM NUM_WORKERS OUTPUT_DIR
RUN_ID=uncond_1d_formal_01 bash scripts/train_libero_uncond_16gpu.sh
# 结束后再跑下一个模型：
RUN_ID=joint_1d_formal_01 bash scripts/train_libero_joint_16gpu.sh
RUN_ID=idm_1d_formal_01 bash scripts/train_libero_idm_16gpu.sh
```

weights `.pt`用于评测；full-state目录用于完整恢复训练。对resume要另做相同总schedule下的中断恢复实验，本轮run_pair专用于初始训练。

评测使用同一个variant的训练weights + 训练config + 对应dataset_stats.json；Wan初始化目录不能填到CKPT。先验证同一权重下两repo单task，再验证完整基准：

```bash
export CKPT=/actual/training/run/checkpoints/weights/step_XXXXXX.pt
export DATASET_STATS=/actual/training/run/dataset_stats.json
python scripts/alignment/check_assets.py --mode eval
EVAL_OUTPUT_DIR=/shared/review/eval_joint_single_01 bash scripts/eval_libero.sh joint
EVAL_MODE=manager NUM_EVAL_GPUS=16 EVAL_OUTPUT_DIR=/shared/review/eval_joint_full_01 \
  bash scripts/eval_libero.sh joint
```

uncond/IDM替换variant与对应checkpoint。参考eval命令与差异边界见 [EVALUATION_ALIGNMENT.md](EVALUATION_ALIGNMENT.md)。真实LIBERO成功率与真实GPU动作仍需执行后记录；不能用本地24例替代。

## 9. agent交付要求

1. 保存verify_snapshot结果、实际commit/source hash、pip freeze和GPU/CUDA信息。
2. 确认三个训练入口除了task名以外结构完全相同；公共配置按第3节的不同batch/GAS展开。
3. 不修改参考源码，不叠加旧uncond建议patch；不引入3d-action/4d-video、latent cache或optional-IDM。
4. 上传完整目标目录及参考目录/可访问的参考checkout；只上传目标repo无法进行两边逐loss验证。
5. ActionDiT/cache采用参考的仓库相对路径，环境沿用当前shell；stats默认自动计算，严格对照先运行prepare_stats一次。保留运行时资产检查，不再要求用户另提供固定stats或环境路径。
6. 报告清楚区分：文件迁移验证、本地CPU检查、真实16卡对照、完整训练、真实LIBERO评测。未运行的项标为未运行。

## 10. 生产代码完整差异

以下附录由基线 `git diff` 自动导出，包含全部生产代码修改及新增公共路径/评测脚本。新增验证工具、文档与证据的完整内容在 `baseline_to_aligned.patch`，应用该文件即可一次迁移，避免复制Markdown时损坏缩进。

```diff
diff --git a/configs/model/fastwam.yaml b/configs/model/fastwam.yaml
index 296f9e1..e8e34d8 100644
--- a/configs/model/fastwam.yaml
+++ b/configs/model/fastwam.yaml
@@ -34,6 +34,7 @@ video_dit_config:
   action_group_causal_mask_mode: "group_diagonal"

 action_dit_config:
+  action_rope_mode: 1d
   action_dim: ${data.train.processor.action_output_dim}
   hidden_dim: 1024
   ffn_dim: 4096
diff --git a/configs/model/fastwam_idm.yaml b/configs/model/fastwam_idm.yaml
index 403c245..502e795 100644
--- a/configs/model/fastwam_idm.yaml
+++ b/configs/model/fastwam_idm.yaml
@@ -35,6 +35,7 @@ video_dit_config:
   action_group_causal_mask_mode: "group_diagonal"

 action_dit_config:
+  action_rope_mode: 1d
   action_dim: ${data.train.processor.action_output_dim}
   hidden_dim: 1024
   ffn_dim: 4096
diff --git a/configs/model/fastwam_joint.yaml b/configs/model/fastwam_joint.yaml
index 426ba15..e5c603d 100644
--- a/configs/model/fastwam_joint.yaml
+++ b/configs/model/fastwam_joint.yaml
@@ -33,6 +33,7 @@ video_dit_config:
   action_group_causal_mask_mode: "group_diagonal"

 action_dit_config:
+  action_rope_mode: 1d
   action_dim: ${data.train.processor.action_output_dim}
   hidden_dim: 1024
   ffn_dim: 4096
diff --git a/configs/sim_libero.yaml b/configs/sim_libero.yaml
index dc44f89..fc30f53 100644
--- a/configs/sim_libero.yaml
+++ b/configs/sim_libero.yaml
@@ -35,7 +35,8 @@ EVALUATION:
   negative_prompt: ""
   rand_device: cpu
   tiled: false
-  compile_action_infer: true
+  # 与参考版 eager 推理对齐；编译加速需要另做数值验收。
+  compile_action_infer: false

   # Optional external inputs
   dataset_stats_path: null
diff --git a/configs/task/libero_idm_2cam224_1e-4.yaml b/configs/task/libero_idm_2cam224_1e-4.yaml
index 376b570..292cb99 100644
--- a/configs/task/libero_idm_2cam224_1e-4.yaml
+++ b/configs/task/libero_idm_2cam224_1e-4.yaml
@@ -11,6 +11,10 @@ num_workers: 8

 model:
   mot_checkpoint_mixed_attn: false
+  compile_training_denoise: false
+  action_scheduler:
+    train_shift: 5.0
+    infer_shift: 5.0

 # scheduler
 lr_scheduler_type: "cosine"
@@ -25,3 +29,6 @@ eval_every: 200
 gradient_accumulation_steps: 1
 weight_decay: 1e-2
 resume: null
+
+# 与参考版一致；仅本任务默认保留最近两组 checkpoint。
+keep_last_ckpts: 2
diff --git a/configs/task/libero_joint_2cam224_1e-4.yaml b/configs/task/libero_joint_2cam224_1e-4.yaml
index c6864ab..6075fe5 100644
--- a/configs/task/libero_joint_2cam224_1e-4.yaml
+++ b/configs/task/libero_joint_2cam224_1e-4.yaml
@@ -11,6 +11,10 @@ num_workers: 8

 model:
   mot_checkpoint_mixed_attn: false
+  compile_training_denoise: false
+  action_scheduler:
+    train_shift: 5.0
+    infer_shift: 5.0

 # scheduler
 lr_scheduler_type: "cosine"
@@ -25,3 +29,6 @@ eval_every: 200
 gradient_accumulation_steps: 1
 weight_decay: 1e-2
 resume: null
+
+# 与参考版一致；仅本任务默认保留最近两组 checkpoint。
+keep_last_ckpts: 2
diff --git a/configs/task/libero_uncond_2cam224_1e-4.yaml b/configs/task/libero_uncond_2cam224_1e-4.yaml
index 37c27d7..fe3e7a6 100644
--- a/configs/task/libero_uncond_2cam224_1e-4.yaml
+++ b/configs/task/libero_uncond_2cam224_1e-4.yaml
@@ -11,6 +11,10 @@ num_workers: 8

 model:
   mot_checkpoint_mixed_attn: false
+  compile_training_denoise: false
+  action_scheduler:
+    train_shift: 5.0
+    infer_shift: 5.0

 # scheduler
 lr_scheduler_type: "cosine"
@@ -25,3 +29,6 @@ eval_every: 200
 gradient_accumulation_steps: 1
 weight_decay: 1e-2
 resume: null
+
+# 只影响本次 uncond 任务；其他任务沿用 trainer 的不清理回退值。
+keep_last_ckpts: 2
diff --git a/experiments/libero/eval_libero_single.py b/experiments/libero/eval_libero_single.py
index 62c8fa8..2a86434 100644
--- a/experiments/libero/eval_libero_single.py
+++ b/experiments/libero/eval_libero_single.py
@@ -23,6 +23,10 @@ from tqdm import tqdm
 project_root = Path(__file__).resolve().parents[2]
 if str(project_root) not in sys.path:
     sys.path.insert(0, str(project_root))
+src_root = project_root / "src"
+if str(src_root) in sys.path:
+    sys.path.remove(str(src_root))
+sys.path.insert(0, str(src_root))

 from experiments.libero.libero_utils import (
     LIBERO_ENV_RESOLUTION,
diff --git a/experiments/libero/libero_utils.py b/experiments/libero/libero_utils.py
index 68edfd8..ea04f85 100644
--- a/experiments/libero/libero_utils.py
+++ b/experiments/libero/libero_utils.py
@@ -24,6 +24,8 @@ def get_libero_env(task, resolution, seed, env_num=1):
         / task.problem_folder
         / task.bddl_file
     )
+    # 与参考版保持相同参数类型，也兼容需要字符串的 LIBERO wrapper。
+    task_bddl_file = str(task_bddl_file)
     env_args = {
         "bddl_file_name": task_bddl_file,
         "camera_heights": resolution,
diff --git a/experiments/libero/run_libero_manager.py b/experiments/libero/run_libero_manager.py
index 7ccd966..dff515c 100644
--- a/experiments/libero/run_libero_manager.py
+++ b/experiments/libero/run_libero_manager.py
@@ -18,6 +18,10 @@ from omegaconf import DictConfig, OmegaConf
 project_root = Path(__file__).resolve().parents[2]
 if str(project_root) not in sys.path:
     sys.path.insert(0, str(project_root))
+src_root = project_root / "src"
+if str(src_root) in sys.path:
+    sys.path.remove(str(src_root))
+sys.path.insert(0, str(src_root))

 from experiments.libero.summarize_results import summarize_results
 from experiments.libero.worker_pool import pending_task_count, read_worker_status, requeue_task
diff --git a/pyproject.toml b/pyproject.toml
index 71a05cd..22ed494 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -9,10 +9,10 @@ description = "FastWAM: standalone Wan2.2-TI2V-5B training/inference package"
 requires-python = ">=3.10"
 dependencies = [
   "accelerate==1.12.0",
-  "av==15.1.0",
+  "av==16.0.1",
   "boto3==1.35.99",
-  "datasets==4.8.5",
-  "deepspeed==0.18.7",
+  "datasets==3.6.0",
+  "deepspeed==0.18.5",
   "einops==0.8.1",
   "gitpython==3.1.45",
   "huggingface-hub==0.29.2",
@@ -21,20 +21,20 @@ dependencies = [
   "imageio-ffmpeg==0.6.0",
   "jsonlines==4.0.0",
   "modelscope==1.34.0",
-  "numpy==2.2.6",
+  "numpy==1.26.4",
   "omegaconf==2.3.0",
   "packaging==25.0",
   "pandas==2.2.3",
   "pillow==12.0.0",
-  "pyarrow==24.0.0",
+  "pyarrow==23.0.0",
   "regex==2025.11.3",
   "rich==14.2.0",
   "safetensors==0.5.3",
   "termcolor==2.5.0",
   "torch==2.7.1+cu128",
-  "torchcodec==0.4.0",
+  "torchcodec==0.5",
   "torchvision==0.22.1+cu128",
-  "tqdm==4.68.3",
+  "tqdm==4.66.5",
   "transformers==4.49.0",
   "typing-extensions==4.15.0",
   "wandb==0.23.1",
diff --git a/scripts/train_libero_16gpu_common.sh b/scripts/train_libero_16gpu_common.sh
index a10084c..1232c10 100644
--- a/scripts/train_libero_16gpu_common.sh
+++ b/scripts/train_libero_16gpu_common.sh
@@ -3,6 +3,20 @@ set -euo pipefail
 # Shared launcher; invoke a variant script, not this file directly.
 TASK="${1:?Use train_libero_joint_16gpu.sh, train_libero_uncond_16gpu.sh or train_libero_idm_16gpu.sh}"
 shift
+source "$(dirname "${BASH_SOURCE[0]}")/libero_cluster_paths.sh"
+case "${TASK}" in
+  libero_uncond_2cam224_1e-4) VARIANT=uncond; DEFAULT_BATCH=4; DEFAULT_ACCUM=2 ;;
+  libero_joint_2cam224_1e-4) VARIANT=joint; DEFAULT_BATCH=8; DEFAULT_ACCUM=1 ;;
+  libero_idm_2cam224_1e-4) VARIANT=idm; DEFAULT_BATCH=8; DEFAULT_ACCUM=1 ;;
+  *) echo "Unsupported LIBERO task: ${TASK}" >&2; exit 2 ;;
+esac
+# 三个入口共用全部设置，仅 task、输出目录、参考配方的 batch/GAS 不同。
+export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/libero_${VARIANT}_16gpu/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
+export RESUME_STATE="${RESUME_STATE:-null}"
+export NNODES="${NNODES:-1}" GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
+export NODE_RANK="${NODE_RANK:-0}" MASTER_PORT="${MASTER_PORT:-29500}"
+export BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH}}" GRAD_ACCUM="${GRAD_ACCUM:-${DEFAULT_ACCUM}}"
+export NUM_WORKERS="${NUM_WORKERS:-8}" MAX_STEPS="${MAX_STEPS:-null}"
 for value in "${NNODES}" "${GPUS_PER_NODE}" "${NODE_RANK}" "${MASTER_PORT}"; do
   [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "Invalid integer: ${value}" >&2; exit 2; }
 done
@@ -18,8 +32,11 @@ else
   RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
 fi
 export RUN_ID CUDA_HOME
-export PATH="${FASTWAM_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
-export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
+if [[ -n "${CUDA_HOME}" ]]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
+if [[ -n "${FASTWAM_ENV}" ]]; then
+  export PATH="${FASTWAM_ENV}/bin:${PATH}"
+  export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
+fi
 export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
 export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}"
 export DIFFSYNTH_SKIP_DOWNLOAD=true
@@ -54,8 +71,12 @@ command=(
   "num_workers=${NUM_WORKERS}"
   "max_steps=${MAX_STEPS}"
   "wandb.name=${TASK}"
-  "$@"
 )
+# 同一套参数可指向参考版平铺权重目录，仍保留末尾 Hydra 参数的最高优先级。
+if [[ -n "${MODEL_ID:-}" ]]; then command+=("model.model_id=${MODEL_ID}"); fi
+if [[ -n "${TOKENIZER_MODEL_ID:-}" ]]; then command+=("model.tokenizer_model_id=${TOKENIZER_MODEL_ID}"); fi
+if [[ -n "${REDIRECT_COMMON_FILES:-}" ]]; then command+=("model.redirect_common_files=${REDIRECT_COMMON_FILES}"); fi
+command+=("$@")
 # Print the command only: no training or GPU access.
 if [[ "${DRY_RUN:-0}" == "1" ]]; then
   printf '%q ' "${command[@]}"
diff --git a/scripts/train_libero_idm_16gpu.sh b/scripts/train_libero_idm_16gpu.sh
index 7447770..314f2e5 100644
--- a/scripts/train_libero_idm_16gpu.sh
+++ b/scripts/train_libero_idm_16gpu.sh
@@ -1,31 +1,5 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
-# Official FastWAM IDM (not optional-IDM); no geometry modifications.
-# Edit paths here when moving to the cluster, or set environment variables.
+# 三个训练入口使用同一份路径、环境、拓扑及启动实现。
 export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
-export FASTWAM_ENV="${FASTWAM_ENV:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_official}"
-export CUDA_HOME="${CUDA_HOME:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_cuda128}"
-export DATA_ROOT="${DATA_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2}"
-export TEXT_CACHE="${TEXT_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"
-export MODEL_BASE="${MODEL_BASE:-/mnt/homes/zhaoshizhen/lf/repos/FastWAM/checkpoints}"
-export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${MODEL_BASE}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
-# Same dataset and processor as joint/unconditional; reuse normalization stats.
-export DATASET_STATS="${DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
-export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/libero_idm_16gpu/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
-# First training: null. Resume only from a compatible IDM full training state.
-export RESUME_STATE="${RESUME_STATE:-null}"
-
-# Single node: defaults to 16 GPUs. Two nodes: set NNODES=2 GPUS_PER_NODE=8,
-# NODE_RANK=0/1, MASTER_ADDR=<rank-0 IP>, same RUN_ID and shared OUTPUT_DIR.
-export NNODES="${NNODES:-1}"
-export GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
-export NODE_RANK="${NODE_RANK:-0}"
-export MASTER_PORT="${MASTER_PORT:-29500}"
-# Conservative memory defaults; effective batch = 16 * BATCH_SIZE * GRAD_ACCUM.
-export BATCH_SIZE="${BATCH_SIZE:-1}"
-export GRAD_ACCUM="${GRAD_ACCUM:-1}"
-export NUM_WORKERS="${NUM_WORKERS:-2}"
-export MAX_STEPS="${MAX_STEPS:-null}"
-
-bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_idm_2cam224_1e-4 "$@"
+exec bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_idm_2cam224_1e-4 "$@"
diff --git a/scripts/train_libero_joint_16gpu.sh b/scripts/train_libero_joint_16gpu.sh
index 2af5a4d..a9c246e 100644
--- a/scripts/train_libero_joint_16gpu.sh
+++ b/scripts/train_libero_joint_16gpu.sh
@@ -1,29 +1,5 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
-# Edit paths here when moving to the cluster, or set environment variables.
+# 三个训练入口使用同一份路径、环境、拓扑及启动实现。
 export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
-export FASTWAM_ENV="${FASTWAM_ENV:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_official}"
-export CUDA_HOME="${CUDA_HOME:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_cuda128}"
-export DATA_ROOT="${DATA_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2}"
-export TEXT_CACHE="${TEXT_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"
-export MODEL_BASE="${MODEL_BASE:-/mnt/homes/zhaoshizhen/lf/repos/FastWAM/checkpoints}"
-export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${MODEL_BASE}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
-# The two variants use the same official LIBERO processor and normalization.
-export DATASET_STATS="${DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
-export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/libero_joint_16gpu/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
-export RESUME_STATE="${RESUME_STATE:-null}"
-
-# Single node: defaults to 16 GPUs. Two nodes: set NNODES=2 GPUS_PER_NODE=8,
-# NODE_RANK=0/1, MASTER_ADDR=<rank-0 IP>, same RUN_ID and shared OUTPUT_DIR.
-export NNODES="${NNODES:-1}"
-export GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
-export NODE_RANK="${NODE_RANK:-0}"
-export MASTER_PORT="${MASTER_PORT:-29500}"
-# Conservative memory defaults; effective batch = 16 * BATCH_SIZE * GRAD_ACCUM.
-export BATCH_SIZE="${BATCH_SIZE:-1}"
-export GRAD_ACCUM="${GRAD_ACCUM:-1}"
-export NUM_WORKERS="${NUM_WORKERS:-2}"
-export MAX_STEPS="${MAX_STEPS:-null}"
-
-bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_joint_2cam224_1e-4 "$@"
+exec bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_joint_2cam224_1e-4 "$@"
diff --git a/scripts/train_libero_uncond_16gpu.sh b/scripts/train_libero_uncond_16gpu.sh
index dcb5984..a3229ad 100644
--- a/scripts/train_libero_uncond_16gpu.sh
+++ b/scripts/train_libero_uncond_16gpu.sh
@@ -1,28 +1,5 @@
 #!/usr/bin/env bash
 set -euo pipefail
-
-# Edit paths here when moving to the cluster, or set environment variables.
+# 三个训练入口使用同一份路径、环境、拓扑及启动实现。
 export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
-export FASTWAM_ENV="${FASTWAM_ENV:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_official}"
-export CUDA_HOME="${CUDA_HOME:-/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam_cuda128}"
-export DATA_ROOT="${DATA_ROOT:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/libero_mujoco3.3.2}"
-export TEXT_CACHE="${TEXT_CACHE:-/mnt/homes/zhaoshizhen/lf/new_repos/4DWAM/FastWAM/data/text_embeds_cache/libero}"
-export MODEL_BASE="${MODEL_BASE:-/mnt/homes/zhaoshizhen/lf/repos/FastWAM/checkpoints}"
-export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${MODEL_BASE}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
-export DATASET_STATS="${DATASET_STATS:-/mnt/homes/zhaoshizhen/checkpoints/fastwam/libero_uncond_2cam224_dataset_stats.json}"
-export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/libero_uncond_16gpu/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
-export RESUME_STATE="${RESUME_STATE:-null}"
-
-# Single node: defaults to 16 GPUs. Two nodes: set NNODES=2 GPUS_PER_NODE=8,
-# NODE_RANK=0/1, MASTER_ADDR=<rank-0 IP>, same RUN_ID and shared OUTPUT_DIR.
-export NNODES="${NNODES:-1}"
-export GPUS_PER_NODE="${GPUS_PER_NODE:-16}"
-export NODE_RANK="${NODE_RANK:-0}"
-export MASTER_PORT="${MASTER_PORT:-29500}"
-# Conservative memory defaults; effective batch = 16 * BATCH_SIZE * GRAD_ACCUM.
-export BATCH_SIZE="${BATCH_SIZE:-1}"
-export GRAD_ACCUM="${GRAD_ACCUM:-1}"
-export NUM_WORKERS="${NUM_WORKERS:-2}"
-export MAX_STEPS="${MAX_STEPS:-null}"
-
-bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_uncond_2cam224_1e-4 "$@"
+exec bash "${REPO_ROOT}/scripts/train_libero_16gpu_common.sh" libero_uncond_2cam224_1e-4 "$@"
diff --git a/src/fastwam/models/wan22/action_dit.py b/src/fastwam/models/wan22/action_dit.py
index 09ab921..f89732d 100644
--- a/src/fastwam/models/wan22/action_dit.py
+++ b/src/fastwam/models/wan22/action_dit.py
@@ -53,8 +53,13 @@ class ActionDiT(nn.Module):
         attn_head_dim: int,
         num_layers: int,
         use_gradient_checkpointing: bool = False,
+        action_rope_mode: str = "1d",
     ):
         super().__init__()
+        # 当前对齐基线只支持 1d；显式配置，防止误用参考版其他 RoPE 模式。
+        if action_rope_mode != "1d":
+            raise ValueError(f"This ActionDiT requires action_rope_mode='1d', got {action_rope_mode!r}")
+        self.action_rope_mode = action_rope_mode
         self.hidden_dim = hidden_dim
         self.action_dim = action_dim
         self.ffn_dim = ffn_dim
diff --git a/src/fastwam/models/wan22/fastwam.py b/src/fastwam/models/wan22/fastwam.py
index f312ea0..35ece3e 100644
--- a/src/fastwam/models/wan22/fastwam.py
+++ b/src/fastwam/models/wan22/fastwam.py
@@ -246,18 +246,14 @@ class FastWAM(torch.nn.Module):

     @torch.no_grad()
     def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
-        if tiled:
-            raise NotImplementedError("Batched VAE encoding does not support tiled encoding.")
-        if not hasattr(self, "_vae_encode_compiled"):
-            self._vae_encode_compiled = torch.compile(
-                self.vae.model.encode,
-                backend="cudagraphs",
-                fullgraph=True,
-            )
-        return self._vae_encode_compiled(
-            video_tensor.to(self.device),
-            self.vae.scale,
-        ).clone()
+        z = self.vae.encode(
+            video_tensor,
+            device=self.device,
+            tiled=tiled,
+            tile_size=tile_size,
+            tile_stride=tile_stride,
+        )
+        return z

     @torch.no_grad()
     def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
@@ -267,10 +263,11 @@ class FastWAM(torch.nn.Module):
             raise ValueError(
                 f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
             )
-        if tiled:
-            raise NotImplementedError("Batched VAE image encoding does not support tiled encoding.")
         image = input_image.to(device=self.device)[0].unsqueeze(1)
-        return self.vae.model.encode(image.unsqueeze(0), self.vae.scale)
+        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
+        if isinstance(z, list):
+            z = z[0].unsqueeze(0)
+        return z

     def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
         video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
diff --git a/src/fastwam/models/wan22/fastwam_idm.py b/src/fastwam/models/wan22/fastwam_idm.py
index bddb385..6b888ac 100644
--- a/src/fastwam/models/wan22/fastwam_idm.py
+++ b/src/fastwam/models/wan22/fastwam_idm.py
@@ -13,7 +13,8 @@ logger = get_logger(__name__)
 class FastWAMIDM(FastWAMJoint):
     """IDM variant with teacher-forcing video conditioning for action denoising."""

-    video_cond_noise_prob: float
+    # 与参考版直接构造及预训练工厂的默认行为一致。
+    video_cond_noise_prob: float = 0.5

     @classmethod
     def from_wan22_pretrained(cls, *, video_cond_noise_prob: float = 0.5, **kwargs):
diff --git a/src/fastwam/models/wan22/wan_video_vae.py b/src/fastwam/models/wan22/wan_video_vae.py
index b2e99cb..1152fa0 100644
--- a/src/fastwam/models/wan22/wan_video_vae.py
+++ b/src/fastwam/models/wan22/wan_video_vae.py
@@ -1219,10 +1219,13 @@ class WanVideoVAE(nn.Module):


     def encode(self, videos, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
-        raise NotImplementedError(
-            "WanVideoVAE.encode() legacy per-video loop is disabled. "
-            "Call `vae.model.encode(batched_video, vae.scale)` for fixed-shape batched encode."
-        )
+        if tiled:
+            raise NotImplementedError("Tiled encoding is not allowed yet.")
+        hidden_states = []
+        for video in videos:
+            hidden_state = self.single_encode(video.unsqueeze(0), device)
+            hidden_states.append(hidden_state.squeeze(0))
+        return torch.stack(hidden_states)


     def decode(self, hidden_states, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
diff --git a/src/fastwam/runtime.py b/src/fastwam/runtime.py
index e521dda..ddd9cee 100644
--- a/src/fastwam/runtime.py
+++ b/src/fastwam/runtime.py
@@ -12,6 +12,7 @@ from einops import repeat
 from omegaconf import OmegaConf

 from .trainer import Wan22Trainer
+from .utils.pytorch_utils import set_global_seed
 from .utils.logging_config import get_logger, setup_logging
 from .utils.video_io import save_mp4
 from .utils import misc
@@ -466,6 +467,8 @@ def run_training(cfg: DictConfig):
     model_device = _resolve_train_device()
     mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
     model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
+    # 在模型初始化前固定随机数；trainer 仍会为数据读取再次设 seed。
+    set_global_seed(int(cfg.get("seed", 42)))
     model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
     train_ds, val_ds = build_datasets(cfg.data)

diff --git a/src/fastwam/trainer.py b/src/fastwam/trainer.py
index 5310837..c3fd7d4 100644
--- a/src/fastwam/trainer.py
+++ b/src/fastwam/trainer.py
@@ -3,6 +3,7 @@ import json
 import inspect
 import os
 import re
+import shutil
 from math import ceil
 from pathlib import Path
 import time
@@ -10,6 +11,7 @@ import time
 import numpy as np
 import torch
 from accelerate import Accelerator
+from accelerate.utils import DistributedType, GradientAccumulationPlugin
 from omegaconf import DictConfig
 from PIL import Image
 from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
@@ -41,6 +43,7 @@ class Wan22Trainer:
         self.max_steps = int(max_steps) if max_steps is not None else None
         self.log_every = int(cfg.log_every)
         self.save_every = int(cfg.save_every)
+        self.keep_last_ckpts = int(cfg.get("keep_last_ckpts", 0))
         self.eval_every = int(cfg.eval_every)
         self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
         self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
@@ -57,15 +60,24 @@ class Wan22Trainer:
         self.wandb_enabled = bool(cfg.wandb.enabled)

         self.accelerator = Accelerator(
-            gradient_accumulation_steps=self.gradient_accumulation_steps,
+            gradient_accumulation_plugin=GradientAccumulationPlugin(
+                num_steps=self.gradient_accumulation_steps,
+                sync_with_dataloader=False,
+                sync_each_batch=True,
+            ),
             mixed_precision=self.mixed_precision,
             step_scheduler_with_optimizer=False,
         )

+        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
+        zero_stage = (
+            ds_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown")
+            if ds_plugin is not None else "n/a"
+        )
         logger.info(
             "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
             self.accelerator.distributed_type,
-            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
+            zero_stage,
             self.accelerator.num_processes,
             self.accelerator.process_index,
             self.mixed_precision,
@@ -106,7 +118,7 @@ class Wan22Trainer:
         self.epoch = 0
         self.batch_in_epoch = 0

-        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
+        self.checkpoint_root = os.environ.get("FASTWAM_CKPT_DIR") or os.path.join(self.output_dir, "checkpoints")
         self.weights_dir = os.path.join(self.checkpoint_root, "weights")
         self.state_dir = os.path.join(self.checkpoint_root, "state")
         self.eval_dir = os.path.join(self.output_dir, "eval")
@@ -546,7 +558,10 @@ class Wan22Trainer:
             self.eval_dir,
             f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
         )
-        save_mp4(stitched_frames, video_path, fps=8)
+        try:
+            save_mp4(stitched_frames, video_path, fps=8)
+        except Exception as exc:
+            logger.warning("Eval video write failed, metrics unaffected: %s", exc)

         local_metrics = torch.tensor(
             [
@@ -619,8 +634,49 @@ class Wan22Trainer:
             self._save_trainer_state(state_path)
         self.accelerator.wait_for_everyone()

+        # 清理只保留最近 N 个，必须在 wait_for_everyone 之后由 main process 执行，
+        # 保证所有 rank 已完成本轮 state 写入，避免误删仍在写的内容。
+        if self.accelerator.is_main_process and self.keep_last_ckpts > 0:
+            self._prune_old_checkpoints()
+        self.accelerator.wait_for_everyone()
+
         return {"weights_path": ckpt_path, "state_path": state_path}

+    def _prune_old_checkpoints(self):
+        """只保留最近 keep_last_ckpts 个 ckpt（weights + state），删除更早的。
+
+        weights 是 checkpoints/weights/step_%06d.pt，state 是
+        checkpoints/state/step_%06d/ 目录；步号零填充，按数字排序即时间顺序。
+        """
+        weights_re = re.compile(r"^step_(\d+)\.pt$")
+        state_re = re.compile(r"^step_(\d+)$")
+
+        def _step_entries(root: str, pattern) -> list[tuple[int, str]]:
+            if not os.path.isdir(root):
+                return []
+            found = []
+            for name in os.listdir(root):
+                m = pattern.fullmatch(name)
+                if m:
+                    found.append((int(m.group(1)), os.path.join(root, name)))
+            found.sort(key=lambda x: x[0])
+            return found
+
+        stale = []
+        for root, pattern, remove in (
+            (self.weights_dir, weights_re, os.remove),
+            (self.state_dir, state_re, shutil.rmtree),
+        ):
+            entries = _step_entries(root, pattern)
+            for _, path in entries[:-self.keep_last_ckpts]:
+                remove(path)
+                stale.append(path)
+        if stale:
+            logger.info(
+                "[ckpt-prune] keep_last=%d 删除 %d 个旧 checkpoint: %s",
+                self.keep_last_ckpts, len(stale), ", ".join(stale),
+            )
+
     def load_training_state(self, state_dir: str):
         self.accelerator.load_state(input_dir=state_dir)
         state_file = Path(state_dir) / "trainer_state.json"
@@ -691,34 +747,45 @@ class Wan22Trainer:
                 continue

             with self.accelerator.accumulate(self.model):
-                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
-
-                with self.accelerator.autocast():
-                    loss, loss_dict = train_model.training_loss(sample)
+                # 对齐 fastwam4d_pp 的现有调用方式；不单侧改变 DS 的累积语义。
+                if self.accelerator.distributed_type == DistributedType.MULTI_GPU:
+                    with self.accelerator.autocast():
+                        loss, loss_dict = self.model(sample)
+                else:
+                    train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)
+                    with self.accelerator.autocast():
+                        loss, loss_dict = train_model.training_loss(sample)
                 self.accelerator.backward(loss)

                 if self.accelerator.sync_gradients:
-                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
+                    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
+                        # 读取 engine 范数不等于启用了裁剪；复现配置保持参考版行为。
+                        grad_norm = self.model.get_global_grad_norm()
+                    else:
+                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                     self.optimizer.step()
                     if not self.accelerator.optimizer_step_was_skipped:
                         self.scheduler.step()
                     self.optimizer.zero_grad(set_to_none=True)
                     self.global_step += 1
-                    global_loss = float(
-                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
-                    )
-                    global_loss_metrics = {}
-                    for key, value in loss_dict.items():
-                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
-                        global_loss_metrics[key] = float(
-                            self.accelerator.gather(metric_tensor).mean().item()
+                    do_log = self.log_every > 0 and self.global_step % self.log_every == 0
+                    if do_log:
+                        global_loss = float(
+                            self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                         )
-                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
-                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())
+                        global_loss_metrics = {}
+                        for key, value in loss_dict.items():
+                            metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
+                            global_loss_metrics[key] = float(
+                                self.accelerator.gather(metric_tensor).mean().item()
+                            )
+                        grad_norm_value = float("nan") if grad_norm is None else float(grad_norm)
+                        grad_norm_tensor = torch.tensor(grad_norm_value, device=loss.device, dtype=torch.float32)
+                        global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

-                    current_lr = float(self.optimizer.param_groups[0]["lr"])
+                        current_lr = float(self.optimizer.param_groups[0]["lr"])

-                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
+                    if do_log and self.accelerator.is_main_process:
                         eta_str, steps_per_sec = self._estimate_eta()
                         description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                             self.epoch,
diff --git a/scripts/libero_cluster_paths.sh b/scripts/libero_cluster_paths.sh
new file mode 100644
index 0000000..b481500
--- /dev/null
+++ b/scripts/libero_cluster_paths.sh
@@ -0,0 +1,20 @@
+#!/usr/bin/env bash
+# 训练与评测共用。可以在这里改路径，也可以在调用前 export 覆盖。
+export REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
+export DATA_ROOT="${DATA_ROOT:-/new-interaction/share/user_folder/jiahao.ljh/data/LIBERO-fastwam}"
+export WAN_CHECKPOINT_DIR="${WAN_CHECKPOINT_DIR:-/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/Wan2.2-TI2V-5B}"
+export MODEL_BASE="${MODEL_BASE:-$(dirname "${WAN_CHECKPOINT_DIR}")}"
+export MODEL_ID="${MODEL_ID:-$(basename "${WAN_CHECKPOINT_DIR}")}"
+# 直接使用 Wan 目录里的 Wan2.2_VAE.pth / models_t5_umt5-xxl-enc-bf16.pth。
+export REDIRECT_COMMON_FILES="${REDIRECT_COMMON_FILES:-false}"
+# 跟随参考 train_libero_offline_16gpu.sh 及其 model/data 配置的仓库相对路径。
+export ACTION_DIT_CHECKPOINT="${ACTION_DIT_CHECKPOINT:-${REPO_ROOT}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
+export TEXT_CACHE="${TEXT_CACHE:-${REPO_ROOT}/data/text_embeds_cache/libero}"
+# 参考脚本不指定预先计算的 stats：训练时自动计算，保存到 OUTPUT_DIR/dataset_stats.json。
+# 严格 run_pair 对照先用 prepare_stats.py 计算一次，再 export DATASET_STATS 供两边共用。
+export DATASET_STATS="${DATASET_STATS:-null}"
+# tokenizer 沿参考模型配置和模型根目录；绝对 model_id 不受上方 Wan 根目录覆盖影响。
+export TOKENIZER_MODEL_ID="${TOKENIZER_MODEL_ID:-/new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P}"
+# 参考启动链直接调用当前 PATH 的 python/accelerate，没有固定 conda/CUDA 路径。
+export FASTWAM_ENV="${FASTWAM_ENV:-}"
+export CUDA_HOME="${CUDA_HOME:-}"
diff --git a/scripts/eval_libero.sh b/scripts/eval_libero.sh
new file mode 100644
index 0000000..37bab66
--- /dev/null
+++ b/scripts/eval_libero.sh
@@ -0,0 +1,37 @@
+#!/usr/bin/env bash
+set -euo pipefail
+# 用法：CKPT=/path/to/weights.pt bash scripts/eval_libero.sh uncond|joint|idm [Hydra参数...]
+VARIANT="${1:?Choose uncond, joint or idm}"
+shift
+case "${VARIANT}" in uncond|joint|idm) ;; *) echo "Unknown variant: ${VARIANT}" >&2; exit 2 ;; esac
+source "$(dirname "${BASH_SOURCE[0]}")/libero_cluster_paths.sh"
+: "${CKPT:?Set CKPT to trained weights/step_XXXXXX.pt, not the Wan initialization directory}"
+if [[ -n "${CUDA_HOME}" ]]; then export PATH="${CUDA_HOME}/bin:${PATH}"; fi
+if [[ -n "${FASTWAM_ENV}" ]]; then
+  export PATH="${FASTWAM_ENV}/bin:${PATH}"
+  export LD_LIBRARY_PATH="${FASTWAM_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
+fi
+export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
+export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE}" DIFFSYNTH_SKIP_DOWNLOAD=true
+export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
+export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
+case "${EVAL_MODE:-single}" in
+  single) ENTRY=experiments/libero/eval_libero_single.py ;;
+  manager) ENTRY=experiments/libero/run_libero_manager.py ;;
+  *) echo 'EVAL_MODE must be single or manager' >&2; exit 2 ;;
+esac
+cd "${REPO_ROOT}"
+command=(python "${ENTRY}" "task=libero_${VARIANT}_2cam224_1e-4"
+  "ckpt=${CKPT}" "model.model_id=${MODEL_ID}"
+  "model.tokenizer_model_id=${TOKENIZER_MODEL_ID}" "model.redirect_common_files=${REDIRECT_COMMON_FILES}"
+  "model.action_dit_config.action_rope_mode=1d"
+  "EVALUATION.dataset_stats_path=${DATASET_STATS}" "EVALUATION.compile_action_infer=false"
+  "EVALUATION.num_trials=${NUM_TRIALS:-50}" "MULTIRUN.num_gpus=${NUM_EVAL_GPUS:-1}"
+  "EVALUATION.output_dir=${EVAL_OUTPUT_DIR:-${REPO_ROOT}/evaluate_results/libero/${VARIANT}/${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}}"
+  "$@")
+printf '%q ' "${command[@]}"
+printf '\n'
+if [[ "${DRY_RUN:-0}" == 1 ]]; then exit 0; fi
+[[ -f "${CKPT}" ]] || { echo "Missing CKPT: ${CKPT}" >&2; exit 2; }
+[[ "${DATASET_STATS}" == null || -f "${DATASET_STATS}" ]] || { echo "Missing DATASET_STATS: ${DATASET_STATS}" >&2; exit 2; }
+exec "${command[@]}"
```
