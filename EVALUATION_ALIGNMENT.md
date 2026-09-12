# LIBERO 三模型评测一致性审查

目标：FastWAM，基线 `acaa9581ff6488e26e1b60ef9d0b21c4230b89af` 加本交付修改。
参考：fastwam4d_pp，`19d989111193ec2b69ec00493ef250dff531ad03`，源码未修改。
范围：标准 LIBERO 的 uncond / joint / IDM，action RoPE=1d，video 默认3d。
不包含 optional-IDM、RoboTwin、LIBERO-plus、corruption、attention 可视化和 RoPE 消融。

## 结论和证据范围

两份评测代码**不是逐文件相同**。将配置、资产、输入和 seed 对齐并关闭目标版推理编译后，三模型的实际小模型 eager 推理测试完全一致。暂未运行真实预训练权重、CUDA/bf16、T5 或 LIBERO 环境，不能据此宣称集群动作或成功率已复现。

本地证据：

- `scripts/alignment/evidence/inference_parity.json`：24例，3模型 × 2接口 × 4组采样参数。实际两层 VideoDiT/ActionDiT/MoT、小 VAE、proprio；比较两 repo 每次 scheduler 更新的速度预测、delta、输入 latent、输出 latent，以及最终动作。`infer_joint` 还比较9帧解码后像素。所有比较 `atol=rtol=0`。
- 参数包括正式配置的10步/seed42/shift=None，以及4步、1步/seed7/shift3、20步/shift5。10步用与 encode_prompt 相同的“padding context 置零、mask 全True”形式；其他案例覆盖含False的 context_mask。没有加载真实 T5。
- `eval_config_parity.json`：实际评测脚本展开命令再解析 Hydra；三模型 × single/manager，模型配置和共有评测参数一致。
- `eval_pipeline.json`：实际函数 AST 对照和假环境执行；证明指定辅助函数相同，以及 wait=0/1/30 时消费的帧、动作、seed 和 done 相同。假环境不等于 MuJoCo 测试。

## 从启动命令到动作的调用链

目标原始 README：

```text
python experiments/libero/run_libero_manager.py task=... ckpt=...
  → configs/sim_libero.yaml + configs/task/... + configs/model/...
  → 每GPU一个 eval_libero_single.py 常驻worker
  → eval_single_process → instantiate(model) → model.load_checkpoint
  → _run_task_to_file → run_single_task → run_single_episode
  → _predict_action_chunk → _obs_to_model_input → infer_action
  → _denormalize_action → gripper符号转换 → env.step
```

参考版也从 `run_libero_manager.py` 进入，但默认调度脚本为 `run_libero_parallel_test.sh`；配置最多每GPU两任务进程。另有 `LIBERO_WORKERS_SCRIPT` 切换常驻 worker 的入口。不能把不同调度方式的耗时直接当成模型性能差异。

参考 joint 示例 `experiments/libero/run_libero_ppu_16gpu_eval.sh`；IDM 示例 `run_libero_idm_3d_16gpu_eval.sh` 中带有旧3d默认。**本交付三个模型都为action1d，不能原样沿用IDM这个3d脚本。** 参考模型组必须显式选：

| 模型 | 目标task | 参考model覆盖 |
|---|---|---|
| uncond | libero_uncond_2cam224_1e-4 | fastwam_3d，action改为1d |
| joint | libero_joint_2cam224_1e-4 | fastwam_joint_ppu，action改为1d |
| IDM | libero_idm_2cam224_1e-4 | fastwam_idm，action改为1d |

新统一入口：`bash scripts/eval_libero.sh uncond|joint|idm`。默认 `EVAL_MODE=single`，便于先验证一个task；全量任务使用 `EVAL_MODE=manager NUM_EVAL_GPUS=16`。训练与测试共用 `scripts/libero_cluster_paths.sh`。

## 实际有效评测参数

| 参数 | 对齐值及影响 |
|---|---|
| mixed_precision | bf16；本地数值测试为CPU float32，二者不能等同 |
| seed | 42；每次replan沿用seed42，默认不按replan递增 |
| eval_num_inference_steps / EVALUATION.num_inference_steps | **10**；Python函数签名的20不是此task实际默认 |
| action/video infer_shift | 5.0；sigma_shift=null不覆盖scheduler |
| action horizon | 32，由data.train.num_frames=33减1 |
| 视频帧数 | 9，由32个action与ratio=4推得 |
| 双相机 | image、wrist_image各224×224，水平拼接为224×448 |
| proprio | eef位置3 + axis-angle3 + gripper_qpos2，共8维 |
| wait/replan | 30步dummy action；每次执行预测chunk的前10步 |
| trial数 | 每task50；标准4套件各10任务，完整基准2000 episodes |
| gripper | 反归一化后 `*2-1`、invert、sign；顺序一致 |
| use_action_ensembler | false |
| visualize_future_video / tiled | false / false |
| rand_device | cpu，随后转到模型device/dtype |
| text_cfg_scale / negative_prompt | 1.0 / 空字符串；不把它们改成其他CFG设置来做本轮验收 |
| compile_action_infer | 本次从目标默认true改为false，与参考eager路径对齐 |

## 逐文件差异、影响和处理

### 1. configs/sim_libero.yaml

原始目标 `compile_action_infer:true`；参考没有此开关、走eager。目标uncond会编译video cache/action denoise，IDM会编译两阶段路径；joint当前接口接受参数但直接 `del compile_action_infer`，并没有启用对应编译。

影响：同一个开关对三个模型的行为不一致；编译可能改变kernel与舍入，不能用于最初的逐位验收。修改为false。编译功能仍保留，之后可以显式开启并另做GPU验收。

参考额外的 `save_video=true`、`skip_unused_render=false`、`infer_increment_seed=false` 与目标默认行为一致。它们不是目标版可直接传入的Hydra字段。参考 `MULTIRUN.max_tasks_per_gpu=2` / `trials_per_shard=0` 属于调度能力，未移植。

### 2. src/fastwam/models/wan22/fastwam.py：uncond

首帧 VAE 编码已经在训练对齐阶段改成参考 eager 路径，因此同时作用于测试。`infer_action` 两边均只缓存观测首帧的video K/V，action每步关注首帧和action；不会生成未来video作为uncond动作条件。

目标版用 `prepare` / `prefill_video_cache_tensor` / `_denoise_action_with_video_cache`，参考版用 `pre_dit` / `prefill_video_cache` / `_predict_action_noise_with_cache` 的字典接口。目标的 `infer_joint` 使用 `_joint_denoise_core`，参考调用 `_predict_joint_noise`。

影响：表示接口和可编译性不同。**没有因为函数不同就整文件替换模型。** 当前action1d、video3d、compilefalse的实际每步数值测试完全一致，保留tensor接口。

`encode_prompt`、`_append_proprio_to_context`、`save_checkpoint`、`load_checkpoint` 的核心语义一致。padding embedding先置零，再把mask设为全True，是两边共有的历史行为，本次不单侧修改。

### 3. src/fastwam/models/wan22/fastwam_joint.py：joint

两边action关注全部video tokens，video不会反向关注action。video/action在同一迭代中共同去噪，每次重新固定首帧。

参考版额外有corruption/attention hook等研究接口；baseline关闭时不参与计算。尤其 `FASTWAM_ACTION_ATTN_MASK_TARGET=obs|future` 在模块导入时读取，会改变joint/IDM的action→video mask，必须unset，不能填字符串none。目标保留tensor core，`compile_action_infer` 目前被忽略。action-only都省略最终video decode。`infer_joint`会解码视频。

影响：默认有效数值对照通过；不能把参考的研究接口参数传给目标版。此文件本次无须修改。

### 4. src/fastwam/models/wan22/fastwam_idm.py：IDM

两边都是两个阶段：先用video expert去噪完整video；再将video作为零timestep条件，建立K/V cache后去噪action。

参考 `infer_action` 调用 `infer_joint` 再只返回action，因此仍支付video decode成本。目标 `infer_action` 直接返回action及video_latents，`infer_joint`再解码video。目标stage1 `_denoise_video` 手工调用prepare/blocks/head；参考调用video_expert.forward。

影响：action-only的速度和返回字典字段不同；同输入/seed下并未在本地发现动作数值差异。24例覆盖IDM两个接口，不需要回退性能优化。训练侧 `.video_cond_noise_prob=0.5` 是训练条件噪声概率，并非测试时自动给video加噪。

### 5. MoT、ActionDiT、VideoDiT、VAE、scheduler及加载器

- `mot.py` / `action_dit.py` / `wan_video_dit.py`：目标tensor路径与参考字典/研究扩展路径不同；action1d下实际推理数值已比较。保留接口，ActionDiT仅已有的1d构造参数与拒绝非1d检查。
- `wan_video_vae.py`：目标mean/inv_std是非持久buffer，参考scale是普通tensor列表。实际encode/decode都会将scale转到运算dtype/device。训练阶段恢复逐video eager encode，测试首帧也使用这一路径。CPU包含非平凡mean/std；真实bf16仍需GPU验收。
- `schedulers/scheduler_continuous.py`：文件完全相同；timesteps、delta及步进公式相同。关键是有效shift配置必须相同。
- `wan_video_text_encoder.py`、`helpers/io.py`、`helpers/state_dict_converters.py`：文件完全相同。
- `helpers/loader.py`：主要差别是参考有progress日志回调，加载、转换、转dtype/device顺序相同。文件发现依赖MODEL_BASE/MODEL_ID/redirect，必须按同一资产布局设置。

### 6. experiments/libero/eval_libero_single.py

相同函数（实际AST检查）：`_center_crop_resize`、`_normalize_proprio`、`_obs_to_model_input`、`_extract_sim_state`、`_denormalize_action`、`_get_num_video_frames`、`_get_max_steps`、`_resolve_dataset_stats_path`、`_load_model_checkpoint`。

其中RGB分别BILINEAR缩放/中心裁剪再拼接，像素映射到[-1,1]；state/action使用同一normalizer。`normalizer.py` 与 `action_state_merger.py` 文件完全相同。processor额外num_image_steps/训练preprocess扩展不改变此评测分支。

主要差别：

1. 原始目标只把repo根放sys.path，src-layout项目可能导入另一份已安装的fastwam。**本次修改：把当前repo/src置于首位。** 已用另一个假安装包验证目标与参考都选本repo。
2. 目标默认支持compile/optional-IDM参数，参考支持corruption和attention hooks；本轮只走三种标准模型的无corruption分支。
3. 参考 `_align_rope_config_with_ckpt` 会从checkpoint旁config.yaml同步video RoPE属性，action mismatch仅告警；目标无自动同步。目标仅支持本轮1d；不移植4d自动改写。**agent必须核对训练config中的模型类型、action1d、video3d，不能加载旧3d-action/4d-video checkpoint。**
4. 参考经task_suite API读取初始状态并临时使torch.load使用weights_only=false；目标按相同任务字段直接读取init_states文件且显式weights_only=false。标准LIBERO原始文件相同；LIBERO-plus自定义初始化逻辑不在保证范围。
5. 参考有render gate、递增seed和attention collector；目标无。默认关闭时，假环境三组rollout的消费帧/动作/seed/done相同。
6. 目标把多个task复用一个模型，参考默认task新进程。seed42+全权重加载下不应依赖task顺序，但必须以真实环境实验验证，不能仅比较runtime。

### 7. experiments/libero/libero_utils.py / action_ensembler.py

libero_utils只发现bddl参数类型差别：参考转str，目标原为Path。本次加 `task_bddl_file = str(task_bddl_file)`，标准环境路径语义不变，也避免字符串检查wrapper的TypeError。

相机翻转、dummy action、quaternion转axis-angle、gripper转换等算法一致；`action_ensembler.py`文件相同。当前ensembler关闭。

### 8. run_libero_manager.py / worker_pool.py / summarize_results.py

目标管理常驻进程和共享文件队列，参考默认shell/tmux动态任务调度，也有独立常驻worker及trial分片。目标不支持参考的trial分片格式，不能复制参考shard task文件给目标。目标manager本次只加src优先路径。

目标summary逐结果文件累加，参考会先合并trial shards且输出更多分类统计。在标准无分片、每task恰好一个结果文件时，suite成功率的基本公式一致；二者元数据、耗时、文件字段不必相同。完整基准的4套件均应恰好10task×50trial，不能用缺任务的均值冒充全量结果。

以下问题已有探针证据，**本轮未重写调度器修复**：

- 目标按结果文件存在判断完成，未检查旧结果对应的checkpoint/trial数。每次用新 EVAL_OUTPUT_DIR，禁止复用旧结果目录。
- 目标目录锁的持有进程若异常退出，锁目录可能遗留并使队列等待。本地进程崩溃探针复现。严格首次验收默认用single入口；manager遇到挂起应停止该次实验、排查worker/锁，再用新目录重跑，不能写成成功完成。
- 两边 `_resolve_dataset_stats_path` 对不存在的显式stats路径会尝试checkpoint上级的stats；新eval脚本会检查显式DATASET_STATS文件；默认null时与参考一样从checkpoint上级查找训练生成的dataset_stats.json。check_assets --mode eval也会检查这个查找结果。
- 两边mot加载使用strict=false，缺失proprio会保留初始化值；两边都是共有行为。采用这次训练产生的完整weights与对应config/stats，不拿其他结构的checkpoint做验收。基于相同success rate不能排除漏加载。

## 本次实际修改的评测相关代码

| 文件 | 修改 |
|---|---|
| configs/sim_libero.yaml | compile_action_infer: true → false |
| experiments/libero/eval_libero_single.py | 当前repo/src优先导入 |
| experiments/libero/run_libero_manager.py | 同上 |
| experiments/libero/libero_utils.py | bddl路径转str |
| scripts/eval_libero.sh | 新增统一3模型评测入口，默认single/eager，共用资产路径 |
| scripts/libero_cluster_paths.sh | 新增共同路径配置，训练和评测共用 |
| scripts/alignment/check_inference_parity.py | 24例实际小模型推理数值对照 |
| scripts/alignment/check_eval_configs.py | 6例真实Hydra评测配置对照 |
| scripts/alignment/check_eval_pipeline.py | AST与假环境rollout、导入及已知风险探针 |
| scripts/alignment/check_assets.py | 集群资产路径、stats基本形状与prompt缓存覆盖检查 |

完整从原始clone迁移的补丁、所有训练变更和验证命令见 [ALIGNMENT_AGENT_HANDOFF.md](ALIGNMENT_AGENT_HANDOFF.md)。

## 集群评测顺序

先完成训练短跑的逐loss/参数对照，再选取相同训练step、相同variant的weights；对照“代码实现”时可以先让两边加载**同一个weights文件**，避免把不同训练产物带入推理比较。

```bash
cd "$TARGET_REPO"
source scripts/libero_cluster_paths.sh
export CKPT=/actual/run/checkpoints/weights/step_000008.pt
export DATASET_STATS=null  # 按checkpoint位置自动查找本次训练生成的stats
python scripts/alignment/check_assets.py --mode eval

# 一个task先跑两次：目标与参考同一个Python/GPU/模型权重/环境资产。
export CUDA_VISIBLE_DEVICES=0
EVAL_OUTPUT_DIR=/shared/review/eval_uncond_target_01 \
bash scripts/eval_libero.sh uncond EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0
```

参考侧手工命令（使用同一个当前Python环境；先设置公共路径）：

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_BASE"
export DIFFSYNTH_SKIP_DOWNLOAD=true HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FASTWAM_ACTION_ROPE_MODE=1d FASTWAM_CORRUPT_TARGET=none FASTWAM_CORRUPT_SIGMA=0
export CROSS_ROPE_DEBUG_INTERVAL=0
unset FASTWAM_COLLECT_ATTENTION_ENTROPY FASTWAM_COLLECT_FULL_HEATMAP
unset FASTWAM_ACTION_ATTN_MASK_TARGET
export PYTHONPATH="$REFERENCE_REPO/src:$REFERENCE_REPO"
cd "$REFERENCE_REPO"
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 model=fastwam_3d \
  model.action_dit_config.action_rope_mode=1d \
  "model.model_id=$MODEL_ID" "model.tokenizer_model_id=$TOKENIZER_MODEL_ID" \
  "model.redirect_common_files=$REDIRECT_COMMON_FILES" \
  "ckpt=$CKPT" "EVALUATION.dataset_stats_path=$DATASET_STATS" \
  EVALUATION.output_dir=/shared/review/eval_uncond_reference_01 \
  seed=42 EVALUATION.num_trials=50 EVALUATION.num_inference_steps=10 \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 \
  EVALUATION.infer_increment_seed=false EVALUATION.skip_unused_render=false
```

joint改task为libero_joint_2cam224_1e-4、参考model为fastwam_joint_ppu；IDM分别改libero_idm_2cam224_1e-4和fastwam_idm。每个实验新目录、使用对应模型weights。原始基准资产、LIBERO代码版本、MuJoCo/robosuite版本和渲染backend也须相同。

对比每episode成功/失败列表，不比较耗时、GPU编号和输出目录字符串。若要声称“真实GPU逐步动作完全相同”，还须记录并比较每次replan输入/输出及环境step轨迹；本交付的训练trace不能代替这项测试，当前提供的推理逐步数值证据只针对本地小模型。

完成单task/全部模型后，再用manager跑完整4套件；检查无缺任务、无重复结果、无失败worker。参考调度器不要求与目标逐文件相同，但这一步的成功率仍须实测。
