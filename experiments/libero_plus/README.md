# LIBERO Plus：三个 1d 模型的评测

统一入口为 `bash experiments/libero_plus/run_eval.sh`，通过 `EVAL_MODE=uncond|joint|idm` 选择模型。使用该模型自己的训练 checkpoint 和训练 stats；三个模型均为 action RoPE=1d、原始 video RoPE、eager 推理。

交给集群 agent 执行时，可直接使用 [CLUSTER_VALIDATION_PROMPT.md](CLUSTER_VALIDATION_PROMPT.md)。

## 环境与输入

本目录是评测接入代码，Plus benchmark、BDDL、初始状态与资产需要另外存在。默认环境路径来自参考入口：

- `LIBERO_PLUS_ROOT=/new_interaction_group/tine/LIBERO-plus-main`
- `LIBERO_PLUS_ASSETS_DIR=/new_interaction_group/common_datasets/dreanzeo_data/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets`
- BDDL/init states 默认位于 `$LIBERO_PLUS_ROOT/libero/libero/bddl_files` 和 `init_files`。

模型、tokenizer、环境路径继承 `scripts/libero_cluster_paths.sh`。Wan 默认目录为 `/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/Wan2.2-TI2V-5B`；tokenizer 默认位于 `/new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P/google/umt5-xxl`。

需要当前 Python 环境能够导入 FastWAM 的依赖和所选 LIBERO Plus 的仿真依赖，并有 `tmux`。沿用集群已经验证的 torch/PPU 环境；本次迁移未修改 `pyproject.toml`，不要为了运行评测直接覆盖平台定制 torch。

默认 `MUJOCO_GL=osmesa`，如使用 EGL，两边需使用同样设置。入口离线加载模型。Plus 使用本地 T5 编码任务文本，训练文本缓存不自动覆盖 Plus 新任务。

必须提供：

- `CKPT`：对应 uncond/joint/IDM 的 **1d 训练后权重**，如 `.../checkpoints/weights/step_XXXXXX.pt`。Wan 基础目录不是评测 checkpoint。
- `DATASET_STATS_PATH`：与该训练对应的 stats。未指定时使用 `DATASET_STATS`，再沿 checkpoint 的前四层父目录寻找 `dataset_stats.json`。显式路径不存在时立即失败。

参考入口默认的 3d_aligned joint checkpoint 不适用于这套 1d 配置。发现训练 config 中明确的模型/RoPE 冲突时会失败；缺失或未解析字段记录为“未核验”，不能视为已通过匹配检查。

## 配置和小规模运行

`eval_config.yaml` 是默认配置，`--config my.yaml` 可覆盖其中的已有 key。已有环境变量优先于 YAML，命令末尾 Hydra 参数再覆盖最终配置。worker 读取 manager 保存的已解析配置快照。

先预览配置，不加载模型、不访问 Plus 目录：

```bash
cd /实际路径/FastWAM
DRY_RUN=1 EVAL_MODE=uncond bash experiments/libero_plus/run_eval.sh
DRY_RUN=1 EVAL_MODE=joint bash experiments/libero_plus/run_eval.sh
DRY_RUN=1 EVAL_MODE=idm bash experiments/libero_plus/run_eval.sh
```

创建覆盖四个 suite 的小任务清单：

```bash
cat > /tmp/libero_plus_smoke.txt <<'EOF'
libero_10,0
libero_goal,0
libero_spatial,0
libero_object,0
EOF
```

配置所选模型的实际权重和 stats，然后运行：

```bash
export CKPT=/实际训练目录/checkpoints/weights/step_XXXXXX.pt
export DATASET_STATS_PATH=/实际训练目录/dataset_stats.json

EVAL_MODE=uncond NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_TASKS_PER_GPU=1 \
TASK_FILE=/tmp/libero_plus_smoke.txt SHARE_BACKUP=false \
bash experiments/libero_plus/run_eval.sh
```

joint/IDM 使用相同命令，只修改 `EVAL_MODE` 并选择各自的 checkpoint/stats。`CUDA_VISIBLE_DEVICES` 的卡数必须与 `NUM_GPUS` 一致，支持不连续的物理卡号。每个 worker 只看到一张卡，`EVALUATION.device` 使用 `cuda` 或 `cuda:0`。

仅检查真实 benchmark 并生成任务清单、元信息，不读取模型权重：

```bash
NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_TASKS_PER_GPU=1 \
TASK_FILE=/tmp/libero_plus_smoke.txt \
bash experiments/libero_plus/run_eval.sh MULTIRUN.create_only=true
```

显式清单保留原顺序，不会被覆盖。`TASK_FILE=auto` 从实际 benchmark 枚举任务。每次使用新的输出目录；非空 `OUTPUT_DIR` 会被拒绝，当前入口不支持断点续跑或混用旧结果。

## 两仓库真实动作与轨迹对照

以下方式使用目标入口统一环境/调度/最终配置，但分别导入两个仓库自己的模型、Plus worker 和共享 rollout 代码。不会改写参考仓库。它验证的是相同配置下的 worker/推理/rollout 行为；参考原始 launcher 与汇总的差异另外通过源码审查和本地测试检查。

先选择一种模型并设好 `CKPT`、`DATASET_STATS_PATH`，保持两次运行的硬件、seed、worker 数、清单和渲染后端完全相同：

```bash
export EVAL_MODE=uncond
export NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_TASKS_PER_GPU=1
export TASK_FILE=/tmp/libero_plus_smoke.txt
export SHARE_BACKUP=false
export FASTWAM_PLUS_TRACE=1

FASTWAM_PLUS_TRACE_REPO=/实际路径/fastwam4d_pp \
OUTPUT_DIR=/实际结果路径/plus_uncond_reference \
bash experiments/libero_plus/run_eval.sh

FASTWAM_PLUS_TRACE_REPO=/实际路径/FastWAM \
OUTPUT_DIR=/实际结果路径/plus_uncond_target \
bash experiments/libero_plus/run_eval.sh

python experiments/libero_plus/compare_traces.py \
  /实际结果路径/plus_uncond_reference/traces \
  /实际结果路径/plus_uncond_target/traces \
  --output /实际结果路径/plus_uncond_comparison.json
```

依次对 joint、IDM 重复，分别使用对应 checkpoint 和新输出目录。比较同一个模型在两个仓库中的结果，不要求三个不同模型之间输出相同。

记录器保存每个 episode 的初始状态、每次 replan 的归一化图像/proprio、seed、模型归一化动作、反归一化动作、gripper 转换后动作，以及每个环境 step 的实际动作、观测状态和 done。观测状态是 proprio，不是完整 MuJoCo 内部状态快照。

比较器先核对 checkpoint/stats/任务清单 hash、最终配置与任务名称，再精确比较记录数组和 episode 结果；发现首次偏离返回非零，不自动放宽容差。缺少完整 episode 不能通过。可先重复同仓库运行，检查平台自身是否可复现。

参考工厂若没有 `compile_training_denoise` 参数，记录器仅在其值为 false 时省略该参数，保留 eager 语义；true 会报错。其他参数不作静默兼容处理。每个 worker 的 `traces/worker*/source.json` 记录真实源码位置，须核对确实来自所选仓库。

记录器只用于小规模验证，包含图像数组的轨迹文件会占用额外空间，正式全量前关闭：

```bash
unset FASTWAM_PLUS_TRACE FASTWAM_PLUS_TRACE_REPO
```

## 全量运行与结果

默认任务清单 `full_10030_lpt.txt` 有 10030 个唯一任务，SHA256 为 `e0e2c9b2291c2b0c92e7c3a5eafe6141ba4ea7caccdbede4906411be235f1e08`。当前 benchmark 的任务 ID 范围和名称会在启动时检查并保存。

小规模对照通过后，先验证每卡一个 worker，再根据资源实测增加到参考默认的每卡三个 worker：

```bash
EVAL_MODE=uncond NUM_GPUS=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
MAX_TASKS_PER_GPU=3 SHARE_BACKUP=true \
TASK_FILE=experiments/libero_plus/full_10030_lpt.txt \
bash experiments/libero_plus/run_eval.sh
```

每个 worker 加载一次模型，然后顺序执行分片任务。默认每批启动一张卡一个新 worker，批间隔 120 秒；可通过 `WORKER_START_WAVE_SIZE`、`WORKER_START_WAVE_INTERVAL` 调整。不存在训练中的跨卡梯度同步。每卡三个 worker 意味着每卡三份模型，必须确认显存和主机内存承受能力。

默认每个任务一个 trial；等待 30 步，每次预测 32 步动作、执行前 10 步后重新规划；diffusion 步数 10。`INFER_INCREMENT_SEED=true` 在每个 episode 内使用 seed、seed+1……，每个新 episode 重新开始。`SKIP_UNUSED_RENDER=true` 只在不保存视频且不生成未来视频时生效。

主要输出：

- `manager_config.yaml`、`worker_config.yaml`：最终配置。
- `eval_meta.json`、`task_manifest.json`、`tasks.txt`：权重/stats/清单 hash、路径、版本、任务定义索引。
- `worker_tasks/`、`worker_launchers/`、`worker_status/`、`task_logs/`：分片、实际命令、退出状态和日志。
- `worker_results/`：逐任务原子更新的 JSON。
- `coverage.json`：是否完整覆盖指定清单，无重复、漏跑或 trial 计数错误。
- `summary.json`、`summary.csv`、`task_success_rates.csv`：总体和任务结果；分类文件可用时增加 category/difficulty CSV。

失败会保留已完成结果与日志，并清理本次创建的 tmux 会话，不关闭其他作业。启动超时或窗口消失会失败；活跃长任务不会因结果 JSON 暂时不更新就被判死。

汇总保留参考口径：`suite_stats.success_rate` 是 0–1 比例；`overall.average_success_rate` 是各 suite 成功率的宏平均百分比。`coverage.json` 未通过时不能把部分任务的成功率报告为全量结果。

默认成功或失败后均尝试 share 备份，备份失败不会隐藏评测退出状态，原结果仍留在输出目录。重复备份到已有同名目录会报出未完成，不覆盖已有备份。

## 本地验证

在有 torch、numpy、Pillow、Hydra、pandas、pytest 的 Python 环境执行：

```bash
FASTWAM_REFERENCE_REPO=/实际路径/fastwam4d_pp \
python -m pytest experiments/libero_plus/test -q

python scripts/alignment/check_inference_parity.py \
  --reference-repo /实际路径/fastwam4d_pp \
  --output /tmp/libero_plus_inference_parity.json
```

测试包含实际 Python 函数、模拟环境、真实子进程 launcher，以及缩小的真实 DiT/MoT/VAE 推理。tmux 的本地调度测试使用替身；真实 tmux、PPU/GPU、T5 权重与 Plus 仿真仍须按上面的集群流程验证。本地通过不能当作真实 benchmark 成功率或训练 loss 对齐的新证据。
