# 交给集群 agent 的 LIBERO Plus 验证任务

请实际验证新接入的 LIBERO Plus 评测代码，产出可追溯的实验报告。目标为同一个 uncond/joint/IDM 模型在两个仓库中，用相同 1d 训练 checkpoint、stats、任务、环境和随机种子得到一致的输入、动作和 rollout 结果。三个不同模型之间不要求结果相同。

## 1. 准备独立工作目录并固定源码

目标仓库：`https://code.alipay.com/new_interaction_group/WFM.git`，分支 `fastwam-official-align-1d-libero-plus`。

参考仓库：`https://code.alipay.com/ljh488565/Fast-WAM-new.git`，分支 `jiahao-dev`，本地验证使用的参考提交为 `19d989111193ec2b69ec00493ef250dff531ad03`。

使用独立目录，不切换或覆盖正在训练的仓库，也不停止已有训练进程。先检查可用计算资源；资源不足时继续做只读预检并报告缺口。

```bash
git clone --branch fastwam-official-align-1d-libero-plus --single-branch \
  https://code.alipay.com/new_interaction_group/WFM.git FastWAM_libero_plus

git clone --branch jiahao-dev --single-branch \
  https://code.alipay.com/ljh488565/Fast-WAM-new.git fastwam4d_pp_plus_reference

git -C fastwam4d_pp_plus_reference checkout --detach 19d989111193ec2b69ec00493ef250dff531ad03
git -C FastWAM_libero_plus log -1 --oneline
git -C fastwam4d_pp_plus_reference log -1 --oneline
```

记录两边实际提交与工作区状态。如果参考提交不可取得，不把其他提交冒充该版本；报告后再判断能否依据实际版本重新审查。

先读取目标仓库的 `experiments/libero_plus/README.md`、`CHANGES.md`、`validation.json`。目标继承 `983c595` 的训练对齐基线；已有 34 项本地功能测试与 24 组缩小模型 CPU 推理对照通过，这些不代表真实集群评测已经通过。

在目标仓库根目录核验交付文件：

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
report = json.loads(Path('experiments/libero_plus/validation.json').read_text())
for name, expected in report['sha256_except_this_report'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('交付文件 SHA256 检查通过')
PY
```

## 2. 确认环境与三个模型的训练权重

沿用集群已验证的 PPU/torch 环境；记录 Python、torch/PPU、Accelerate、DeepSpeed、MuJoCo、robosuite 和 LIBERO Plus 版本，不直接用仓库依赖声明覆盖平台定制 torch。

路径默认如下，先核验实际挂载和读取权限：

- Wan：`/mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/Wan2.2-TI2V-5B`
- tokenizer：`/new_interaction_group/common_models/Wan-AI__Wan2.1-I2V-14B-480P/google/umt5-xxl`
- Plus 根目录：`/new_interaction_group/tine/LIBERO-plus-main`
- Plus assets：`/new_interaction_group/common_datasets/dreanzeo_data/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets`
- BDDL/init states：Plus 根下 `libero/libero/bddl_files` 和 `libero/libero/init_files`。

查找现有训练输出中的 uncond、joint、IDM 的 **1d 训练后 checkpoint**，按训练 `config.yaml` 确认模型类型和 action RoPE，并绑定该训练使用的 stats。Wan 初始化目录不是评测 checkpoint；参考原入口默认的 3d_aligned joint 权重不适用。

此前对齐使用的 stats 候选为 `/ossfs/workspace/alignment_results/20260912_001614/libero_reference_dataset_stats.json`，SHA256 为 `c878e8b955209973638c03c90d1e47b5c910a647be25bfd160f1ee02611ee589`；仅在确认所选训练权重使用这份 stats 后使用它。

每个模型建立权重表：模型类型、checkpoint 路径/hash、训练 config 路径、训练 action RoPE、stats 路径/hash。缺失某个模型的权重时明确列为未验证，继续验证具备条件的模型；不能借用另一个模型或 3d 权重充当通过证据。

## 3. 配置、环境与单卡预检

在目标仓库运行三个模式的 DRY_RUN，确认最终配置为 1d、eager、正确的模型工厂和路径。默认推理参数应保持：seed=42、等待 30 步、预测 32 步动作、执行 10 步后重新规划、diffusion 推理 10 步、INFER_INCREMENT_SEED=true、SKIP_UNUSED_RENDER=true、SAVE_VIDEO=false。

```bash
DRY_RUN=1 EVAL_MODE=uncond bash experiments/libero_plus/run_eval.sh
DRY_RUN=1 EVAL_MODE=joint bash experiments/libero_plus/run_eval.sh
DRY_RUN=1 EVAL_MODE=idm bash experiments/libero_plus/run_eval.sh

cat > /tmp/libero_plus_smoke.txt <<'EOF'
libero_10,0
libero_goal,0
libero_spatial,0
libero_object,0
EOF

NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_TASKS_PER_GPU=1 \
TASK_FILE=/tmp/libero_plus_smoke.txt SHARE_BACKUP=false \
bash experiments/libero_plus/run_eval.sh MULTIRUN.create_only=true
```

GPU 编号应替换为实际空闲卡；`NUM_GPUS` 必须等于可见卡列表长度。确认导入的 libero 确实来自 Plus 根目录，保存实际任务名称/BDDL 索引。然后按 README 用一个有效模型 checkpoint 完成这四个任务，核验 `coverage.json` 和进程退出状态。

## 4. 同模型、两仓库的动作与轨迹对照

使用 README 的 `FASTWAM_PLUS_TRACE=1` 方式。目标入口统一环境和最终配置，`FASTWAM_PLUS_TRACE_REPO` 分别指向参考目录和目标目录，实际加载各自的模型、Plus worker 与共享 rollout 代码。

每次运行设置新 OUTPUT_DIR，保持所选 GPU、checkpoint、stats、任务顺序、worker 数、seed、dtype、backend、渲染设置完全相同。先重复同一仓库的小规模运行，检查平台自身的可重复性，再比较两个仓库。

```bash
# 以下路径替换为本次实际路径。
export EVAL_MODE=uncond
export CKPT=/实际训练目录/checkpoints/weights/step_XXXXXX.pt
export DATASET_STATS_PATH=/实际训练使用的/dataset_stats.json
export NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 MAX_TASKS_PER_GPU=1
export TASK_FILE=/tmp/libero_plus_smoke.txt SHARE_BACKUP=false
export FASTWAM_PLUS_TRACE=1

FASTWAM_PLUS_TRACE_REPO=/实际路径/fastwam4d_pp_plus_reference \
OUTPUT_DIR=/本次结果目录/uncond_reference \
bash experiments/libero_plus/run_eval.sh

FASTWAM_PLUS_TRACE_REPO=/实际路径/FastWAM_libero_plus \
OUTPUT_DIR=/本次结果目录/uncond_target \
bash experiments/libero_plus/run_eval.sh

python experiments/libero_plus/compare_traces.py \
  /本次结果目录/uncond_reference/traces \
  /本次结果目录/uncond_target/traces \
  --output /本次结果目录/uncond_comparison.json
```

joint/IDM 使用对应 checkpoint 和独立结果目录重复。核验 `traces/worker*/source.json` 的源码位置，确认两次确实导入不同仓库。参考工厂不接受 compile 参数时，记录器只省略值为 false 的 `compile_training_denoise`，不改变 eager 计算。

比较内容包括：初始状态、每次 replan 的归一化图像/proprio、seed、归一化动作、反归一化动作、gripper 转换后动作、实际环境 step 动作、观测状态、done 和 episode 成功结果。记录的观测状态是 proprio，不是完整 MuJoCo 内部状态快照。

通过标准是输入匹配、完整性检查通过、比较器精确一致。仅成功率相同不算推理实现对齐。出现差异时记录第一个偏离的 task/trial/replan/字段与差值；先区分配置、输入、模型和仿真差异，不单侧调整 seed/backend，也不把近似相同写成零差异。

## 5. 渲染与并发验证

对已通过的模型，在同仓库做 SKIP_UNUSED_RENDER 开/关的 A/B，保持 SAVE_VIDEO=false，检查真正进入模型的图像和动作是否不变。该 A/B 的配置有意不同，不能直接通过要求配置完全相同的 CLI 元信息检查；可调用 `compare_traces.compare` 比较数组和 episode，并单独记录唯一的配置差异。

单卡对照通过后，资源允许时增加每卡 worker 数，再验证 16 卡分片和状态汇总。选择至少 48 个唯一任务的小清单以覆盖 16×3 个 worker；避免拿只有四个任务的清单声称所有 48 个 worker 都已验证。检查任务无重复/漏跑、显存/主机内存峰值、worker 异常退出和汇总口径。

全量扩展时关闭轨迹记录，使用仓库中的 `full_10030_lpt.txt`，核验 10030 个唯一任务和完整 episode 覆盖。全量任务运行与小规模对齐分别报告，未完成全量时明确标记未完成。

## 6. 输出报告

请将以下内容保存到一个新的结果目录，并向用户报告：

1. 两边源码版本、工作区状态、环境版本、GPU/PPU 与渲染后端。
2. 三模型各自的 checkpoint/stats/训练配置核验表。
3. 预检、单仓库重复性、两仓库动作对照、渲染 A/B、并发、全量各阶段的状态。
4. 配置快照、输入 hash、source.json、coverage.json、comparison JSON、worker 日志及结果目录。
5. 若失败，具体命令、退出码、第一处差异或异常、定位证据；修改代码后另记 diff 和新版本，再重新验证受影响项。

训练 Phase 1/2 的 loss 对齐不能替代此次评测验证；此前训练 Phase 3 的 backward 确定性限制也不直接说明 Plus 前向推理会失败。请根据本次真实执行结果下结论。
