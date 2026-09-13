# LIBERO Plus 迁移：实际修改与验证

日期：2026-09-13。修改基线：`FastWAM` 的 `dca6f6c26c42d1ee6406281dae6815cc39f9f250`。
参考源码：`fastwam4d_pp` 的 `19d989111193ec2b69ec00493ef250dff531ad03`。

发布分支为 `fastwam-official-align-1d-libero-plus`，仓库为 `https://code.alipay.com/new_interaction_group/WFM.git`。发布以 `983c595037ba58e931b703e7bb2a73cc12a25fbd` 为父提交，保留此前 joint 修复与 16 卡对齐记录；开发时的源码改动基线仍为上面的 `dca6f6c`。参考仓库保持不变。旧的训练对齐交付文档和树 hash 对应之前版本，不包含本次新增 Plus 文件。

## 已修改的原有文件

| 文件 | 实际修改 | 行为影响 |
| --- | --- | --- |
| `.gitignore` | 为 `experiments/libero_plus/full_10030_lpt.txt` 增加例外 | Git 会包含全量任务清单，避免上传后缺文件 |
| `src/fastwam/runtime.py` | `create_fastwam_joint` 新增 `compile_training_denoise: bool = False`，传给 `from_wan22_pretrained` | 修复 joint 配置实例化；默认 false 保持 eager |
| `experiments/libero/eval_libero_single.py` | `_predict_action_chunk` 接收可选 `infer_seed`，未给定时回退到原 cfg.seed | Plus 可按次递增 seed；普通评测默认仍使用固定 seed |
| 同上 | 新增 `_resolve_infer_seed`、`_SimRenderGate`、`_maybe_install_render_gate`；`run_single_episode` 接入递增 seed 和渲染门控，finally 恢复 render | 对齐参考 Plus 的随机数和渲染行为；关闭开关时保持原流程 |
| 同上 | `run_single_task` 按 `save_video` 判断视频写出，finally 关闭环境 | Plus 不再无条件保存视频；普通评测未设置此项时默认仍保存 |

共享评测文件的 AST 检查确认：原有函数中只有 `_predict_action_chunk`、`run_single_episode`、`run_single_task` 改变，另新增上表三个助手。图像处理、proprio 处理、动作反归一化、最大步数、权重加载等原有函数保持不变。循环的 try/finally 引入了缩进变化，因此文本 diff 行数多于实际逻辑修改数。

joint 两行补丁为：

```python
# create_fastwam_joint 的参数列表
compile_training_denoise: bool = False,

# FastWAMJoint.from_wan22_pretrained(...) 的参数
compile_training_denoise=bool(compile_training_denoise),
```

## 新增文件

以下路径除首行外均位于 `experiments/libero_plus/`：

| 文件 | 用途与参考差异 |
| --- | --- |
| `configs/sim_libero_plus.yaml` | 继承目标普通 LIBERO 配置，设置 Plus 默认值；关闭 Hydra 共用日志写出 |
| `eval_config.yaml` | 三模型统一配置，默认 uncond/1d，不绑定 3d checkpoint；模型路径复用现有集群配置 |
| `load_eval_config.py` | 从参考移植 YAML→环境变量加载器，保留覆盖优先级和未知 key 检查 |
| `run_eval.sh` | 统一 shell 入口、离线环境、路径覆盖、三模型 task 映射；DRY_RUN 预览最终 Hydra 配置 |
| `run_libero_plus_manager.py` | 核验实际 Plus 包/路径，复用显式任务清单、支持 auto/create_only，保存元信息与最终 worker 配置快照 |
| `run_libero_plus_parallel_workers.sh` | 保留 shell 调度入口，转交 Python 监督进程 |
| `parallel_workers.py` | 保持参考的轮转分片、多 worker/卡、常驻模型和分批启动；直接用 tmux 启动脚本，取代 send-keys；检测异常退出并只清理本次会话 |
| `eval_libero_plus_worker.py` | 从参考 worker 移植，模型加载一次；接入目标的配置/初始状态助手，保留结果 JSON 格式与逐任务原子写入 |
| `eval_utils.py` | 1d/eager 配置校验、训练 checkpoint 配置核验、严格 stats 选择、模型资产预检、Plus 初始状态兼容 |
| `task_utils.py` | 任务格式/唯一性、GPU 映射、单 worker 和全量任务/episode 完整性检查 |
| `full_10030_lpt.txt` | 逐字节复制参考清单；10030 个唯一任务，顺序保持 |
| `summarize_results.py` | 移植参考汇总实现，保留分类与宏平均口径，增加完整性检查；普通 LIBERO 原汇总文件不变 |
| `trace_worker.py` | 小规模真实 worker 轨迹记录；可导入参考仓库执行，不改参考源码 |
| `compare_traces.py` | 核对运行输入、完整 episode 覆盖，精确比较输入/动作/观测状态/成功结果 |
| `test/test_libero_plus.py` | 配置、预检、分片、常驻 worker、共享 rollout、视频开关、参考汇总对照 |
| `test/test_worker_execution.py` | 真实 launcher 子进程、监督进程失败处理、joint 工厂、记录器和比较器验证 |
| `README.md` | 路径、三个模型启动方式、小规模两仓库对照、全量运行说明 |
| `CHANGES.md` | 本文件 |
| `CLUSTER_VALIDATION_PROMPT.md` | 可直接交给集群 agent 的分阶段验证任务 |
| `validation.json` | 本次验证结果和源文件 SHA256 清单 |

调度使用 Python 保存已解析命令与配置，避免 shell 长命令转义和 tmux 交互输入竞争。没有改变任务分配公式：第 i 行分配给 `worker i % (NUM_GPUS * MAX_TASKS_PER_GPU)`；worker 的物理 GPU 为 `GPU_ARRAY[worker_id % NUM_GPUS]`。当前不实现动态工作窃取或断点恢复，防止改变参考的 worker seed/任务顺序。

原参考的 3d/4d 实验矩阵、corruption 和 attention/phase 消融脚本未纳入本次三个 1d 模型评测范围。

## 本次验证结果

- **34 项本地功能测试全部通过。** 包含模拟 benchmark 的完整 create_only 入口、三种模型的最终配置和 worker 快照、常驻 worker 模型仅加载一次、10030 任务的轮转覆盖、异常状态与旧输出保护。
- 参考与目标的实际 `run_single_episode` 函数，在等待步数 0/1/30、seed 固定/递增、渲染门控开/关的 12 组组合中，模型消费的画面、seed、实际动作、done 和渲染次数相同。环境和模型预测由测试替身提供。
- 汇总对照包含不同任务数的 suite 与 category/difficulty 映射；两边生成的 `summary.json`、`summary.csv`、`task_success_rates.csv`、`category_success_rates.csv`、`difficulty_success_rates.csv` 五份文件逐字节相同。
- **24 组 CPU 模型推理对照全部零差异。** 执行实际的缩小 DiT/MoT/VAE，覆盖三模型的 `infer_action`、`infer_joint`，多 seed、推理步数和 shift；动作、scheduler 中间张量相同，解码视频路径也逐帧相同。
- 新 Python 文件通过 Python 3.10 语法解析；shell 通过 `bash -n`；原有文件的修改通过 `git diff --check`。
- 全量清单 SHA256：`e0e2c9b2291c2b0c92e7c3a5eafe6141ba4ea7caccdbede4906411be235f1e08`。

本地测试环境为 Mac、Python 3.12、torch 2.7.1 CPU。具体版本和对照结果记录在 `validation.json`。测试中 tmux 使用替身，执行的是实际生成的 shell launcher 与本地子进程。没有安装或运行真实 Plus 仿真，也没有加载真实 5B/T5 训练评测权重。

## 集群剩余验收

按 README 顺序执行：配置预览 → 真实 Plus create_only → 单卡小任务 → 同模型两仓库轨迹精确比较 → 16 卡并发验证 → 全量 10030 任务。

每个模型需提供自己的 1d 训练 checkpoint 和对应 stats。若默认 backend 下同仓库重复运行也不一致，应先定位平台随机性；不要单侧修改 seed/backend 或把容差比较描述成零差异。

训练 Phase 1/2 的 loss 对齐与本次评测是不同的证据。此前训练 Phase 3 的 backward 确定性限制未在此改动中消除；本次未新增真实 16 卡训练 loss 或 LIBERO Plus 成功率结果。
