# GPU 16-PPU 真实逐步对照证据（2026-09-12, 节点 ppulingjun033163041110.wa180）

本目录存放 `fastwam-official-align-1d`（目标 FastWAM, commit `dca6f6c26c42d1ee6406281dae6815cc39f9f250`）
与 `fastwam4d_pp`（参考, commit `19d989111193ec2b69ec00493ef250dff531ad03`）在单机 16×PPU-ZW810E
上的真实训练逐步对照结果。比较器 `scripts/alignment/compare_traces.py` 为零容差精确比较：
全部 16 rank、每个 microbatch 的原始 loss 及分量、样本/输入 hash、stats、RNG、noise/timestep、
初始参数、更新边界、LR、逐次更新后的参数 hash 完全一致方为 passed=true。

## 已通过（6/9）

| 阶段 | 配置 | uncond | joint | idm |
|---|---|---|---|---|
| Phase 1 `phase1_math_workers0_r4` | SDPA math, workers=0, 8 updates | ✅ 16 rank / 256 mb / diff=0.0 | ✅ 16 rank / 128 mb / diff=0.0 | ✅ 16 rank / 128 mb / diff=0.0 |
| Phase 2 | SDPA math, workers=8, 8 updates | ✅ diff=0.0（`phase2_uncond_comparison.json`） | ✅ diff=0.0（`phase2_joint_comparison.json`） | ✅ diff=0.0（`phase2_idm_comparison.json`，try4） |
| Phase 3 default SDPA, workers=8 | 进行中（`phase3_*_runner.log`） | — | — | — |

共享环境：torch 2.4.0+ppu1.4.0.oe / accelerate 1.12.0 / deepspeed 0.16.0 / ZeRO-1 bf16，
seed=42，DATASET_STATS 共用同一 JSON（sha256 前缀 c878e8b955209973），
数据 /new-interaction/share/user_folder/jiahao.ljh/data/LIBERO-fastwam，
文本 cache /new-interaction/share/user_folder/jiahao.ljh/data/fastwam-text-emb/libero（40 prompt 全覆盖），
ActionDiT /mnt/new-interaction-p/common/user_folder/shizhen/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt。

## 真实 GPU 验证发现并已修复的问题

1. **目标仓库 joint 工厂缺陷（本次唯一代码修复，见 `target_runtime_fix_joint_compile.diff`）**：
   `configs/task/libero_joint_2cam224_1e-4.yaml` 传 `model.compile_training_denoise=false`，
   但 `create_fastwam_joint` 不接受该参数 → joint 在目标侧实例化即 `TypeError` 崩溃。
   修复为与 `create_fastwam`/`create_fastwam_idm` 相同的 2 行透传（默认 False=eager，无数值影响）。
   证据：`phase1_run_attempt3_joint_factory_bug.log`。
2. **环境缺 boto3**（两 repo pyproject 均声明 1.35.99）：`utils/misc.py` 顶层 import。
   证据：`phase1_run_attempt1_boto3_failure.log`。
3. **accelerate 1.4.0 初始化顺序缺陷**：目标 dataset 用 `PartialState()` 做 main-process 判断，
   在 DeepSpeed launch 下先于 trainer 的 `Accelerator()` 构造会触发 1.4.0 的
   `_from_accelerator` 守卫 ValueError；1.12.0 已改为检查 `AcceleratorState._shared_state`，
   按两 repo 声明版本安装后解决。证据：`phase1_run_attempt2_accelerate_failure.log`。

## 平台已知问题（非两 repo 差异，Phase 2/3 重试的根因）

workers=8 时约 40–60% 的 16 卡 launch 在第一个 microbatch 挂死：某个 rank 的 DataLoader worker
在 fork 后的初始化中卡住（py-spy 显示 `_worker_loop` 停在 `signal_handling._set_worker_signal_handlers()`
/`torch.set_num_threads(1)` 区域，疑为 fork 继承的 OMP 锁竞态），该 rank 数据永不到达，
其余 15 rank 阻塞在 ZeRO-1 梯度 allreduce 直至 NCCL watchdog 600s 超时 abort。
workers=0（Phase 1）100% 复现良好。详见 `pyspy_rank1_stuck_worker.txt`、`pyspy_rank1_stuck_main.txt`、
`pyspy_rank0_allreduce_wait.txt`、`phase2_run_attempt1_nccl_hang.log`。
运维对策：`run_pair_retry.sh` 用 py-spy 探测区分"真挂死"与"NFS 慢速存盘"，真挂死换新目录自动重试。

## 文件清单

- `phase1_{uncond,joint,idm}_comparison.json`：Phase 1（math/workers=0）通过结果
- `phase2_{uncond,joint,idm}_comparison.json`：Phase 2（math/workers=8）通过结果
- `config_parity.json` / `cpu_model_parity.json` / `trace_check.json`：GPU 前检查（6 例配置全等；CPU 小模型 8 例逐步参数差 0.0）
- `assets_initial.json` / `assets_shared_stats.json`：资产与共享 stats 检查
- `env_fixes.json`：boto3 与 accelerate 两个环境修复记录
- `target_runtime_fix_joint_compile.diff`：joint 工厂 2 行修复
- `phase1_run*.log` / `phase2_*.log`：各次真实运行日志（成功与失败定位过程）
- `run_pair_retry.sh` / `run_phase_inner.sh` / `check_gate.py`：本次集群运维脚本
- `pyspy_*.txt`：挂死现场堆栈

## 结论边界

- 以上为 8-update 短跑逐步对照（改变 scheduler 总长，不等价于 10 epoch 训练的前 8 步）；
  全部通过不宣称全程 loss 一致。Phase 3 通过后启动 16 卡 uncond 正式训练。
