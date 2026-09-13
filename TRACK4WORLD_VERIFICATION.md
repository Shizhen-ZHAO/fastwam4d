# FastWAM_3_track 本机验证（2026-09-13）

基线 FastWAM_3：`31770396e672bfa283f4283e59775057ab69ec89`。
本目录独立分支 `fastwam-3-track`；本次验收完成时尚未提交/推送，原 FastWAM_3 工作区保持干净。
没有停止或替换其他 GPU 任务。检查结束后本次 GPU 进程全部退出。

## 已执行

- `geometry_tests` 与 `experiments/libero_plus/test`：72 passed、2 skipped。
  两项跳过均因本机未提供 `FASTWAM_REFERENCE_REPO=fastwam4d_pp`，不是测试通过。
- 新增/修改代码语法检查：24 个 Python、5 个 shell 通过。
- 全部899个基线文件内容比较：仅四个必要接线文件不同；原核心模型、runtime、优化器/
  训练循环/采样器、原训练脚本、10030任务清单与历史报告保留。
- 小模型零 gate：基线 loss、梯度、首次参数更新、推理输出一致；非零 gate 三路有梯度，
  原未来 latent target 不变。
- 新 Plus 测试：几何开/关 launcher 配置及 worker 快照通过；几何开启时等待步和非 replan
  帧均真实渲染，逐 episode 历史重置，seed 递增保留。
- 新完整 state 检查：修改几何配置、修改数据合同或删除 geometry_state.json 均拒绝恢复；
  相同合同可以恢复；正式入口只在几何开启时选择 GeometryTrainer，执行后恢复原绑定。
- 真实 Track4World + DA3 从 LeRobot 样本0、7提取到本目录独立缓存，written=2。
- 真实在线重提取与缓存 parity：mask 完全一致；track 最大相对 RMSE
  `0.0002445803547743708`，track_aux `0.00013114686589688063`，阈值0.005内。
- 单 A100 小尺寸实际 WAM/VAE，真实两个样本和缓存：4次更新，184个参数张量变化，
  训练/训练期验证无在线 tracker，权重及完整state保存/恢复通过。
- 两张 A100、DeepSpeed ZeRO-1、同类小模型：两个rank完成4次更新，175个参数张量变化，
  完整state hooks 保存/恢复通过。
- 真实普通 LIBERO：加载小模型训练权重、真实 T5 缓存，8步等待后2步控制，
  在线 Track4World 提取2次，动作均有限且为 `[32,7]`；几何模型峰值 allocated 约8.13GB。

本次环境为已有 `lf_fastwam`，torch2.5.1/CUDA12.1；已有 TorchCodec 不兼容时沿用
基线 torchvision/PyAV fallback。没有修改依赖版本。

## GPU 报告

- `outputs/verification/train_v1/report_rank0.json`
- `outputs/verification/ds2_v1/report_rank0.json`、`report_rank1.json`
- `outputs/verification/sim_v1/report.json`
- `outputs/verification/ds2_v1/checkpoints/state/step_000004/geometry_state.json`

输出目录沿原 .gitignore 忽略，不作为源码交付。复现命令见 TRACK4WORLD.md。
双卡复现，在配置好环境后执行：

```bash
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 2 --main_process_port 29763 \
  scripts/check_ablation_gpu.py --skip-online --output outputs/verification/new_ds2
```

## 边界

- 缓存只有2/277713个时刻；正式数据工厂仍拒绝不完整缓存。
- GPU训练检查使用两样本 wrapper 和小网络，不是生产全量DataLoader/完整5B训练；
  不证明16卡B4的显存和长期收敛，不证明几何提高成功率。
- 模拟检查复用真实 T5 缓存，不是完整T5 encoder正式启动验证。
- 本机未配置 Plus 专用源码/资产；Plus 仅完成配置、worker、渲染/history 等单元测试，
  未运行真实 Plus rollout、10030任务或成功率评测。
- `experiments/libero_plus/validation.json` 是基线历史证据，不是本次几何版本的验收清单。
