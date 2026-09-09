# 当前 VAE latent 的一次三路残差几何融合

这是独立于 [Action-Only v1](track4world_online_libero.md) 的新实验路径。`geometry.target=vae_latent` 时，没有 ActionDiT 内部的 geometry adapter，也没有 VideoDiT block 内部的 geometry adapter。几何只在原 VideoDiT 的 patch embedding 之前融合一次。

后续已增加 [离线几何提取 → 缓存训练 → 在线测试](vae_geometry_offline_train_online_test.md) 路径。本文保留原来“训练也在线提取”的实验及结果，两者不覆盖。

## 实测结果：60 步（2026-09-09）

| 指标 | 训练前 | 训练后 | 说明 |
| --- | ---: | ---: | --- |
| 固定训练探针 video loss | 0.12725277 | 0.11024372 | 下降 13.37% |
| 固定验证探针 video loss | 0.19736771 | 0.18818784 | 下降 4.65%，非单调下降 |
| 单个验证窗口未来平均 PSNR | 22.5715 dB | 22.4093 dB | 略降，不能宣称生成质量整体提升 |

60 次训练更新用时合计 142.74 秒，不含模型加载、探针、生成和保存；每步中位数 2.56 秒，在线 Track4World 提取中位数 1.37 秒。三路均收到来自 video loss 的梯度，原主干未收到参数梯度。

20 步在线联合推理输出 9 帧 `224×448` 视频和 `[32,7]` 动作；每次调用严格检查只提取一次 geometry、只融合一次当前 latent。最终生成调用耗时约 5.05 秒。独立缓存动作推理约 2.33 秒，与联合推理动作的最大绝对差为 0.0078125，通过 BF16 数值容差检查。适配器保存/读取逐参数一致。

当前 22 项 CPU 测试通过，覆盖旧 Action-Only 回归和新 VAE-entry 路径。2 步 smoke 产物另存于 `outputs/libero_vae_geometry_smoke2`，不覆盖之前的 Action-Only 实验。

新进程重载本次适配器后的 LIBERO 闭环测试也已完成：task 2 / trial 0 **1/1 成功**，执行 94 个策略动作步、10 次在线提取，同时生成并保存未来视频。结果见 [闭环 JSON](../outputs/libero_vae_geometry_online_rollout/libero_spatial/gpu0_task2_results.json)，实际 rollout 和每轮 GT/预测对比视频分别在同目录的 `videos/`、`predicted_videos/`。这只是一个 episode 的功能验证，没有原始模型配对成功率实验。原评测入口的在线视频 PSNR 包含当前帧且只比较短期执行窗口，不能与上表排除首帧、覆盖完整预测窗口的 PSNR 直接比较。

![固定训练和验证探针](../outputs/libero_vae_geometry_online_60steps/video_loss_curve.png)

loss 的改善不等于图像生成质量改善。该验证视频的早期帧 PSNR 下降，较远期部分帧上升；只是一条窗口的观测，不据此推断一般性长时预测收益。逐帧结果可查看 [PSNR 曲线](../outputs/libero_vae_geometry_online_60steps/future_psnr_curve.png)。

## 实现与边界

```text
当前双相机 RGB [B,3,224,448]
  → 冻结 VAE → 原始当前 latent z_t [B,48,1,14,28]
  → 展平为 392 个 48 维 Query
  → 三路独立 cross-attention + 3 个 tanh 门控残差
  → 条件 latent z_cond [B,48,1,14,28]
  → 与未来加噪 latent 拼接
  → 原始 patch embedding → 原始 VideoDiT → 原始 latent 空间的未来预测

外部/腕部相机各最近 8 帧 → 冻结 Track4World（在线）
  → Scene / Camera / Track tokenizer → 上述三路 K/V
```

VAE latent 采用上游原有的标准化空间，不是 VAE encoder 中间层，也不是投影后的 3072 维 DiT token。Query 的线性映射为 `48→512`，每路 K/V 为 `512→512`，8 heads，结果 `512→48`。Query 额外加入每个相机内部的二维位置 MLP 和 view embedding。默认同视角 mask：当前外部视角的 Query 不直接读取腕部相机 geometry，反之亦然；null token 始终可读。双视角仍通过原 VideoDiT 的视觉 attention 交互。

注意：VAE 在拼接图像上编码，边界附近的感受野可能跨视角。这里的 view mask 是按 latent 横向分区定义的，不等于精确三维投影或跨相机标定。

令 `U=flatten(z_t)`，`M_b` 为 geometry memory，则：

\[
Q_b=W^Q_b\operatorname{LN}(U)+e_{xy}+e_{view},\quad
K_b=W^K_b\operatorname{LN}(M_b),\quad V_b=W^V_b\operatorname{LN}(M_b),
\]
\[
U_{cond}=U+\sum_{b\in\{scene,camera,track\}}
\tanh(\alpha_b)W^O_b\operatorname{Attention}(Q_b,K_b,V_b;m_b).
\]

三个门控初始化为 0，输出投影不同时置零。初始残差为 0；第一步门控有梯度，之后三路投影和 tokenizer 开始学习。共 **9,240,355 个可训练参数**，FP32；原 FastWAM 使用 BF16。

Scene/Camera 取当前时刻 hidden；Track 对过去 8 帧做时序编码后按每条轨迹池化。原始及最终 memory 形状沿用 Action-Only v1：Scene `[B,129,512]`，Camera `[B,3,512]`，Track `[B,129,512]`。同一窗口内包含当前帧，但不包含未来帧。跟踪器和几何提取规则不变。

## 训练：仅 video loss

配置 `training_loss_weights: {video: 1.0, action: 0.0}`，不使用 action loss 优化入口适配器。没有解冻原 VideoDiT/ActionDiT、VAE、T5 或 proprio encoder。

对原始未来 latent `z_F`：

\[
z^F_\sigma=(1-\sigma)z^F+\sigma\epsilon,\qquad
u^*=\epsilon-z^F,
\]
\[
L=L_{video}=\mathbb{E}[w(\sigma)\|u_\theta([z_{cond},z^F_\sigma],\sigma,c,p_t)-u^*\|^2].
\]

loss 只计算未来 latent，排除当前条件帧。`input_latents` 和 target 从未被几何融合覆盖；只在送入 VideoDiT 时替换首帧条件。原 action 预测仍可计算，但其 loss 权重为 0，不能将日志中的 `loss_action=0` 理解为动作误差为零。

VAE 编码和 Track4World 在 `no_grad` 中；适配器在其外部。冻结世界模型参数不等于关闭世界模型的 autograd：视频 loss 必须通过冻结主干反传到入口。训练时 MoT 的 train flag 用于启用已有 gradient checkpointing，冻结参数不会进入 optimizer；VAE 和 Track4World 保持 eval。

当前预训练配置 `action_conditioned=false` 未修改。这仍是观测/语言条件的未来视频预测，不是给定任意候选动作的可控动力学模型。

## 推理：提取和融合都只做一次

`infer_joint(..., history_images, history_timestamps, history_valid)`：

1. 只根据在线历史提取一份三路 memory。
2. 编码当前图像，计算一次 `z_cond`。
3. 每个去噪步将同一份 `z_cond` 放入模型输入；只更新原 latent 空间里的未来预测。
4. scheduler 状态中的当前帧始终保留原始 `z_t`，最终 VAE decoder 也接收原始当前 latent，而不是 `z_cond`。

`infer_action`：同样提取/融合一次，然后用增强后的当前 latent 做视觉 KV prefill。整个动作去噪过程复用该 cache。没有直接的 Action geometry attention。

联合推理与缓存动作推理应在数值容差内一致。FP32 小模型测试使用严格容差；真实 BF16 模型采用 `atol=rtol=1e-2`，不是要求逐位相同。带 geometry 的 `infer_joint` 默认不递归执行原有自检，否则会额外调用一次提取器；显式指定 `test_action_with_infer_action=True` 则会执行该第二次诊断调用。

## 真实实验与复现

所有权重、数据和依赖沿用现有本地环境。脚本清除代理变量，设置 HF 镜像地址并开启离线模式，没有新下载模型，也没有几何预抽取。T5 文本缓存不是 geometry 缓存。

默认 FastWAM 在 GPU 0，Track4World 在 GPU 1。使用现有 `lf_fastwam` 环境和隔离依赖目录，不升级 torch，不终止其他用户的 GPU 进程。

```bash
cd /mnt/homes/zhaoshizhen/lf/repos/FastWAM

# 两步接口 smoke，可缩短生成去噪：
bash scripts/run_vae_geometry_online.sh --steps 2 --inference-steps 3 \
  --output-dir outputs/libero_vae_geometry_smoke_reproduce

# 60 步在线训练 + 固定探针 + 20 步视频生成 + 保存重载 + 缓存推理检查：
bash scripts/run_vae_geometry_online.sh --steps 60 \
  --output-dir outputs/libero_vae_geometry_reproduce

# 默认加载本次 60 步结果；新进程执行一个 LIBERO episode，并保存未来视频：
bash scripts/eval_vae_geometry_online_libero.sh \
  EVALUATION.output_dir=outputs/libero_vae_geometry_rollout_reproduce

# 替换适配器时：
bash scripts/eval_vae_geometry_online_libero.sh \
  EVALUATION.geometry_adapter=outputs/libero_vae_geometry_reproduce/geometry_adapter.pt \
  EVALUATION.output_dir=outputs/libero_vae_geometry_rollout_new

# 仅生成报告/曲线，不加载模型：
PYTHONPATH=/mnt/homes/zhaoshizhen/lf/.deps/fastwam_track4world \
  /mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam/bin/python \
  scripts/report_vae_geometry_online.py

# CPU 测试：
PYTHONPATH=src OMP_NUM_THREADS=4 \
  /mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam/bin/python \
  -m pytest -q tests/test_geometry_online.py tests/test_vae_geometry.py
```

配置 [libero_vae_geometry_online.yaml](../configs/experiments/libero_vae_geometry_online.yaml) 显式继承原在线实验的路径/数据设置，覆盖 geometry target、层列表及优化目标。运行脚本会保存完整 `resolved_config.yaml`，拒绝覆盖非空输出目录。`--adapter` 仅热启动权重，不恢复 optimizer 或训练步数；此时 `before` 代表热启动模型，而不是原始基线。

数据仍为真实 LIBERO 原始 128×128 HDF5 双相机图像，经方向处理后 resize，非 FastWAM 高分辨率重渲染数据。任务是 `libero_spatial` 的 table-center bowl-to-plate（仿真 task ID 2）。

- 训练：demo 0–3 的 8 个完整历史窗口，batch size 1。
- 训练探针：其中固定 3 个窗口，噪声种子 10000+j。
- 验证探针：demo 4–5 的固定 3 个窗口，噪声种子 20000+j，与本轮训练 episode 不重叠。
- 预测对比：验证集第一个窗口，训练前/后使用相同生成种子 123；未来 PSNR 排除首帧。
- 这些验证 episode 只保证未参与本轮 adapter 微调，不保证基础 FastWAM checkpoint 从未见过。

结果以 [60 步 summary](../outputs/libero_vae_geometry_online_60steps/summary.json)、[逐步日志](../outputs/libero_vae_geometry_online_60steps/metrics.jsonl) 和 [汇总报告](../outputs/libero_vae_geometry_online_60steps/experiment_report.json) 为准。预测视频比较文件 `comparison_gt_before_after.mp4` 从上至下为 GT、训练前、训练后；每行从左至右为外部视角、腕部视角。

这是一组小样本可运行实验，不是完整 LIBERO benchmark，也未完成历史 RGB 条件/打乱几何/各分支消融。video loss 或单条视频 PSNR 的改善不能单独证明三维知识或机器人成功率提升。

## 代码位置

- [VAELatentGeometryAdapter](../src/fastwam/models/wan22/geometry_adapter.py)：48→512→48、位置/view embedding、同视角 mask、三路门控。
- [FastWAM](../src/fastwam/models/wan22/fastwam.py)：`geometry_target`、`build_inputs`、`training_loss`、`infer_joint`、`infer_action` 及保存重载。
- [实验 runner](../scripts/experiment_vae_geometry_online.py)：真正在线提取、视频目标、训练/验证隔离、梯度及生成检查。
- [CPU 回归测试](../tests/test_vae_geometry.py)：零门控等价、三路梯度、同视角隔离、未来 latent/target 不变、原始空间解码和一次提取/融合。
- [正式 LIBERO 入口](../experiments/libero/eval_libero_single.py)：历史 buffer、重载适配器、联合视频/动作闭环。

通用 `Wan22Trainer.evaluate` 的样本打包与视频评测尚未适配 history，当前应使用上述专用实验入口；多进程 DDP/ZeRO 尚未验证。
