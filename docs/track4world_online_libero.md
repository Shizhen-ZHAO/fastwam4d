# Track4World → FastWAM：LIBERO 在线几何融合 v1

本文保留 Action-Only 实验记录。后续“当前 VAE latent 的一次入口融合”实现与实验见 [VAE-entry 版本](vae_latent_geometry_online.md)。

已完成：真实 LIBERO HDF5 上的在线小样本训练、固定探针 loss 下降、适配器保存/新进程重载，以及官方 LIBERO 仿真中的在线闭环执行。**训练和推理均不读取预抽取的几何特征**；目前只有原有 T5 文本编码缓存。

基于本地官方 FastWAM checkout `45d8e1458921d83f8ad6cf9ce993d371208dabd0` 修改。没有替换成缩小版策略，也没有用随机特征做实验。单元测试中的合成输入仅用于测试。

## 实测结果（2026-09-09）

训练任务：`libero_spatial` 的 `pick up the black bowl from table center and place it on the plate`；对应仿真 task ID 2。选择 demo 0–3 中分散的 8 个完整历史窗口，batch size 1，训练 60 步。

| 训练步数 | 固定探针平均 action loss |
| --- | ---: |
| 0 | 0.05405139 |
| 10 | 0.05347500 |
| 20 | 0.05395701 |
| 30 | 0.05538959 |
| 40 | 0.05386656 |
| 50 | 0.05293573 |
| 60 | 0.05074972 |

最终比初始下降 **6.108%**，不是每一步单调下降。探针取训练窗口中的固定 3 个样本，分别固定随机种子 10000、10001、10002；比较时重新在线提取几何，保持噪声/时间步可比。该指标是官方 scheduler 加权的 action flow-matching loss，不是动作执行误差，也不是 held-out validation。

![固定探针 loss 曲线](../outputs/libero_track4world_online_60steps/fixed_probe_loss.png)

- 训练：54,529,566 个可训练参数；60 次训练更新用时合计 136.68 秒，不包含大模型加载、探针和保存；每步中位数 2.56 秒。
- 训练中在线提取耗时中位数 1.35 秒；实际会随 GPU 共享负载波动。
- 保存前推理：输出 `[32, 7]`，全部有限值，5 次去噪仅调用提取器 1 次。
- 新进程重载：加载原始 LIBERO FastWAM 权重，再加载 `geometry_adapter.pt`；运行官方仿真入口。
- 闭环：task 2、trial 0，**1/1 成功**，97 个策略动作步、10 次在线提取；每轮生成 32 步动作、执行前 10 步，20 次去噪。
- 闭环热启动后单次重新规划用时中位数 3.374 秒。这是在线计算，但**不是 20 Hz 实时控制**；仿真在规划期间等待。
- 单元测试：14 项通过，覆盖三分支梯度、零门控保持基线、全遮挡、BF16/FP32 混合、历史填充/reset、相机运动补偿、未来帧隔离、训练/缓存推理一致性和权重重载。

没有运行完整四套 LIBERO benchmark，也没有做原始 FastWAM 的配对成功率比较；因此不能把这一次成功归因于融合，不能据此宣称泛化或成功率提升。第一轮 6 步 smoke 的探针没有下降，保留在单独的 `outputs/libero_track4world_train_smoke`，没有作为成功结果使用。

## 一次策略前向提取什么

当前时刻为 t，输入各自独立的外部视角和腕部相机历史：

`history_images: [B, 2, 8, 3, 256, 256]`，RGB float `[0,1]`。默认 stride 1、20 Hz，覆盖 `[t−7, …, t]`，时间跨度 0.35 秒。`history_timestamps: [B,8]` 为相对 t 的秒数，最新为 0；`history_valid: [B,8]` 屏蔽 episode 起始处的填充。

两个视角分别运行同一套冻结 Track4World 权重；**不把两相机拼成一段时间序列，不假设它们已经在同一个世界坐标系**。

| 分支 | 冻结网络取值位置 | 原始张量 | 辅助几何 |
| --- | --- | --- | --- |
| Scene | DA3 输出 `feats` 最后 1024 通道，只取最新帧，采样 8×8 网格 | `[B,2,64,1024]` | 最新相机坐标 xyz、归一化 uv、深度质量，共 6 维 |
| Camera | DA3 `cam_dec` 的输入 hidden，只取最新帧 | `[B,2,3072]` | 归一化内参 4 + 相对旋转 6 + 相对平移 3，共 13 维 |
| Track | `flow3d_head` 最后一次迭代的 updated hidden，保留历史目标帧 | `[B,2,8,64,256]` | 位置 3 + 相对最新位置的偏移 3 + 速度 3 + 相对时间 1 + 跟踪置信度 1，共 11 维 |

Track 的第一个时刻是 anchor，其 hidden 使用跟踪器初始化 detail；其余时刻使用更新后的 tracking hidden。每个窗口从最早有效帧的 64 个规则网格 anchor 开始追踪；窗口之间重新选 anchor，**没有跨窗口持久 track ID**。

Scene/Camera 的最新 hidden 本身可在当前历史片段内聚合信息，但输入不包含 t 之后的帧。Track4World 冻结、`eval`、`no_grad`，不加入策略 optimizer/checkpoint。

## 坐标、遮挡和三路 token

Track4World 的 3D residual 不是世界系速度。实现先加回 anchor camera point 得到各目标时刻的相机系 endpoint。令 `T_k` 为上游输出的 camera-to-world 位姿，则对每个视角独立计算：

\[
p_{k\to t}=T_t^{-1}T_k\,p_k,\qquad
\bar p_{k\to t}=p_{k\to t}/s_t,
\]

其中齐次坐标形式隐含在第一式中；`s_t` 是最新帧采样点正深度的中位数。点坐标和位姿平移来自上游同一尺度。由此计算：

\[
\Delta p_k=\bar p_{k\to t}-\bar p_{t\to t},\qquad
v_k=(\bar p_{k\to t}-\bar p_{k-1\to t})/(\tau_k-\tau_{k-1}).
\]

这一步补偿腕部相机运动。双视角仍保留各自当前相机坐标和 view embedding，没有引入模拟器 GT 深度、GT camera pose 或跨相机标定作为策略输入。尺度归一化意味着这不是统一的、绝对米制 robot-base 几何编码。

Scene 有效性使用上游 `depth_conf > 2`，不把 DA3 质量分数误当概率。Track 使用两通道 sigmoid 置信度乘积 ≥ 0.25、anchor 深度质量、有限值和正深度检查。每路 prepend 一个始终有效的可训练 null token，避免全遮挡导致 attention NaN；只有一个有效历史帧时关闭 track bank 的实际轨迹 token。

三路各自通过 hidden projection、辅助几何 MLP 和 view embedding 映射到 512 维：

\[
z_s=P_s(h_s)+E_s(g_s)+e_{view},\quad
z_c=P_c(h_c)+E_c(g_c)+e_{view},\quad
z_{r,k}=P_r(h_{r,k})+E_r(g_{r,k})+e_{view}.
\]

Track 对每条轨迹的 8 个时刻做两层 temporal Transformer，然后 masked mean pooling 成一个轨迹 token。历史内部可双向 attention，因为全部是已经观察到的帧。最后三路 memory（含 null token）分别为：

`M_scene [B,129,512]`、`M_camera [B,3,512]`、`M_track [B,129,512]`。

## 怎样融入 FastWAM

只在 ActionDiT 的零起始层号 `[2,5,8,11,14,17,20,23,26,29]` 注入。位置是 MoT 共享执行逻辑中，原有 text/proprio cross-attention **之后**、FFN **之前**。Action hidden 是 1024 维；三路 attention 的内部维度 512、8 heads。

各分支有独立 Q/K/V/output 权重，Q 来自同一份 Action hidden，K/V 来自各自 geometry bank；不是把三路与文本直接拼接：

\[
C_b= W^O_b\operatorname{Attention}(W^Q_b\operatorname{LN}(x),
W^K_b\operatorname{LN}(M_b),W^V_b\operatorname{LN}(M_b);m_b),
\]
\[
x'=x+\sum_{b\in\{scene,camera,track\}}\tanh(\alpha_{\ell,b})C_b.
\]

所有门控 `alpha` 初始化为 0；attention 输出投影不同时置零。这样初始残差为 0，而第一步门控仍有梯度。v1 冻结原 FastWAM、VAE、T5、proprio encoder 和 Track4World，只训练 tokenizer、temporal encoder、三路 attention 和 gates。新模块 FP32，原策略 BF16；没有用 `.to(bfloat16)` 把新增可训练模块统一降精度。

训练仍调用原始 `FastWAM.training_loss`，动作的 flow matching 为：

\[
a_\sigma=(1-\sigma)a+\sigma\epsilon,\quad
u^*=\epsilon-a,\quad
L_a=\mathbb E[w(\sigma)\|u_\theta(a_\sigma,\sigma,o_t,c,p_t,M)-u^*\|^2].
\]

总 loss 保留官方 `L_video + L_action`。本配置中视频分支冻结且不接收 geometry，因此视频 loss 不驱动新参数；主要看固定探针 action loss。训练新模块普通参数 LR `1e-4`、gates LR `1e-3`，AdamW、梯度裁剪 1.0。

原有 MoT mask 保持不变：当前视觉 token 不看未来视频；action 只看当前视觉和 action，不看未来监督视频。未来 RGB 只用于原视频分支监督，不传给 Track4World。

代码调用链：

```text
训练：HDF5.__getitem__ → 过去 RGB + 独立未来监督
      → FastWAM.training_loss → build_inputs → _prepare_geometry
      → 冻结 Track4World → GeometryTokenizer → 各层三路 K/V
      → MoT._apply_expert_post_block → 三路 gated CA → FFN → loss/backward

推理：env 新观测 → LiberoHistoryBuffer.append（每个环境步）
      → 需要重新规划时 infer_action → _prepare_geometry（仅一次）
      → 当前视觉 KV prefill + geometry K/V
      → 20 次 action 去噪复用本轮 K/V → 32×7 动作 → 执行前 10 步
      → 更新历史，下一轮重新在线提取
```

K/V 只在同一轮推理内复用，不是离线特征，也不跨新观测复用旧 geometry。

## 环境和复现命令

工作目录 `/mnt/homes/zhaoshizhen/lf/repos/FastWAM`。已用环境：`/mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam`，Python 3.10.20、PyTorch 2.5.1 / CUDA 12.1、Transformers 4.49.0。

新增依赖已安装到独立目录 `/mnt/homes/zhaoshizhen/lf/.deps/fastwam_track4world`，清单见 [requirements](../requirements-track4world-online.txt)。它是现有环境的补充，不是从空环境安装的完整 requirements。没有替换原来的 torch 环境，也没有终止其他 GPU 任务。

所有权重已在本地，脚本清除大小写代理变量，设置 `HF_ENDPOINT=https://hf-mirror.com`，并启用 HF/Transformers 离线模式防止意外联网。当前不需要重新下载。默认 FastWAM 在 `cuda:0`、Track4World 在 `cuda:1`，用的是两张 A100 80GB 的剩余显存；不要额外设置 `CUDA_VISIBLE_DEVICES=0` 导致第二张卡不可见。

```bash
cd /mnt/homes/zhaoshizhen/lf/repos/FastWAM

# 新输出目录；已经有 metrics.jsonl 的目录会拒绝覆盖。
bash scripts/run_track4world_online.sh --steps 60 \
  --output-dir outputs/libero_track4world_online_reproduce

# 可单独检查真实在线特征：
bash scripts/run_track4world_online.sh --extract-only \
  --output-dir outputs/libero_track4world_extract_check

# 重载本次已经训练好的适配器，执行完整的单个 LIBERO episode：
bash scripts/eval_track4world_online_libero.sh \
  EVALUATION.output_dir=outputs/libero_track4world_rollout_reproduce

# 改用新训练的权重：
bash scripts/eval_track4world_online_libero.sh \
  EVALUATION.geometry_adapter=outputs/libero_track4world_online_reproduce/geometry_adapter.pt \
  EVALUATION.output_dir=outputs/libero_track4world_rollout_new_adapter

# CPU 单元测试：
PYTHONPATH=src OMP_NUM_THREADS=4 \
  /mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam/bin/python \
  -m pytest -q tests/test_geometry_online.py
```

训练配置见 [libero_track4world_online.yaml](../configs/experiments/libero_track4world_online.yaml)。其中输入数据是官方原始 HDF5 的 128×128 双相机图像，策略端 resize 到 224，geometry 端 resize 到 256；不是 FastWAM 重新渲染的高分辨率 LeRobot 训练集。实际 rollout 按本地官方入口渲染 256×256。原始 HDF5 图像采用 rotate-180，与本地官方评测 `get_libero_image` 的方向一致；仿真状态重放检查支持该选择，但重放图像不逐像素相等。

这轮使用独立实验 runner 调用官方模型/损失，不走通用 `Wan22Trainer` 的视频生成评测。v1 支持 `infer_action`，不支持带 geometry 的 `infer_joint`/未来视频可视化，也未验证多进程 DDP/ZeRO 在线提取。泛用 trainer 已保留冻结/参数筛选接入点，但完整评测路线需要后续适配。

## 文件入口与产物

- [在线提取器](../src/fastwam/models/wan22/track4world_online.py)：冻结权重、三个 hook、最新/历史选择、坐标处理。
- [三路 tokenizer / cross-attention](../src/fastwam/models/wan22/geometry_adapter.py)。
- [FastWAM 接入](../src/fastwam/models/wan22/fastwam.py) 与 [MoT 注入位置](../src/fastwam/models/wan22/mot.py)。
- [LIBERO 历史数据与 buffer](../src/fastwam/datasets/libero_geometry.py)、[实验 runner](../scripts/experiment_track4world_online.py)、[官方闭环入口](../experiments/libero/eval_libero_single.py)。
- [完整实验报告](../outputs/libero_track4world_online_60steps/experiment_report.json)、[原始逐步指标](../outputs/libero_track4world_online_60steps/metrics.jsonl)、[训练摘要](../outputs/libero_track4world_online_60steps/summary.json)。
- [适配器 checkpoint](../outputs/libero_track4world_online_60steps/geometry_adapter.pt)；不是完整 FastWAM 权重，必须与配置中的原始 LIBERO checkpoint、T4W/DA3 权重一起加载。
- [闭环结果](../outputs/libero_track4world_online_rollout/libero_spatial/gpu0_task2_results.json)，视频在同目录的 `videos/`。

`--adapter path` 是新优化器下的权重热启动，不恢复 optimizer/step。更改为 `train_mode=action/full` 时不能只保存 adapter，必须用完整 `save_checkpoint`。

后续可以讨论离线训练特征，但本轮没有开始预抽取。切换时应缓存本实现相同因果窗口、同一相机顺序/图像处理/尺度/置信度规则的冻结 raw features；可训练 tokenizer 和 CA 仍在线参与反向传播，推理保留现在的在线提取链路。下一步首先应做固定 held-out episodes 与原始 FastWAM 的配对评测，再判断几何信息是否带来可靠收益。
