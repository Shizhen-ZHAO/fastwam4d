# Track4World 离线提取、FastWAM 缓存训练、在线测试

本方案延续 [当前 VAE latent 一次三路残差融合](vae_latent_geometry_online.md)，只更换训练时冻结几何特征的来源。**没有改为缓存最终 memory，也没有把几何重新注入 ActionDiT。**

全量五个 suite（含 LIBERO-90）的 HDF5 分块缓存、分片提取及自动生成训练配置，见 [全 LIBERO 离线特征与直接训练](libero_full_offline_features.md)。本文保留原先单窗口 `.pt` 小实验的配置及结果。

## 已完成的真实数据验证（2026-09-09）

预提取了 8 个训练窗口及 4 个补齐历史的 episode 起始窗口，合计 19,803,600 字节（约 19.8 MB）。磁盘序列化逐值无损；再次运行预提取时校验、跳过全部 12 个窗口，Track4World 初始化/调用均为零。

| 检查 | 训练前 / 基线 | 60 步后 |
| --- | ---: | ---: |
| 磁盘训练固定探针 video loss（3 窗口） | 0.12725277 | 0.10993548（下降约 13.61%） |
| 在线保留集 video loss（3 窗口） | 0.19736771 | 0.18603447（下降约 5.74%） |
| 单个验证窗口未来平均 PSNR（排除首帧） | 22.5715 dB | 22.8111 dB |

60 步训练全程没有创建 Track4World，在线提取次数为 0；三路 tokenizer 及 cross-attention 均收到梯度，原主干没有参数梯度。当前机器每步中位数约 0.486 秒；更新、其间探针和最后保存合计约 35.89 秒，不含基础权重加载和初始探针。这不是与旧实验严格控制条件的速度 benchmark。

重载 adapter 后，20 步在线去噪生成 9 帧 `224×448` 视频、`[32,7]` 动作；每次调用各一次提取和当前 latent 融合。单独动作推理与联合推理最大动作差 `0.0078125`，通过既有 BF16 容差检查。

数值对照不是“逐位在线等价”：相同窗口的磁盘 video loss `0.0314680077`，在线 `0.0314996243`，差 `3.16e-5`（约 0.10%），各自固定种子重复一致。raw scene/camera hidden 完全一致；track hidden 最大差 `0.00244`，track_aux 最大差 `0.00402`，掩码完全一致；在线重复提取本身的最大差分别为 `0.00288` 和 `0.00410`。此前更严格的 FP32 式独立重算阈值未通过，两个早期测试目录保留了配置和已完成的验证日志；最终检查区分磁盘无损与 FP16/BF16 数值容差，详见下方说明。

产物：[训练 summary](../outputs/libero_vae_geometry_cached_60steps/summary.json)、[adapter 权重](../outputs/libero_vae_geometry_cached_60steps/geometry_adapter.pt)、[在线测试 summary](../outputs/libero_vae_geometry_cached_online_test_final/summary.json)、[在线预测视频](../outputs/libero_vae_geometry_cached_online_test_final/trained_prediction.mp4)。**41 项 CPU 回归测试通过**，包括实际 spawn 多 worker 读取缓存。

新进程 LIBERO 闭环也已完成：task 2 / trial 0 **1/1 成功**，95 个策略动作步、10 次在线 Track4World 提取，结果明确标记 `features_precomputed=false`。本次闭环使用 `infer_action` 的视觉 KV 缓存路径（不是几何磁盘缓存），未来视频的 `infer_joint` 路径另由上面的在线窗口测试验证。参见 [闭环结果](../outputs/libero_vae_geometry_cached_rollout/libero_spatial/gpu0_task2_results.json) 与同目录的 `videos/`。

这些是单任务、8 个训练窗口的小样本功能实验。不能据此声称完整 LIBERO 成功率提升；held-out episode 仅针对本次 adapter 拟合，原基础 checkpoint 可能已见过这些演示。

## 两条路径

```text
预提取：每个决策窗口的双相机因果 RGB 历史
        → 冻结 Track4World → FP32 原始三路特征 / 辅助量 / bool 掩码 → 磁盘

训练：  磁盘 raw → 可训练 GeometryTokenizer → 三路 memory ─┐
        当前 RGB → 冻结 VAE → 当前 latent ───────────────┤
                                                        ↓
                                     一次三路残差 cross-attention
                                                        ↓
                                  原 VideoDiT → 原空间的未来 video loss

测试：  实时 RGB 历史 → 冻结 Track4World → 同一个 tokenizer / 融合模块
        → 在线世界预测 / 动作；一次决策只提取、融合各一次
```

预提取窗口严格截止于当前决策时刻 `t`，不包含未来图像，不读取未来动作来生成几何。每个相机单独编码，顺序是外部视角、腕部视角。Scene / Camera 取当前帧，Track 取最近 8 帧（含当前帧），20Hz、stride=1。每个窗口独立定锚，不能对完整 episode 双向提取后切片，那会泄露未来。

缓存内容（每条记录不含 batch 维，DataLoader 自动堆叠）：

| 字段 | 单窗口形状 | 含义 |
| --- | --- | --- |
| `scene`, `scene_aux`, `scene_valid` | `[2,64,1024]`, `[2,64,6]`, `[2,64]` | 当前 scene hidden、空间/图像位置等辅助量、有效性 |
| `camera`, `camera_aux`, `camera_valid` | `[2,3072]`, `[2,13]`, `[2]` | 当前 camera hidden、归一化内参/窗口相对位姿、有效性 |
| `track`, `track_aux`, `track_valid` | `[2,8,64,256]`, `[2,8,64,11]`, `[2,8,64]` | 近期轨迹 hidden、位置/位移/速度/时间/置信度、有效性 |

左侧缺历史时，读取 RGB 可重复最早帧，但提取器只使用有效帧，输出的 track 特征按历史长度补零，并保存无效掩码；不把重复帧当成真实运动。相机运动补偿、尺度归一化和 track 坐标转换均沿用同一个在线提取器。

缓存之后仍可训练：三路投影、辅助量编码、view embedding、track temporal transformer、三路 cross-attention 和门控；共 9,240,355 参数。未来监督视频仍从 HDF5 读取、经原 VAE 编码，未预存 VAE latent。原 FastWAM 主干保持冻结，video loss 权重 1，action loss 权重 0。

## 运行

使用现有本地 `lf_fastwam` 环境和权重，不新增下载。包装脚本清除代理变量，设置 HF 镜像并启用离线模式。默认 FastWAM GPU 0，Track4World GPU 1。训练进程不创建 Track4World，不需要为其分配 GPU。

```bash
cd /mnt/homes/zhaoshizhen/lf/repos/FastWAM

# 1. 预提取默认 8 个训练窗口，额外检查 4 个 episode 起始窗口。
#    --verify-online 会独立重算一个完整历史窗口，检查数值一致性。
bash scripts/run_vae_geometry_cached.sh extract \
  --include-episode-starts --verify-online

# 2. 60 步缓存训练。缓存缺失、损坏、不匹配均报错，不回退在线。
#    重跑时必须换空输出目录；--adapter 可仅热启动 adapter 权重。
bash scripts/run_vae_geometry_cached.sh train

# 3. 新进程在线验证与视频/动作生成。不读取也不要求几何缓存存在。
bash scripts/run_vae_geometry_cached.sh test

# 可选开发诊断：额外核对训练窗口的在线 raw / 缓存 raw 和固定噪声 loss。
# 此模式会读取缓存，但验证集预测、生成和 action 仍只使用在线 RGB。
bash scripts/run_vae_geometry_cached.sh test --verify-online \
  --output-dir outputs/libero_cached_online_parity_check

# 4. 真正的 LIBERO 闭环仿真。默认 task 2、一个 episode、20 去噪步。
bash scripts/eval_vae_geometry_cached_libero.sh

# 可选：闭环时同时生成未来视频（正确的配置键是 visualize_future_video）。
bash scripts/eval_vae_geometry_cached_libero.sh \
  EVALUATION.visualize_future_video=true \
  EVALUATION.output_dir=outputs/libero_cached_joint_rollout
```

默认配置是 [libero_vae_geometry_cached.yaml](../configs/experiments/libero_vae_geometry_cached.yaml)，继承此前实验的本地权重、数据和融合配置。当前默认训练只覆盖 task 2 的 demo 0–3 中 8 个窗口，batch size 1；测试是 demo 4–5 的 3 个窗口，不参与本轮 adapter 优化。训练固定探针也是从磁盘读特征；真正的 held-out 测试和闭环测试在独立进程在线提取，训练进程里不加载跟踪器。

训练保存 `geometry_adapter.pt`、`resolved_config.yaml`、`metrics.jsonl`、`summary.json`。在线测试保存基线/训练后预测视频、动作和 summary。基线通过将三路门控归零恢复原当前 latent；相同测试窗口、噪声/生成种子便于比较。测试从 checkpoint 恢复 geometry 架构和在线提取配置，缓存目录不是推理依赖。

## 更多数据 / 多进程读取

`train_samples=0` 选择配置中所有 episode 的所有有效决策窗口（包括补齐历史的起始窗口）。默认 `sample_stride=4`：每隔四个决策时刻缓存一次，而不是每一帧。若需要每帧训练，先在新配置中设 `data.sample_stride=1`。

```bash
# 扩大到当前配置的全部训练窗口，可复用同配置下已有条目并补齐缺失。
bash scripts/run_vae_geometry_cached.sh extract --train-samples 0
bash scripts/run_vae_geometry_cached.sh train --train-samples 0 \
  --num-workers 2 --steps 1000 --output-dir outputs/libero_cached_all_windows
```

要扩大到更多任务/四个 suite，请复制配置并修改 `data.files`、`data.episode_ids`、`validation_episode_ids`。改变文件集合时使用新的 `--cache-dir`；预提取与训练使用同一配置。数据集仍是原始 128×128 LIBERO HDF5 release 经方向处理和 resize，不宣称复现 FastWAM 高分辨率重渲染训练数据。新任务的 T5 context 在训练/测试入口自动按本地 T5 权重生成；几何预提取不需要 T5 context。

`--batch-size` 可调整训练 batch；峰值显存仍受原世界模型的反向传播影响。DataLoader 逐批读取，不把全部几何或监督视频驻留内存；worker>0 使用 spawn，避免 fork 已初始化 CUDA 的训练进程。这里提供单进程训练入口及多 worker 数据读取，不是新的多机 DDP trainer。

## 缓存安全与限制

- 存储边界是冻结 raw，FP32 + bool，无量化；torch.save/load 要求数值完全相同（允许原本的 NaN）。独立在线重算因上游 FP16 / CUDA 可有小误差，迭代跟踪和 20Hz 速度差分会放大该误差。诊断检查 raw 逐元素 `atol=1e-2, rtol=1e-3` 且 RMS 误差不超过 `1e-3 × max(参考RMS,1)`；bool 掩码必须完全一致。另记录在线/在线、磁盘/磁盘的重复对照。磁盘固定噪声 loss 重复检查用 `atol=1e-5, rtol=1e-4`；与独立在线提取比较时考虑原 FastWAM 的 BF16 量化，使用 `atol=1e-5, rtol=max(1e-4, policy dtype eps)`，当前 BF16 的 eps=1/128。不宣称独立在线推理逐位一致。
- `manifest.json` 记录相机顺序、history 长度/stride/fps、resize/旋转方式、grid、迭代次数、置信度阈值、权重路径、提取代码 SHA256、PyTorch/CUDA 版本。大型数据/权重使用文件大小和 mtime_ns 指纹，**不是全文件内容 SHA256**；每个特征记录另有真实张量 SHA256。
- 每条窗口身份包含源文件、episode、决策帧、实际历史帧索引、相对时间和有效性。缺失、错窗口、错形状、校验失败均拒绝使用，不静默切在线。
- 原子发布文件，已完成条目不覆盖。再次执行 `extract` 会逐个校验并跳过已有条目；中断后重跑即可。损坏条目不会自动删除，应查清原因后使用新缓存目录。
- 训练不 import / 构造 Track4World 模型。若本机有提取权重/源码，会校验是否改变；缓存专用训练机可以没有这些模型文件，但需要同一源数据路径/指纹和配置。测试机器必须有 checkpoint 指定的在线权重/源码。
- 修改 history、相机顺序、RGB 几何预处理、跟踪器权重/源码需要重新预提取；改 tokenizer 维数或其他可训练融合参数不需重提取。
- 不要在缓存训练时额外随机裁剪、翻转或改变视角排列；否则缓存几何与当前 RGB 不再对应。要做这些增强，需同步变换所有空间量，或对每种增强重新提取。
- 当前是一窗口一 `.pt` 文件，约 1.65 MB/窗口；大规模全 LIBERO 需先估算磁盘空间。未实现分片压缩、分布式预提取调度或自动缓存淘汰。

## 代码与测试

- [geometry_cache.py](../src/fastwam/datasets/geometry_cache.py)：manifest、原子发布、校验和、磁盘 Dataset。
- [libero_geometry.py](../src/fastwam/datasets/libero_geometry.py)：`history_item` 仅因果 RGB，`load_history=False` 不读历史图像。
- [FastWAM](../src/fastwam/models/wan22/fastwam.py)：`build_inputs` 显式选择 `geometry_raw` 或 RGB；`_encode_geometry_raw` 共享可训练路径；`infer_joint/infer_action` 保持在线。
- [实验入口](../scripts/experiment_vae_geometry_cached.py)：`extract/train/test` 三个独立进程模式与实际梯度、加载检查。
- [缓存测试](../tests/test_geometry_cache.py)：因果读取、无损存取、过期/损坏拒绝、batch 对齐、三路梯度、训练零提取、推理在线。

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 \
  /mnt/homes/zhaoshizhen/lf/.conda/envs/lf_fastwam/bin/python \
  -m pytest -q tests/test_geometry_online.py tests/test_vae_geometry.py tests/test_geometry_cache.py
```
