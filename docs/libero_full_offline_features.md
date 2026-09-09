# 全 LIBERO 离线几何提取与直接训练

入口：[`extract_libero_geometry_full.sh`](../scripts/extract_libero_geometry_full.sh)。
它会生成 `training_config.yaml`，直接交给现有训练入口，**无需再转换缓存格式或修改模型代码**。

## 一次提取，然后直接训练

```bash
cd /mnt/homes/zhaoshizhen/lf/repos/FastWAM

# 只读盘点，不加载模型、不提取特征。默认就是 plan 模式。
bash scripts/extract_libero_geometry_full.sh --mode plan

# 真正全量提取：五个 suite、全部 episode、每一个决策帧，包含尾部。
# 先确认容量！也可以用 --cache-dir 指向有足够空间的目录。
bash scripts/extract_libero_geometry_full.sh --mode extract --device cuda:1

# 检查全量完成标记（所有分片完成后，用 num-shards=1 汇总检查）。
bash scripts/extract_libero_geometry_full.sh --mode status

# 可选完整 checksum 审核，不加载 Track4World；会读取全部特征。
bash scripts/extract_libero_geometry_full.sh --mode verify --log-every 1000

# 提取完成后，直接训练。不是未修改的默认 train.py，而是此离线特征入口。
bash scripts/run_vae_geometry_cached.sh train \
  --config outputs/libero_geometry_full_hdf5/training_config.yaml \
  --output-dir outputs/libero_full_train_run1

# 测试完全从 RGB 历史在线提取，不需要几何缓存。
bash scripts/run_vae_geometry_cached.sh test \
  --config outputs/libero_geometry_full_hdf5/training_config.yaml \
  --adapter outputs/libero_full_train_run1/geometry_adapter.pt \
  --output-dir outputs/libero_full_online_test_run1

# 正式闭环仿真：同样在线提取，默认一个 task 2 episode。
bash scripts/eval_vae_geometry_cached_libero.sh \
  EVALUATION.geometry_adapter=outputs/libero_full_train_run1/geometry_adapter.pt \
  EVALUATION.output_dir=outputs/libero_full_rollout_run1
```

目前默认目录已创建 manifest、inventory 和正式的 [training_config.yaml](../outputs/libero_geometry_full_hdf5/training_config.yaml)，**尚未执行百万窗口的全量提取**。未完成提取时，训练入口会在加载基础模型之前报告缺失窗口，不会偷跑在线提取。

训练默认每个任务用 `demo_0..demo_44`，`demo_45..demo_49` 留给在线验证；这是本实验显式设定的 episode 划分，不是官方新增 split。缓存仍包含全部 50 条演示。若希望调整训练划分，可以复制生成的配置并修改 `data.episode_ids`，同时保持验证不重叠。用全部 50 条演示训练时，不能再把同一批演示当作 held-out 验证；请用独立仿真评测。

生成配置固定 `train_samples=0`、`train_indices=null`、`cache_backend=hdf5`、`load_history=false`、`history_only=false`。这意味着 DataLoader 覆盖所有所选训练演示的有效训练窗口，而不是此前的 8 个小样本。训练默认 10,000 个优化步，**不代表跑完一个 epoch**；可用 `--steps` 改变步数。默认 batch size 1、两个 spawn worker；可通过 `--batch-size` / `--num-workers` 调整。

正式配置已实际构建检查：130 个任务、5,850 条训练演示、**719,393 个有效训练窗口**。在全量缓存尚未提取的状态下，覆盖检查在模型加载前正确报错，Track4World 模型没有被导入。

原 FastWAM / VAE / Track4World 仍冻结，只训练 tokenizer 和当前 VAE latent 的三路残差 cross-attention（9,240,355 参数，video loss 权重 1、action loss 权重 0）。每个 batch 仍需读取未来监督视频、动作和状态，并计算原 VAE latent；只预存冻结几何 raw，不预存可训练 memory 或 VAE latent。

## 覆盖范围与空间

2026-09-09 对本地原始 HDF5 release 的完整只读盘点：

| Suite | 任务文件 | 演示 | 决策窗口（stride=1） |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 10 | 500 | 62,250 |
| LIBERO-Object | 10 | 500 | 74,507 |
| LIBERO-Goal | 10 | 500 | 63,728 |
| LIBERO-10 | 10 | 500 | 138,090 |
| LIBERO-90 | 90 | 4,500 | 669,043 |
| 合计 | 130 | 6,500 | 1,007,618 |

默认包含 LIBERO-90，不会把只有四个 benchmark suite 的 40 个任务当作全 LIBERO。脚本检查每个 suite 的任务数量、每个任务的 `demo_0..demo_49`、双相机 RGB、动作/状态的形状和长度；缺失时失败，不静默遗漏。

每窗口原始张量 1,646,826 字节，全量未压缩约 **1.66 TB（1.51 TiB）**。本机检查时空余约 1.32 TB。真实完整历史窗口的无损压缩张量约 1.30 MB/窗口，粗估全量仍在 **1.3 TB 左右，且另有元数据开销**，当前目录余量不足以放心启动全量作业。建议预先选择至少约 1.8 TB 可用空间的缓存目录；不要用起始补零窗口的较高压缩率估算全量。

脚本每个待写窗口前检查空余容量，默认低于 50 GiB 停止；已经提交的窗口保留，可以扩容后续跑。压缩大小和耗时依赖数据、GPU 负载，不保证上述估计。此次小范围真实提取平均约 1.42 秒/窗口；线性外推单 GPU 是十几天量级，不是几小时。此轮只启动了 smoke，未自动启动全量作业。

`--sample-stride 4` 可以降低提取密度，但此时不是逐帧全量；训练配置也必须采用对应 stride，读取未提取的帧会报错。`--suites libero_spatial libero_object libero_goal libero_10` 可以显式只处理四个 benchmark suite。

## 多 GPU 分片、续跑和校验

在两个终端启动，使用同一个缓存根目录、数据/提取配置；不同进程仅改变设备和分片号：

```bash
# 终端 A
bash scripts/extract_libero_geometry_full.sh --mode extract \
  --num-shards 2 --shard-index 0 --device cuda:0

# 终端 B
bash scripts/extract_libero_geometry_full.sh --mode extract \
  --num-shards 2 --shard-index 1 --device cuda:1

# 完成后统一核对；full_libero_complete=true 才是默认全 LIBERO 全量完成。
bash scripts/extract_libero_geometry_full.sh --mode status --num-shards 1
```

分片按 `SHA256(源文件路径, episode)` 分配，不按窗口随机分配，因此同一 episode 不会正常落入多个写进程。相同命令重跑会跳过已提交窗口；默认检查提交表和窗口身份，`--verify-existing` 还会重新读取并校验所有已有张量。不要仅凭 `.h5` 文件存在就认为该 episode 已提完。

`--max-windows N` 是调试用的分片前缀限制；不会将剩余未提取窗口标为完成。`--smoke` 明确只选每个 suite 第一个任务的 demo_0，提取起始帧、完整历史帧和尾帧；要求另选缓存目录，不能误写全量缓存根目录。`prepare` 只创建清单和训练配置；`plan/status/verify` 不写缓存。

自动生成的配置保持不可静默覆盖。调整训练超参数建议复制配置到新文件，不要在提取过程中编辑缓存目录里的生成配置。不同分片的 `--device` 不改变生成配置中的在线提取设备，避免并行进程发布互相冲突的训练配置；在线 GPU 设置可在训练配置副本的 `geometry.extractor.device` 中修改，设备号不影响缓存身份。

当前提供单训练进程和多 DataLoader worker，不是多机 DDP trainer。`--adapter` 仅热启动适配器权重，不恢复 optimizer / sampler / 训练步数；默认适配器在训练结束时保存，长训练如需中途 checkpoint / 完整恢复还需扩展该训练 runner。

## 为什么训练时能直接用、不会错位

1. **按真实帧号索引。** 一个 HDF5 文件对应源文件内一个 episode，slot `t` 就是原始决策帧 `t`，不是采样后的第几行。改成每四帧训练时，会读取 `0,4,8,...` 的槽位。
2. **缓存身份绑定。** manifest 绑定源数据、历史长度/stride/fps、旋转/resize、相机顺序、grid、跟踪器权重和源码版本。每条记录还保存窗口身份及张量 SHA256。大权重/数据文件采用 size+mtime_ns 指纹，不冒充全文件内容哈希。
3. **提交标志最后写。** raw 特征及掩码和 checksum 全部写入、flush 后才将 `written[t]=1`。不完整槽位不可读取；损坏/错位/错形状直接报错。LZF + byte shuffle + Fletcher32 为无损存储，保持 FP32/bool，不做 FP16 存储量化。
4. **完整覆盖检查在模型加载前。** 训练先逐 episode 检查所选训练窗口的提交状态和身份，之后每个 batch 再校验张量 SHA256。不在每次训练启动时强制读取整个 TB 级缓存；有需要可启用 `verify_cache_before_training=true`。
5. **历史和监督分开。** 离线提取只调用 `history_item`，最多看到当前帧；也可提取 episode 尾部。训练 Dataset 仍只产生有完整 32 步未来监督的窗口，且 `load_history=false`，不读历史 RGB、不构造跟踪器。
6. **同一个可训练路径。** `geometry_raw → _encode_geometry_raw → GeometryTokenizer → 当前 latent 三路残差 CA → 原世界模型`。只 detach 冻结 raw；tokenizer 和融合模块正常参与 autograd。缓存中没有可训练 tokenizer 输出、投影后的 K/V 或优化器状态。
7. **多 worker 安全。** 每次操作打开并关闭 HDF5，不继承句柄；进程间使用读写锁。允许训练从全量缓存中选择任务文件/episode 子集，但不允许读取未登记的新源文件或配置不匹配的数据。

HDF5 不是断电事务数据库。正常中断可续跑；底层文件若因硬杀/断电损坏，会失败退出，不自动覆盖、删除或猜测修复。

## 测试时仍在线

`infer_joint` 和 `infer_action` 都只接收实时 RGB 历史，不接收磁盘几何特征。每次策略决策执行一次冻结 Track4World 提取和一次当前 latent 融合；去噪内部可复用该条件。动作推理中的视觉 KV cache **不是离线几何缓存**。每个新 episode 都新建历史 buffer；历史按环境步更新，不仅在重规划时更新。

在线和离线使用相同的相机顺序、方向处理、缩放、历史有效性、时间戳和冻结提取器。FP16 跟踪器与 BF16 FastWAM 的独立重算可有数值误差；磁盘序列化则要求无损，二者不能混为一谈。此前数值对照和容差说明见 [详细融合实验说明](vae_geometry_offline_train_online_test.md)。

## 本轮实测与代码

- **57 项测试通过**：旧方案回归、新 HDF5 损坏/错位/未提交拒绝、源文件子集、非连续真实帧号、shuffle + spawn workers、历史与在线 buffer 精确对齐、尾部、训练反传、推理禁止读取磁盘缓存等。
- 五个 suite 各一任务：共 15 个真实窗口，包含 5 个起始窗口和 5 个尾部窗口；全部无损写入/读取。续跑全部跳过，跟踪器初始化和调用均为 0。
- 使用生成的配置原样直接进入真实 FastWAM，2 个 worker，5 个训练窗口，10 次 optimizer 更新。三路 tokenizer / CA 均收到梯度，冻结主干没有参数梯度，训练全程无 Track4World；固定探针 `0.13939907 → 0.13925727`，仅作为接线验证，不能视为有效性研究结论。
- [真实直接训练结果](../outputs/libero_full_hdf5_direct_train_10steps/summary.json)；[本轮 smoke 自动生成配置](../outputs/libero_full_geometry_hdf5_smoke/training_config.yaml)。
- 新进程把 `--cache-dir` 指向一个确认不存在的目录，重载本轮 10 步训练的权重，完成正式 20 步在线验证、9 帧视频生成与 `[32,7]` 动作推理。共 9 次在线提取，测试不依赖缓存；联合/视觉 KV 缓存动作的最大差 `0.0078125`，通过既有 `atol=rtol=1e-2` 检查。见 [无磁盘缓存在线测试结果](../outputs/libero_full_hdf5_online_no_cache_test_20steps/summary.json)。此前 3 步快速测试已完成在线验证和生成，但动作对照有 1/224 个元素超出此容差（差 0.015625）；未放宽阈值，最终验收使用正式 20 步设置。

源码：

- [全量脚本](../scripts/extract_libero_geometry_full.py)：盘点、分片、提取、验证、生成训练配置。
- [HDF5 缓存实现](../src/fastwam/datasets/geometry_cache_hdf5.py)：按真实帧号、提交、无损校验、读取锁。
- [直接训练 / 在线测试入口](../scripts/experiment_vae_geometry_cached.py)：自动选择 `cache_backend`，训练无跟踪器。
- [新增回归测试](../tests/test_geometry_cache_hdf5.py)。

复跑本轮小范围直接训练验收：

```bash
bash scripts/extract_libero_geometry_full.sh --mode extract --smoke \
  --cache-dir outputs/libero_full_geometry_hdf5_smoke_new --train-steps 10
bash scripts/run_vae_geometry_cached.sh train \
  --config outputs/libero_full_geometry_hdf5_smoke_new/training_config.yaml \
  --output-dir outputs/libero_hdf5_direct_train_new
```
