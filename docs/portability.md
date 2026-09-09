# FastWAM4D：异机部署与移植到另一份 FastWAM

本仓库是 FastWAM + Track4World + LIBERO 的研究复现包。根目录保留完整 FastWAM 源码和上游历史；第三方源码用固定 commit 的 Git 子模块获取，不依赖原训练服务器目录。

融合方案保持不变：当前 VAE latent 接收 scene / camera / track 三路残差 cross-attention，再进入世界模型。默认冻结原 FastWAM、VAE、Track4World，只训练几何 tokenizer 和融合模块，优化 video loss。不是直接给 ActionDiT 注入几何，也不是全参数训练。

## 包含什么

```text
src/fastwam/                       FastWAM + 三路融合 + 数据集/缓存
third_party/Track4World/           固定版本的几何模型源码
third_party/LIBERO/                固定版本的模拟器、任务和场景资产
third_party/utils3d/, Pi3/         Track4World 必要的源码依赖
patches/                          第三方修补 + 移植到其他 FastWAM 的差异
configs/experiments/libero_portable.yaml   无原服务器路径的配置模板
scripts/portability/              源码初始化、环境安装、HF 镜像下载
scripts/fastwam4d.py               特征抽取/两种训练/在线测试的统一入口
tests/                            CPU 回归和迁移测试
```

不上传模型权重、LIBERO 演示数据、几何特征缓存、训练输出、私钥或虚拟环境。第三方子模块保留各自许可证；尤其 Track4World 的使用条件应以其 `LICENSE.txt` 为准，不被根目录的 FastWAM MIT 许可证替代。

## 1. 获取源码和环境

```bash
git clone https://github.com/Shizhen-ZHAO/fastwam4d.git
cd fastwam4d

# 只初始化需要的四个固定版本依赖，不递归拉取无关 Open-d4rt。
python3 scripts/portability/bootstrap_sources.py
python3 scripts/portability/bootstrap_sources.py --check

# 先阅读安装计划；具体安装选项见 --help。
bash scripts/portability/install_environment.sh --help
bash scripts/portability/install_environment.sh
# 确认计划后创建新环境，不会修改已有环境。
bash scripts/portability/install_environment.sh --execute
conda activate "$PWD/.conda/envs/fastwam4d-cu121"
```

安装脚本默认只打印计划，需要显式 `--execute`。不要把三个上游仓库的 requirements 混合安装：它们的 Torch/CUDA、Transformers、NumPy/Python 要求不同。使用这里提供的联合依赖清单；目标是 Python 3.10、Torch 2.5.1 / CUDA 12.1。不要用根目录原版 `pip install -e .` 自动把 Torch 升级到另一套版本；环境脚本使用 `--no-deps` 安装 FastWAM。

需要 Linux / NVIDIA GPU。CUDA 驱动、EGL/OpenGL 运行库和显存取决于目标机器，不会由 Python 包自动提供。不要在原有训练环境里盲目升级依赖；使用独立环境。首次先跑 smoke，确认显存，再扩大 batch。默认两模型均在 `cuda:0`；双 GPU 可把几何提取器放到 `cuda:1`，但该进程必须能看见两张卡。

## 2. 下载权重和原始 LIBERO 数据

```bash
# 默认仅打印计划；没有 --execute 不会联网下载。
python scripts/portability/download_assets.py --models --data
python scripts/portability/download_assets.py --models --data --execute
```

下载器使用 `https://hf-mirror.com`，清除大小写 HTTP/HTTPS/ALL_PROXY，支持续传。权重默认在 `assets_local/`，数据默认在 `datasets/libero/`。可以用 `--assets-root`、`--data-root` 放到容量足够的磁盘，然后在下一步使用同样的路径。

可移植默认配置使用官方 `Wan-AI/Wan2.2-TI2V-5B` 的 VAE/T5 `.pth` 权重，设置 `redirect_common_files=false`。开发机原有的 DiffSynth 转换权重仍可通过自定义配置使用，但其 HF 来源在打包检查时不可访问，因此不作为新机器默认下载依赖。

需要的是原始 HDF5 格式的五个 suite：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`、`libero_90`，共 130 个任务、6,500 条演示。不是原版 FastWAM README 中的 LeRobot 数据集；这两种格式不能混用。

## 3. 在新机器生成配置

下载完演示文件后执行：

```bash
python scripts/fastwam4d.py configure \
  --assets-root assets_local \
  --data-root datasets/libero \
  --cache-dir feature_cache/libero \
  --model-device cuda:0 --geometry-device cuda:0 \
  --config configs/local/libero.yaml

python scripts/fastwam4d.py doctor --config configs/local/libero.yaml
python scripts/fastwam4d.py plan --config configs/local/libero.yaml
```

生成配置时会核查任务文件数量，并把路径解析为**新机器自己的绝对路径**。`configs/local/` 被 Git 忽略；不要将旧服务器生成的配置原样搬过来。再次写入不同配置会拒绝覆盖，可选择新文件名，或明确手工编辑本机配置。

训练超参数可用 `configure --steps 10000 --batch-size 1 --num-workers 2`，其他设置用 `--set geometry.history_stride=1 data.sample_stride=1`。单个 suite 的调试可以配置 `--suites libero_spatial`；正式默认包含全部五个 suite。

## 4. 先完成小范围验收

```bash
# 五个 suite 各一个任务，demo_0 的起始/完整历史/尾帧，共 15 个窗口。
python scripts/fastwam4d.py extract --config configs/local/libero.yaml \
  --cache-dir feature_cache/libero_smoke --smoke --train-steps 10

# 直接使用提取器自动生成的配置，训练过程中不构造 Track4World。
python scripts/fastwam4d.py train-offline \
  --config feature_cache/libero_smoke/training_config.yaml \
  --output-dir outputs/smoke_offline

# 同一批训练窗口，不读取几何缓存，每个训练前向在线运行 Track4World。
python scripts/fastwam4d.py train-online \
  --config feature_cache/libero_smoke/training_config.yaml \
  --output-dir outputs/smoke_online --steps 10

# 单独进程在线测试：只需 adapter、原始 RGB 和基础模型权重。
python scripts/fastwam4d.py test-online \
  --config feature_cache/libero_smoke/training_config.yaml \
  --adapter outputs/smoke_offline/geometry_adapter.pt \
  --output-dir outputs/smoke_online_test --inference-steps 20
```

在线训练使用 smoke 配置只是为了复用相同样本索引；**并不读取该目录的几何特征**。不想做任何预提取时，也可以直接用 `configs/local/libero.yaml` 配合 `train-online --train-samples 5 --steps 10`。

旧的 `scripts/run_*.sh` 和旧实验配置保留为历史参考，含原服务器目录。异机部署应使用上述 `scripts/fastwam4d.py`，不要直接使用旧包装脚本。

## 5. 全量提取、离线训练和在线训练

```bash
python scripts/fastwam4d.py extract --config configs/local/libero.yaml
python scripts/fastwam4d.py status --config configs/local/libero.yaml

python scripts/fastwam4d.py train-offline \
  --config feature_cache/libero/training_config.yaml \
  --output-dir outputs/full_offline --steps 10000

# 无需先提取，也不依赖 feature_cache 是否存在。
python scripts/fastwam4d.py train-online \
  --config configs/local/libero.yaml \
  --output-dir outputs/full_online --steps 10000
```

每个任务默认 `demo_0..44` 训练，`demo_45..49` 验证。`train_samples=0` 表示全部有效训练窗口，默认配置约 719,393 个窗口；10,000 步不代表完整 epoch。DataLoader 按需读取视频，在线模式不会一次性把所有历史图像加载进内存。`train-offline` / `train-online` 只区别冻结几何 raw 的来源，可训练部分和监督目标相同。

全量离线提取包含约 1,007,618 个决策窗口（含不参加训练的尾部）。FP32 原始特征未压缩约 1.66 TB，建议预留至少约 1.8 TB 可用空间；压缩实际大小依数据而定。本机小范围提取平均约 1.42 秒/窗口，单 GPU 全量是十几天量级的估计，不是保证。默认低于 50 GiB 空余会停止。

两终端可以分片提取，配置和缓存目录保持相同：

```bash
# 终端 A
python scripts/fastwam4d.py extract --config configs/local/libero.yaml \
  --num-shards 2 --shard-index 0 --device cuda:0
# 终端 B
python scripts/fastwam4d.py extract --config configs/local/libero.yaml \
  --num-shards 2 --shard-index 1 --device cuda:1
# 汇总全部分片，检查 full_libero_complete=true。
python scripts/fastwam4d.py status --config configs/local/libero.yaml --num-shards 1
```

分片按 episode 分配。普通已提交窗口会被跳过；可用 `--verify-existing` 重新核对已有张量。已知首次初始化的极小中断窗口尚有恢复缺陷，见后文；不要手工把未完成槽位标成完成。数据、权重和预处理绑定缓存身份，**不支持把既有缓存目录随意搬路径后继续使用**；本次流程是在新机器重新提取。

## 6. 在线推理和 LIBERO 闭环

`test-online` 是 HDF5 held-out 视频/动作验证，不是完整成功率基准。正式环境闭环：

```bash
python scripts/fastwam4d.py rollout-online --config configs/local/libero.yaml \
  --adapter outputs/full_offline/geometry_adapter.pt \
  --suite libero_spatial --task-id 0 --num-trials 1 \
  --output-dir outputs/rollout_spatial_0
```

`full_online` 训练出的 adapter 同样可用于在线测试和闭环。每个 episode 重置历史；每个环境步追加 RGB；每次决策提取一次，在去噪步骤内部复用。视觉 KV cache 不等于磁盘几何缓存。

当前提供单进程 rollout。不要直接把保存 `cuda:1` 的 adapter 交给仅暴露一张卡的旧 manager；那里的设备映射兼容性问题尚未修正。若迁移**已训练** adapter 到另一台机器，checkpoint 内的路径/设备仍需显式处理，本指南保证的主流程是迁移代码后在新机器提取、训练、测试。

## 7. 参考本仓库修改另一份 FastWAM

不必覆盖你另一份 FastWAM 的整个源码，也不要直接覆盖其未提交改动。先提交/备份那份工作树，再建实验分支。参考基线为 FastWAM `45d8e14`；不同版本优先手工移植。

最小改动分两部分：

1. **新增文件**：复制 `geometry_adapter.py`、`geometry_features.py`、`track4world_online.py` 到 `src/fastwam/models/wan22/`；复制 `libero_geometry.py`、`geometry_cache.py`、`geometry_cache_hdf5.py` 到 `src/fastwam/datasets/`。再带上本仓库的实验入口、portable 配置、测试和第三方初始化资料。
2. **已有文件挂接**：参考 `patches/fastwam-geometry-integration.diff`，涉及 `fastwam.py`、`mot.py`、`runtime.py`、`trainer.py` 以及 LIBERO 评测入口。先在目标仓库执行 `git apply --check /path/to/fastwam4d/patches/fastwam-geometry-integration.diff`；能干净应用才执行 `git apply`。检查失败则对照修改，不使用强制覆盖或忽略冲突。

挂接顺序：

```text
训练 batch.geometry_raw 或历史 RGB
  → build_inputs → _encode_geometry_raw → GeometryTokenizer
  → _condition_observation_latent（仅当前 VAE latent）
  → VideoDiT.pre_dit → MoT → 原 video loss

推理历史 RGB → _prepare_geometry → 同一 tokenizer/latent adapter
  → infer_joint 或 infer_action → LIBERO 动作
```

必须保持：raw detach、tokenizer 不 detach；未来监督 latent 不被融合改写；冻结主干仍允许对输入的 autograd；checkpoint 保存新增参数；训练 offline 模式不偷偷回退 online。改完至少运行：

```bash
PYTHONPATH=src python -m pytest -q tests
```

然后在目标机器跑第 4 节 smoke。测试通过只证明所覆盖的路径，不能代替完整 LIBERO 成功率实验。

## 已知限制与验收范围

- 本轮移植不改变之前审查的核心模型实现。尚有：HDF5 初始身份属性部分写入后的恢复问题；adapter 热启动未校验 heads/same_view_only；无缓存测试未完整校验训练预处理/producer/base checkpoint；旧并行 manager 的设备映射问题。不要把已有小规模通过当作这些边界已修复。
- 目前训练 runner 是单进程 + 多 DataLoader worker，不是 DDP trainer。`--adapter` 是权重热启动，不恢复 optimizer/sampler/步数；默认结束时保存 adapter，长训练的周期 checkpoint/完整恢复需另行实现。
- 旧实验报告中的 `outputs/...` 链接指向未上传的本机产物，不是 GitHub 附件。原版 57 项 CPU 测试、真实五任务离线训练及无缓存在线测试已在开发机通过；新迁移入口另有 CPU 回归。目标机器的驱动/EGL、完整环境安装和全量训练需按上述步骤验收，不能预先保证。
