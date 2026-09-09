# 可移植发布验收记录

日期：2026-09-09。对应代码提交：`ca6e111`（后续仅补充本文档）。这是功能验收记录，不是完整 LIBERO 成功率报告，也不证明模型效果提升。

## 源码与 CPU 测试

- 从已提交代码创建干净 Git 克隆，不带原工作树的未提交文件、权重、数据和缓存。
- 四个子模块从现有本地 Git 对象初始化到发布中固定的上游 commit，避免重复下载；本项不代表已在新机器通过网络完成克隆。
- `bootstrap_sources.py` 实际应用 Track4World/LIBERO 补丁及相对导入链接；再次执行 `--check` 通过。
- 实际导入的 Track4World、GeometryTokenizer、utils3d 均来自干净副本；没有构造模型。
- 干净副本运行 `PYTHONPATH=src python -m pytest -q tests`：**165 passed in 28.17s**。
- 安装脚本 shell 语法检查、默认只读安装计划、HF 镜像下载计划通过。下载器和环境脚本的行为有 mock/CPU 测试覆盖，不等于完成真实全新环境安装。

## 真实权重的小范围运行

使用开发机现有 Python 3.10 / Torch 2.5.1 环境、两张可见 A100 80GB：FastWAM 在 `cuda:0`，Track4World 在 `cuda:1`。batch size 1，DataLoader workers 2，历史长度 8，五个 suite 各一个任务的 `demo_0`。可训练参数 9,240,355；只优化当前 VAE latent 的几何 tokenizer/残差融合，基础模型冻结。

以下均通过新的 `scripts/fastwam4d.py` 入口运行。

| 验收项 | 实际结果 |
| --- | --- |
| 新目录 `extract --smoke` | 五任务共 15 个窗口全部写入，回读张量完全一致 |
| 提取统计 | 15 次提取；raw 共 24,702,390 bytes；平均提取 1.447 秒/窗口 |
| 自动生成配置 → `train-offline` | 直接使用提取目录的 `training_config.yaml`，完成 3 步训练 |
| 离线训练路径 | Track4World 未初始化，训练和 probe 的在线提取次数均为 0 |
| `train-online` | 完成 3 步训练，每个训练前向提取一次；另有 2 次 probe 提取 |
| 三路可训练性 | scene / camera / track 的 tokenizer 和 cross-attention 均观察到梯度 |
| 权重保存/重载 | 两种训练模式的 adapter 重载数值一致 |
| 独立进程 `test-online` | 离线训练 adapter 完成在线 held-out 验证、联合生成及 action-only 推理 |

离线/在线训练使用相同的五个训练窗口、seed 和超参数，三个训练步的 loss 在本次运行中完全相同：

```text
step 1: 0.4632073045
step 2: 0.2508562803
step 3: 0.6221858859
```

固定训练 probe 从 `0.1958217323` 到 `0.1960792392`，**没有下降**。3 步只能验证链路，不能作为收敛证据；不同 batch 的逐步 loss 也不能直接当作学习曲线。

在线测试使用 `libero_spatial` 的一个 `demo_45` 窗口，20 步去噪。每次联合生成只在线提取一次、融合当前 latent 一次；生成 9 帧 `224×448` 双视角视频和 `32×7` 动作。action-only 与联合推理的动作在设定容差内一致，最大绝对差 `0.0078125`。这里的 visual KV cache 不是磁盘几何缓存。

该单窗口 video loss：零门控 baseline `0.1851973385`，adapter `0.1849558055`；视频 future PSNR 分别约 `23.96575` 和 `23.96275`。**不能据此声称泛化或操控成功率提高**，基础模型预训练也可能包含此演示。

## 未验收或尚有限制的部分

- 本轮没有新装一套环境，没有下载完整模型/数据，没有执行百万窗口全量提取、长程训练或完整 LIBERO 闭环成功率评测。
- 上述 GPU 实测使用开发机已有的 DiffSynth 转换 VAE/T5。可移植配置采用公开官方 Wan `.pth`，已检查下载路径、固定 revision 和 FastWAM 原生加载分支，但**没有在本轮下载并实际加载该套默认 VAE/T5**。
- 新机器的 CUDA 驱动、单卡显存、EGL 和包解析需要按[部署指南](portability.md)先跑 smoke 验收。直接依赖已固定版本，传递依赖尚未完全锁定。
- HDF5 初次写入中断、adapter 结构/预处理身份校验及旧多 GPU manager 的边界问题仍见[已知限制](portability.md#已知限制与验收范围)，本轮没有修复这些核心边界。

原始运行产物位于开发机的 `outputs/portable_*`，不随源码上传；表中记录来自实际运行的 summary/metrics。新机器会自行生成对应记录。
