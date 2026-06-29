# 遥感语义分割前沿趋势（2022-2025）

本文件梳理遥感语义分割领域近年前沿方向，涵盖基础模型、自监督预训练、视觉-语言模型、SAM 系列在遥感中的应用，用于知识库问答和了解领域最新进展。

## 一、遥感基础模型（Foundation Models for RS）

### 1. SatMAE (2022)

- 全称：Satellite Masked Autoencoder
- 核心思想：将 MAE 自监督预训练范式应用于多光谱遥感影像，利用波段信息设计专用的 patch embedding。
- 关键创新：
  - 多光谱通道组嵌入：将不同波段分组，每组独立做 patch embedding
  - 时序扩展（SatMAE++）：支持多时相影像的时序编码
- 预训练数据：fMoW（Functional Map of the World），约 100 万张多时相多光谱影像
- 下游效果：在 LoveDA、EuroSAT 等遥感分类/分割任务上，比 ImageNet 预训练涨 2~5 mIoU
- 论文：Cong et al. "SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery." NeurIPS 2022.

### 2. ScaleMAE (2023)

- 核心思想：在 SatMAE 基础上引入多尺度感知，使模型对不同地面分辨率（GSD）的影像有自适应能力。
- 关键创新：
  - 输入影像附带 GSD 元信息，模型根据 GSD 调整感受野
  - 多尺度 patch embedding：不同分辨率用不同 patch 大小
- 优势：当训练域与目标域分辨率差异大时（如 0.3m → 0.5m），ScaleMAE 比 SatMAE 泛化更好
- 论文：Reed et al. "ScaleMAE: A Scale-Aware Masked Autoencoder for Multiscale Geospatial Representation Learning." ICML 2023.

### 3. RingMo (2022) / RingMo-Lite (2023)

- 来源：中国电子科技大学 / 华为联合发布
- 核心思想：面向遥感影像的掩码图像建模（MIM）自监督框架，针对遥感影像特性优化：
  - 遥感影像目标密集且尺度变化大 → 设计密集掩码策略
  - 遥感影像背景复杂 → 引入对比学习分支增强区分度
- 预训练数据：200 万张遥感影像（含光学 + SAR）
- RingMo-Lite：面向端侧部署的轻量化版本
- 下游效果：在多个遥感分割/检测基准上超越 ImageNet 预训练 3~6%
- 论文：Sun et al. "RingMo: A Remote Sensing Foundation Model with Masked Image Modeling." IEEE TGRS 2022.

### 4. SpectralGPT (2023)

- 核心思想：专为高光谱遥感影像设计的基础模型，用 3D 掩码自编码器学习光谱-空间联合特征。
- 关键创新：
  - 3D patch embedding：同时建模空间维度和光谱维度
  - 高光谱序列掩码策略：随机遮掩整个波段组
- 优势：在高光谱分类任务上显著优于 2D 方法，能更好地利用光谱特征区分细粒度地物
- 论文：Zhang et al. "SpectralGPT: Spectral Foundation Model." IEEE TPAMI 2024.

## 二、SAM（Segment Anything Model）在遥感中的应用

### 1. SAM 原始模型 (Meta, 2023)

- 核心能力：基于提示（prompt）的通用分割——输入点、框或文本提示，模型分割出对应区域。
- 架构：
  - Image Encoder：ViT-H/L/B（基于 MAE 预训练）
  - Prompt Encoder：位置编码 + CLIP 文本嵌入
  - Mask Decoder：轻量双向 Transformer，输出多个候选 mask
- 训练数据：SA-1B 数据集，10 亿+ mask 标注
- 在自然图像上表现极强，但直接迁移到遥感影像效果有限（训练数据无遥感场景）

### 2. SAM 遥感适配方案

#### RSPrompter (2023)

- 思路：为遥感影像自动生成 SAM 的 prompt（点/框），实现遥感场景的自动语义分割。
- 架构：在 SAM 基础上加一个 prompt 生成网络，自动预测目标位置和类别。
- 优势：不需要人工点击提示，可实现端到端遥感分割。
- 论文：Chen et al. "RSPrompter: Prompting SAM for Remote Sensing Instance Segmentation." arXiv 2023.

#### SAMRS (2024)

- 思路：在遥感数据集（iSAID + DOTA）上微调 SAM，使其适应遥感目标。
- 关键点：遥感的"任意方向旋转"目标与 SAM 训练数据差异大，需旋转增强和遥感域适配。
- 效果：在 iSAID 上超越原始 SAM 的实例分割 mAP 约 10~15 个百分点。

#### SAM 零样本能力评估

| 任务 | SAM 原始模型表现 | 遥感适配后 |
| --- | --- | --- |
| 建筑物提取 | 中等（边界粗） | 优秀（RSPrompter 等） |
| 道路提取 | 差（细长结构不连续） | 中等（需专门微调） |
| 水体分割 | 良好 | 优秀 |
| 船只实例分割 | 差（小目标） | 优秀（SAMRS） |
| 农田分割 | 良好 | 优秀 |

### 3. SAM 在遥感分割中的局限

- **语义标注缺失**：SAM 是"类别无关"的（只分割不分类），语义分割需要额外分类模块。
- **小目标困难**：船只、车辆等极小目标 SAM 容易漏检。
- **线性结构不连续**：道路、河流等细长结构 SAM 分割结果常断裂。
- **计算成本**：ViT-H 编码器推理慢、显存大，不适合实时大图处理。

## 三、视觉-语言模型（VLM）在遥感中的应用

### 1. RemoteCLIP (2023)

- 思路：将 CLIP 范式迁移到遥感领域，构建遥感视觉-语言基础模型。
- 方法：收集 80 万+ 遥感图文对，对 CLIP 做遥感域微调。
- 下游能力：
  - 零样本遥感场景分类
  - 遥感图文检索
  - 作为分割模型的编码器（提供强视觉特征）
- 论文：Liu et al. "RemoteCLIP: A Vision-Language Foundation Model for Remote Sensing." arXiv 2023.

### 2. GeoRSCLIP (2024)

- 思路：比 RemoteCLIP 更大规模的遥感 CLIP，预训练数据超过 150 万图文对。
- 优势：在更多遥感下游任务上取得 SOTA 零样本性能。
- 特点：引入遥感专用文本增强策略（如"高分卫星影像中的..."前缀模板）。

### 3. 遥感 VLM 助手（LLM + 视觉编码器）

| 模型 | 架构 | 能力 |
| --- | --- | --- |
| **GeoGPT** | LLM + 遥感工具链 | 遥感问答 + 工具调用 |
| **EarthGPT** | LLM + 多模态编码器 | 遥感影像理解 + 问答 + 检测引导 |
| **GeoLLaVA** | LLaVA 架构 + 遥感数据 | 遥感视觉问答、场景描述 |
| **RS-GPT4V** | GPT-4V + 遥感提示工程 | 遥感影像细粒度描述 |

这些模型代表了遥感 AI 从"纯视觉"向"视觉-语言-推理"融合演进的趋势。它们的能力目前主要集中在场景描述和视觉问答，在像素级分割任务上仍依赖传统分割模型。

## 四、Diffusion 模型在遥感分割中的应用

### 1. 分割即去噪

- 思路：将语义分割建模为条件去噪过程。给定影像作为条件，从随机噪声逐步去噪生成分割 mask。
- 代表方法：DiffuSeg、DDP（Diffusion-based Decision Process）
- 优势：对边界不确定性建模更好，适合类别边界模糊的场景

### 2. 数据增强用 Diffusion

- 用 Stable Diffusion 或遥感专用 Diffusion 模型生成合成数据
- 优势：可以生成稀有类别的合成影像（如罕见的地物组合），缓解类别不平衡
- 效果：在 LoveDA 等数据集上，Diffusion 增强比传统增强提升小类 IoU 约 2~4%

## 五、自监督预训练（SSL）在遥感中的进展

### 1. 为什么遥感需要专用 SSL

- ImageNet 预训练权重在遥感域存在域差距：
  - ImageNet 是自然场景（水平拍摄），遥感是俯视
  - ImageNet 是 RGB 3 通道，遥感常有多光谱/高光谱
  - ImageNet 目标居中且方向固定，遥感目标方向任意
- 遥感专用 SSL 能利用海量无标注卫星影像（Sentinel-2、Landsat 等公开数据）

### 2. 主流遥感 SSL 方法对比

| 方法 | 范式 | 预训练数据 | 特点 |
| --- | --- | --- | --- |
| **SatMAE** | MAE（掩码自编码器） | fMoW | 多光谱感知 |
| **ScaleMAE** | MAE + 尺度 | fMoW | GSD 自适应 |
| **RingMo** | MIM + 对比学习 | 200 万遥感影像 | 密集掩码 + 对比分支 |
| **MoCo-Cx** | 对比学习 | Sentinel-2 | 多时相对比 |
| **SeCo** | 对比学习 | Sentinel-2 | 季节对比 |
| **SSL4EO** | MAE / 对比 | 150 万 Sentinel-2 | 大规模标准化基准 |

### 3. SSL 预训练对分割的效果

以 LoveDA 数据集为例（DeepLabV3+ backbone）：

| 预训练方式 | Urban mIoU | Rural mIoU | 相比 ImageNet |
| --- | --- | --- | --- |
| ImageNet | 64.45% | 50.53% | 基准 |
| SatMAE | 66.82% | 53.17% | +2~3% |
| RingMo | 67.15% | 53.89% | +3~4% |
| SSL4EO-S12 | 66.98% | 53.45% | +2~3% |

说明：遥感 SSL 预训练在高分辨率数据（如 LoveDA 0.3m）上的提升不如在中分辨率数据（Sentinel-2 10m）上显著，因为高分影像与 SSL 常用预训练数据（多为 10m 级）存在分辨率差距。

## 六、未来方向展望

### 1. 多模态融合

- RGB + DSM + 多光谱 + SAR → 多模态基础模型
- 挑战：不同模态的空间分辨率、光谱范围差异大，对齐和融合策略是关键

### 2. 开放词汇分割（Open-Vocabulary Segmentation）

- 传统分割：类别固定（如 LoveDA 7 类）
- 开放词汇：用文本描述任意类别（如"风力发电机""油罐"），模型分割出对应区域
- 依赖：VLM（如 RemoteCLIP/GeoRSCLIP）的视觉-语言对齐能力
- 前景：让遥感分割不再受预定义类别限制

### 3. Agent + 工具链（Agentic RAG）

- 趋势：LLM Agent 自主调度分割/检测/变化检测等工具
- 本项目（Remote Sensing RAG）正是这一方向的实践：Agent 根据问题决定是否调用知识库检索工具
- 未来：Agent 可直接调用分割模型 API，实现"自然语言 → 遥感分析"的端到端交互

### 4. 实时边缘部署

- 方向：轻量模型 + INT8 量化 + 硬件加速
- 目标：在无人机/卫星星载计算机上实时分割
- 挑战：算力受限（<10W），延迟要求 <100ms
