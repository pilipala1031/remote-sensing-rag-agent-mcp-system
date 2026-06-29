# 遥感图像其他深度学习任务概览

本文件介绍遥感语义分割之外的其他主流深度学习任务，涵盖目标检测、变化检测、超分辨率重建、云检测与去云、SAR 图像分割、高光谱分类等，用于知识库问答和拓宽领域视野。

## 一、遥感目标检测（Object Detection）

### 1. 任务定义

- 在遥感影像中检测目标的位置（旋转框或水平框）并分类
- 与语义分割的区别：检测输出是"框 + 类别"，分割输出是"像素级 mask"
- 典型应用：飞机检测、船只检测、车辆检测、建筑检测

### 2. 核心数据集

#### DOTA（A Dataset for Object Detection in Aerial Images）

- 来源：武汉大学 CAPTAIN 实验室，2018 年
- 影像来源：Google Earth、JL-1 卫星、GF-2 卫星
- 空间分辨率：0.1~1.0 m 不等
- 影像数量：2806 张大尺寸影像（平均 3000×3000）
- 类别数：15 类（含飞机、船只、车辆、桥梁、储罐等）
- 特点：使用**旋转框标注**（Oriented Bounding Box, OBB），目标方向任意
- 评价指标：mAP@0.5
- 版本：DOTA-v1.0 / v1.5 / v2.0（v2.0 扩展到 18 类）
- 官方地址：https://captain-whu.github.io/DOTA/dataset.html

#### DIOR

- 来源：武汉大学，2019 年
- 影像数量：23463 张（水平框标注）
- 类别数：20 类
- 特点：影像尺寸固定 800×800，适合快速实验
- 评价指标：mAP@0.5

### 3. 主流检测模型

| 模型 | 类型 | 框格式 | 特点 |
| --- | --- | --- | --- |
| **Faster R-CNN** | 两阶段 | 水平框 | 经典基线，精度高但速度慢 |
| **RetinaNet** | 单阶段 | 水平框 | Focal Loss 解决类别不平衡 |
| **FCOS** | 无锚框 | 水平框 | Anchor-free，简洁高效 |
| **Rotated RetinaNet** | 单阶段 | 旋转框 | 适配遥感任意方向目标 |
| **Rotated Faster R-CNN** | 两阶段 | 旋转框 | 旋转框版本的 Faster R-CNN |
| **Oriented R-CNN** | 两阶段 | 旋转框 | 高性能旋转检测 |
| **Gliding Vertex** | 单/双阶段 | 旋转框 | 滑动顶点回归旋转框 |

### 4. 遥感检测特有难点

- **目标方向任意**：遥感俯视，目标方向无固定假设 → 需旋转框或旋转不变性设计
- **目标尺度极端**：同一影像中船只可能 20 像素也可能 2000 像素 → 多尺度特征金字塔必需
- **密集小目标**：停车场中车辆紧密排列 → NMS 后处理易漏检，需 Soft-NMS 或 Repulsion Loss
- **大图推理**：单张影像可能 3000×3000 → 需切图检测 + 结果合并

## 二、变化检测（Change Detection）

### 1. 任务定义

- 输入：同一地点不同时间（T1, T2）的两张影像
- 输出：变化区域 mask（二值或多类变化）
- 典型应用：城市扩张监测、违建检测、灾后损毁评估、森林砍伐监测

### 2. 核心数据集

| 数据集 | 影像对数 | 分辨率 | 类别 | 特点 |
| --- | --- | --- | --- | --- |
| **LEVIR-CD** | 637 对 | 0.5m | 二值（变/不变） | 建筑/道路变化，Google Earth |
| **WHU-CD** | 1 对大图 | 0.2m | 二值 | 单张超大影像（32507×15354） |
| **DSIFN-CD** | 360 对 | 0.5m | 二值 | 多场景变化 |
| **CLCD** | 600 对 | 多分辨率 | 6 类变化 | 中国土地利用变化 |

### 3. 主流变化检测模型

#### Siamese 网络（孪生网络）架构

- 核心思想：共享权重的双分支编码器分别提取 T1、T2 特征，在特征层面做差异计算
- 代表模型：

| 模型 | 核心设计 | 论文 |
| --- | --- | --- |
| **FC-Siam-diff** | U-Net 编码器 + 特征差分 | Daudt et al., 2018 |
| **FC-EF**（Early Fusion） | 输入层拼接 T1+T2 | Daudt et al., 2018 |
| **STANet** | 空间-时间注意力 | Chen et al., 2020 |
| **BIT-CD** | Transformer 二值变化检测 | Chen et al., 2021 |
| **ChangeFormer** | 纯 Transformer 孪生网络 | Bondi et al., 2022 |

#### 评价指标

- Precision、Recall、F1（变化检测主要用 F1）
- IoU（变化区域）
- OA（Overall Accuracy，但变化检测中不推荐单独使用，因不变区域占 95%+）

### 4. 变化检测特有难点

- **配准误差**：T1/T2 影像如果未精确对齐（geometric misalignment），会产生伪变化
- **季节/光照差异**：同地点不同季节的植被颜色变化会被误判为真实变化
- **变化极其稀疏**：变化区域通常 <5% 像素，正负样本极度不平衡
- **多类变化**：不仅要检测"变了"，还要分类"从什么变成了什么"

## 三、超分辨率重建（Super-Resolution）

### 1. 任务定义

- 输入：低分辨率（LR）遥感影像
- 输出：高分辨率（HR）影像，恢复细节纹理
- 典型应用：历史低分影像增强、压缩数据还原、下游任务预处理

### 2. 主流方法

| 方法类型 | 代表模型 | 特点 |
| --- | --- | --- |
| **插值法** | 双三次插值（Bicubic） | 最快，但效果差，适合基线 |
| **CNN-based** | SRCNN、EDSR、RCAN | 早期深度方法，×2/×4 放大 |
| **GAN-based** | SRGAN、ESRGAN | 生成对抗网络，纹理更真实 |
| **Transformer-based** | SwinIR、HAT | 当前 SOTA，长程依赖建模强 |
| **Diffusion-based** | SR3、StableSR | 生成质量最高，但速度慢 |

### 3. 遥感超分专用模型

- **LGCNet**（Local and Global Context Network）：针对遥感大尺度场景设计
- **HSENet**（Hybrid-Scale Self-Similarity）：利用遥感影像中的多尺度自相似性
- **TransENet**（Transformer Enhanced Network）：Transformer + CNN 混合

### 4. 评价指标

| 指标 | 全称 | 含义 | 注意 |
| --- | --- | --- | --- |
| PSNR | Peak Signal-to-Noise Ratio | 峰值信噪比（dB） | 最常用，越高越好 |
| SSIM | Structural Similarity | 结构相似性 | 比PSNR更符合人眼感知 |
| LPIPS | Learned Perceptual Image Patch Similarity | 感知距离（基于深度特征） | 越低越相似，GAN/Diffusion 模型常用 |

## 四、云检测与去云（Cloud Detection & Removal）

### 1. 云检测（Cloud Detection / Cloud Segmentation）

- 任务：识别遥感影像中的云层区域（像素级二值或多类分割）
- 难点：
  - 薄云（卷云）与地物光谱相近，边界模糊
  - 雪地、白色屋顶易与云混淆
  - 云阴影检测：云的投影位置依赖太阳角度和云高度
- 主流数据集：Cloud38（38 类场景）、WHU-OPT-SAR（含云标注）
- 主流模型：U-Net 变体、Cloud-Net、Multi-Scale Cloud Detection Network
- 评价指标：mIoU、Precision、Recall（针对云区域）

### 2. 去云（Cloud Removal / Cloud Compensation）

- 任务：恢复被云遮挡区域的地面信息
- 输入：含云的光学影像（+可选 SAR 辅助）
- 输出：无云的光学影像
- 方法：
  - **时序补偿**：利用同地点多时相无云影像填充云区域
  - **SAR 引导**：SAR 穿透云层，用 SAR 引导光学影像恢复（如 Sar2Optical）
  - **生成模型**：GAN/Diffusion 直接生成云下地物

## 五、SAR 图像分割

### 1. SAR（合成孔径雷达）影像特点

- 主动式微波遥感，全天候全天时成像
- 与光学影像差异巨大：
  - 灰度影像（单通道强度图）
  - 固有散斑噪声（Speckle Noise），类似椒盐噪声但乘性
  - 几何畸变（叠掩、阴影）
  - 地物表现与光学完全不同（金属屋顶高亮、水面镜面反射为暗）

### 2. SAR 语义分割任务

- 典型任务：水域提取、建筑区提取、耕地分类、海冰分类、溢油检测
- 核心难点：
  - 散斑噪声严重 → 需先滤波（Lee Filter / Frost Filter）或训练时增强
  - 标注稀缺 → SAR 专家标注成本极高
  - 可解释性差 → 模型错误难调试

### 3. 主流数据集与模型

| 数据集 | 任务 | 规模 | 特点 |
| --- | --- | --- | --- |
| **OpenSARUrban** | 城市分割 | 多城市 Sentinel-1 | 开放城市 SAR 数据集 |
| **SAR-Ship-Dataset** | 船只检测/分割 | 多传感器 | 海面船只目标 |

- 模型：U-Net 仍是最常用基线（SAR 灰度图与 U-Net 的医学图像特性相似）
- SAR 专用预处理：Lee 滤波、子孔径分解、极化特征提取

## 六、高光谱分类（Hyperspectral Image Classification）

### 1. 任务定义

- 输入：高光谱影像（HSI），通常 100~400 个窄波段，空间分辨率低（20~30m）
- 输出：像素级地物分类
- 与普通语义分割区别：利用数百维光谱信息区分细粒度地物（如不同作物品种）

### 2. 核心数据集

| 数据集 | 传感器 | 波段数 | 类别 | 空间尺寸 |
| --- | --- | --- | --- | --- |
| **Indian Pines** | AVIRIS | 200 | 16 | 145×145 |
| **Pavia University** | ROSIS | 103 | 9 | 610×340 |
| **Salinas** | AVIRIS | 204 | 16 | 512×217 |
| **Houston 2013** | ITRES CASI-1500 | 144 | 15 | 多场景 |

### 3. 主流方法演进

| 时代 | 方法 | 特点 |
| --- | --- | --- |
| 传统 | SVM、随机森林、KNN | 光谱特征 + 少量空间特征 |
| 深度学习早期 | 1D-CNN（光谱维度卷积） | 只利用光谱 |
| 空间-光谱联合 | 2D-CNN / 3D-CNN | 同时建模光谱和空间 |
| Transformer | Spectral Transformer / SSFTT | 光谱序列建模 |
| 基础模型 | SpectralGPT | 大规模预训练，零样本分类 |

### 4. 高光谱分类特有难点

- **维度灾难**：波段数（200+）远超训练样本数 → 需 PCA 降维或正则化
- **训练样本极少**：Indian Pines 总共才 10249 个标注像素，需 few-shot 策略
- **波段冗余**：相邻波段高度相关 → 可用 PCA / 选择性波段方法降维
- **空间分辨率低**：每个像素可能是混合地物（混合像素问题）

## 七、遥感任务间的关联与统一

### 1. 任务间的共享技术

| 技术 | 语义分割 | 目标检测 | 变化检测 | 超分辨率 |
| --- | --- | --- | --- | --- |
| U-Net / Encoder-Decoder | ✅ 核心 | ✅ Mask R-CNN | ✅ FC-Siam | ✅ |
| FPN 特征金字塔 | ✅ | ✅ 核心 | ✅ | ✅ |
| Transformer backbone | ✅ Swin/SegFormer | ✅ DETR | ✅ ChangeFormer | ✅ SwinIR |
| 多尺度推理 | ✅ | ✅ | ✅ | - |
| 滑窗大图推理 | ✅ | ✅ | ✅ | ✅ |

### 2. 多任务统一趋势

- **Swin-Transformer**：同一 backbone 可适配分割（UperHead）、检测（Cascade Mask R-CNN）、分类
- **基础模型**：一个预训练 backbone 在多任务上微调（如 RingMo → 分割 + 检测 + 分类）
- **VLM 统一**：RemoteCLIP 类模型可同时做检索、分类、分割引导

### 3. 各任务的典型选型建议

| 如果你的需求 | 推荐任务 | 推荐模型/方法 |
| --- | --- | --- |
| 像素级地物分类 | 语义分割 | DeepLabV3+ / SegFormer |
| 目标定位计数 | 目标检测 | Rotated RetinaNet / Oriented R-CNN |
| 监测变化区域 | 变化检测 | ChangeFormer / BIT-CD |
| 提升影像分辨率 | 超分辨率 | SwinIR / ESRGAN |
| 去除云遮挡 | 去云 | SAR 引导 / 时序补偿 |
| 水域/建筑区提取(SAR) | SAR 分割 | U-Net + Lee 滤波 |
| 精细作物分类 | 高光谱分类 | 3D-CNN / SpectralGPT |
