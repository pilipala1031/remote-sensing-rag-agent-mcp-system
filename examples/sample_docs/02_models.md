# 遥感语义分割常用模型

本文件梳理遥感语义分割领域主流深度学习模型，涵盖架构原理、适用场景、优缺点，用于知识库问答。

## 一、经典自然图像分割模型（遥感中广泛微调使用）

### 1. FCN (Fully Convolutional Network, 2015)

- 核心：把分类网络的最后全连接层换成卷积层，实现端到端像素级分类；用反卷积（转置卷积）上采样恢复分辨率。
- 里程碑：首个真正意义上端到端的语义分割网络。
- 遥感中的局限：感受野有限、上下文建模弱，对大尺度地物边界模糊。
- 论文：Long, Shelhamer, Darrell. "Fully Convolutional Networks for Semantic Segmentation." CVPR 2015.

### 2. U-Net (2015)

- 核心：编码器-解码器对称结构 + skip connection；下采样阶段提取特征，上采样阶段逐步恢复分辨率，跳跃连接拼接低层细节与高层语义。
- 优势：结构简洁、参数量小、对小数据集友好、边界细节保留好。
- 遥感适用：建筑提取、道路提取、农田分割，尤其数据量不大时表现稳健。
- 在遥感分割中是使用频率最高的 baseline 之一。
- 论文：Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.

### 3. DeepLab 系列 (v1~v3+, 2014~2018)

- 核心创新：
  - v1/v2：空洞卷积（Atrous / Dilated Convolution）扩大感受野不降分辨率
  - v2：ASPP（Atrous Spatial Pyramid Pooling）多尺度特征聚合
  - v3：改进 ASPP + 全局平均池化分支
  - v3+：Encoder-Decoder 结构，把 ASPP 输出与低层特征融合
- 优势：多尺度建模强，对大小目标都友好，DeepLabV3+ 是遥感分割最常用的强 baseline。
- 典型 backbone：ResNet-50/101、Xception、HRNet。
- 论文：Chen et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." ECCV 2018.

### 4. PSPNet (Pyramid Scene Parsing Network, 2017)

- 核心：金字塔池化模块（Pyramid Pooling Module），用 1×1、2×2、3×3、6×6 四个尺度的全局先验聚合上下文。
- 优势：全局上下文建模强，对类别混淆（如林地 vs 农业用地）有改善。
- 遥感适用：大场景地表覆盖分类，但对小目标（车辆）细节不如 U-Net。
- 论文：Zhao et al. "Pyramid Scene Parsing Network." CVPR 2017.

## 二、Transformer 与注意力模型

### 5. SegFormer (2021)

- 核心：纯 Transformer encoder（无位置编码，用 3×3 卷积做 overlap patch embedding）+ 轻量 MLP 解码器。
- 优势：无位置编码 → 可适配任意输入分辨率；多尺度特征融合效果好；参数效率高。
- 遥感适用：高分辨率影像、多尺度地物，SegFormer-B0/B2 常用于遥感分割的轻量化部署。
- 论文：Xie et al. "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers." NeurIPS 2021.

### 6. Swin-Transformer + UperHead

- 核心：Swin 用移位窗口（Shifted Window）做局部自注意力，线性复杂度；配合 UperHead 或 SegFormer 风格解码器。
- 优势：层次化特征表达强，在 ImageNet 与 ADE20K 上都是 SOTA，迁移到遥感普遍优于纯 CNN。
- 遥感适用：复杂城市场景、多类地物，是当前学术 SOTA 常选 backbone。
- 论文：Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." ICCV 2021.

### 7. TransUNet (2021)

- 核心：CNN encoder 提取局部细节 + Transformer encoder 建模全局依赖 + U-Net 风格解码器。
- 优势：兼具 CNN 的局部归纳偏置与 Transformer 的全局建模能力。
- 遥感适用：医学影像与遥感小目标分割，常作为对比实验 baseline。
- 论文：Chen et al. "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation." arXiv 2021.

## 三、遥感专用模型

### 8. FAR (Flexible, Accurate, Robust, 2020)

- 针对 LoveDA 等高分遥感数据设计，强化跨域泛化与边界精度。

### 9. ABCNet / Mask R-CNN 遥感实例分割

- 用于 iSAID 等实例分割任务，对密集小目标（船只、车辆）检测 + 分割。

### 10. 遥感 backbone 预训练趋势

- RemoteCLIP / GeoRSCLIP：遥感视觉-语言预训练，可用于下游分割的特征提取。
- SatMAE / ScaleMAE：基于 MAE 的遥感自监督预训练，利用多光谱与多尺度特性。

## 四、模型选择工程建议

| 场景 | 推荐模型 | 理由 |
| --- | --- | --- |
| 小数据集快速起步 | U-Net | 参数少、收敛稳 |
| 强 baseline / 投论文 | DeepLabV3+ (ResNet-101) | 多尺度强、调参成熟 |
| 追求 SOTA | Swin-Transformer + UperHead | 全局建模 + 层次特征 |
| 轻量化部署 | SegFormer-B0 / MobileNet + DeepLabV3 | 参数 <10M，端侧可跑 |
| 跨域泛化 | FAR / 加入 domain randomization | 应对 LoveDA Urban→Rural 场景迁移 |
