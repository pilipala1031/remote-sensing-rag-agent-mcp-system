# 遥感语义分割训练配方与实战指南

本文件系统整理遥感语义分割任务中常用的训练超参数配置、数据增强策略、损失函数选择、学习率调度方案，所有配置基于主流论文和开源框架（MMSegmentation、PyTorch Lightning）的通用实践，用于知识库问答。

## 一、典型训练超参数配置

### 1. 通用配置模板（适用于大多数遥感分割任务）

| 超参数 | 推荐值 | 说明 |
| --- | --- | --- |
| 优化器 | SGD (momentum=0.9) 或 AdamW | SGD 泛化更好但需调 LR；AdamW 收敛快 |
| 初始学习率 | 0.01 (SGD) / 0.0001~0.0006 (AdamW) | 配合 poly 或 cosine 衰减 |
| 学习率调度 | Polynomial decay (power=0.9) 或 Cosine annealing | DeepLab 系列默认 poly |
| 权重衰减 | 0.0001~0.0005 | 防过拟合，AdamW 可取 0.01 |
| Batch size | 8~16（单卡 512×512） | 显存不够时用 Gradient Accumulation |
| 训练轮数 | 80~160 epochs | 大数据集（DeepGlobe）可减少到 40~80 |
| 输入尺寸 | 512×512 或 1024×1024 | 大图裁剪训练 |
| 预训练 | ImageNet 预训练 backbone | 几乎所有方案均从 ImageNet 权重初始化 |

### 2. 各模型典型配置参考

#### U-Net (ResNet-50) on LoveDA

```
优化器：SGD，momentum=0.9，weight_decay=5e-4
学习率：0.01，poly decay (power=0.9)
Batch size：16，输入 512×512
Epoch：100
损失函数：CrossEntropyLoss
预期 mIoU：~61%
```

#### DeepLabV3+ (ResNet-101) on Potsdam

```
优化器：SGD，momentum=0.9，weight_decay=1e-4
学习率：0.01，poly decay (power=0.9)
Batch size：8，输入 512×512（裁剪自 6000×6000）
Epoch：80~120
损失函数：CrossEntropyLoss + OHEM（或 CE + Dice 混合）
辅助损失头：aux_loss_weight=0.4（DeepLabV3+ 有辅助分类头）
预期 mIoU：~74%
```

#### SegFormer (MiT-B2) on LoveDA

```
优化器：AdamW，weight_decay=0.01
学习率：0.00006，cosine annealing
Batch size：8~16，输入 512×512
Epoch：160（Transformer 需要更多 epoch 收敛）
学习率预热：前 1500 iter 线性预热
损失函数：CrossEntropyLoss
预期 mIoU：~68%
```

#### Swin-Transformer (Swin-L + UperHead) on Potsdam

```
优化器：AdamW，weight_decay=0.01
学习率：0.00006，cosine annealing
Batch size：8（Swin-L 显存占用大，可能需 4 卡）
Epoch：160~220
学习率预热：前 1500~3000 iter 线性预热
损失函数：CrossEntropyLoss（辅助头 + 主头，比例 0.4:1.0）
预期 mIoU：~79%
```

## 二、数据增强策略详解

### 1. 标准增强组合（几乎所有任务必备）

- **随机水平翻转**（Random Horizontal Flip）：概率 0.5
- **随机垂直翻转**（Random Vertical Flip）：概率 0.5
- **随机缩放裁剪**（Random Resize Crop）：Scale ratio 0.5~2.0，裁剪到目标尺寸
- **随机旋转**（Random Rotation）：0/90/180/270 度随机（遥感影像无固定方向）
- **归一化**（Normalize）：ImageNet 均值方差（mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]）

### 2. 进阶增强（提升泛化，涨点 1~3% mIoU）

- **颜色抖动**（Color Jitter）：亮度/对比度/饱和度 ±0.2~0.4
- **CutMix**：两张影像及其标签按随机区域混合，比例 50%
- **MixUp**：两张影像按 alpha 比例叠加，分割中较少用但有报告称有效
- **随机擦除**（Random Erasing）：模拟遮挡，概率 0.1~0.2
- **高斯噪声 / 模糊**（Gaussian Noise / Blur）：增强对传感器噪声的鲁棒性

### 3. 遥感专用增强

- **多光谱通道增强**：对 NIR / SWIR 通道单独做颜色抖动
- **直方图均衡**（CLAHE）：增强低对比度影像细节
- **阴影模拟**：随机区域加暗 mask，模拟建筑阴影遮挡
- **波段交换**（Band Swapping）：RGB → BGR 或随机波段排列（谨慎使用）

### 4. 测试时增强（TTA）

TTA 是推理阶段零成本涨点手段，典型组合：

- 水平翻转 + 垂直翻转 + 原图（3 倍推理）
- 或再叠加 90/180/270 度旋转（8 倍推理）
- 最终输出取所有增强版本的 softmax 平均

效果：通常涨 1~3 mIoU，竞赛中几乎必用。

## 三、损失函数选择指南

### 1. 常用损失函数对比

| 损失函数 | 公式核心 | 适用场景 | 参数 |
| --- | --- | --- | --- |
| CrossEntropy (CE) | -Σ y log(p) | 通用基线 | 无 |
| Weighted CE | -Σ w_i · y_i log(p_i) | 类别不平衡 | weight 按频率倒数 |
| Focal Loss | -(1-p_t)^γ · log(p_t) | 难分样本/小类 | γ=2, α=0.25 |
| Dice Loss | 1 - 2|X∩Y|/(|X|+|Y|) | 小目标/区域优化 | smooth=1 |
| Tversky Loss | Dice 变体，FP/FN 可调 | 偏向召回 | α=0.3, β=0.7 |
| Lovász Loss | 直接优化 IoU 的凸代理 | 提升 mIoU | 无 |
| Boundary Loss | 边界距离加权惩罚 | 边界敏感任务 | δ=2~5 |

### 2. 混合损失实践（最常用方案）

**CE + Dice 混合**是分割任务的标准配置：

```python
# 伪代码
loss = α * ce_loss + β * dice_loss
# 常用配置：α=1.0, β=1.0（等权）
# 或：α=1.0, β=0.5（CE 主导）
```

原理：CE 提供像素级梯度（收敛稳定），Dice 直接优化 IoU（对小类友好）。两者互补。

**CE + Focal 混合**用于极端不平衡场景：

```python
loss = α * ce_loss + β * focal_loss(γ=2)
# 常用配置：α=0.5, β=0.5
```

### 3. 辅助损失头

DeepLabV3+、PSPNet、SegFormer 等模型有辅助分类头（auxiliary head）：

- 辅助头在 encoder 中间层接一个分类器
- 总损失 = 主头损失 + λ × 辅助头损失
- λ 通常取 0.3~0.4
- 作用：提供额外梯度监督，加速收敛、稳定训练

## 四、学习率调度方案

### 1. Polynomial Decay（DeepLab 系列默认）

```
lr = base_lr × (1 - current_iter / max_iter)^power
# power = 0.9 是经典设置
```

特点：前期下降慢、后期快速趋近 0，适合固定 epoch 训练。MMSegmentation 默认方案。

### 2. Cosine Annealing（Transformer 模型常用）

```
lr = base_lr × 0.5 × (1 + cos(π × current_iter / max_iter))
```

特点：平滑衰减，配合 warm-up 使用。SegFormer、Swin-Transformer 默认方案。

### 3. Warm-up 策略

Transformer 模型训练初期不稳定，必须 warm-up：

```
前 N 步：lr 从 0 线性增长到 base_lr
之后：按 poly 或 cosine 衰减
# N 通常取 1500~3000 步（约前 1~2 个 epoch）
```

CNN 模型（U-Net、DeepLabV3+）通常不需要 warm-up。

## 五、类别不平衡处理实战

### 1. 统计类别频率

训练前必须统计每个类别的像素占比，用于确定加权策略：

```
LoveDA 典型类别像素占比：
- background: ~30%（视场景而定）
- building: ~20%（Urban 域主导）
- road: ~15%
- forest: ~12%
- agriculture: ~10%
- water: ~8%
- bareland: ~5%
```

### 2. 加权策略选择

| 策略 | 实现 | 效果 |
| --- | --- | --- |
| 频率倒数加权 | w_i = 1/freq_i | 过度惩罚大类，需平滑 |
| 中值频率加权 | w_i = median_freq / freq_i | 较温和，实践中最常用 |
| 有效样本数加权 | w_i = (1-β^n) / (1-β) | β=0.999，来自 Google 论文 |

### 3. Focal Loss 配置

```
Focal Loss:
  γ=2（标准设置，控制难易样本权重）
  α=0.25（前景类权重）
  效果：对 LoveDA 的小类（bareland、water）IoU 提升 2~5%
  注意：γ 太大会导致训练不稳定，建议 γ=1~2 之间
```

## 六、训练监控与调优 Checklist

### 1. 关键监控指标

- **训练 mIoU vs 验证 mIoU**：如果训练远高于验证 → 过拟合，需要更强增强
- **per-class IoU**：定位短板类别（如 water IoU 突然下降 → 检查数据标注）
- **梯度范数**：如果梯度爆炸/消失，调整 LR 或加 gradient clip
- **loss 曲线**：CE loss 应单调下降并收敛；Dice loss 初期可能震荡，正常

### 2. 调优优先级（性价比从高到低）

1. **数据增强**（翻转+旋转+缩放）：几乎零成本，+2~5% mIoU
2. **损失函数**（CE→CE+Dice 混合）：改一行代码，+1~3% mIoU
3. **Backbone 升级**（ResNet-50→ResNet-101 或 Swin-T→Swin-L）：+1~3% mIoU
4. **多尺度 TTA**：推理阶段，+1~3% mIoU
5. **类别加权/Focal Loss**：+1~2% mIoU（仅对小类有效）
6. **遥感预训练权重**：+2~5% mIoU（如有公开权重）
7. **模型集成**：+1~2% mIoU（竞赛用，工程慎用）

### 3. 常见训练问题排查

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| mIoU 不收敛 | LR 太大/太小 | 先用 0.01(SGD)/0.0001(AdamW) 试探 |
| 小类 IoU 接近 0 | 类别不平衡 | 加 Focal/Dice Loss + 类别加权 |
| 训练正常但验证差 | 过拟合 | 加强增强、加 dropout、减小模型 |
| Transformer 训练发散 | 缺 warm-up | 加 1500 步线性预热 |
| 显存不足 | batch/输入太大 | 降 batch、用 gradient accumulation |
| 边界处锯齿严重 | 缺边界监督 | 加 Boundary Loss 或 CRF 后处理 |
