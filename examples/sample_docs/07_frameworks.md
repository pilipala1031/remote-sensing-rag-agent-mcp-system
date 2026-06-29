# 遥感语义分割工具框架与代码实践

本文件介绍遥感语义分割领域常用的深度学习框架、工具库、数据加载方案、预训练权重来源，以及典型代码模板，用于知识库问答和工程实践参考。

## 一、MMSegmentation（OpenMMLab 语义分割框架）

### 1. 概述

- MMSegmentation 是 OpenMMLab 开源的语义分割框架，基于 PyTorch。
- 覆盖 FCN、U-Net、DeepLabV3+、PSPNet、SegFormer、Swin-Transformer、HRNet 等几乎所有主流模型。
- 配置驱动（config-based），通过 YAML/Python 配置文件定义模型/数据/训练全流程。
- GitHub：https://github.com/open-mmlab/mmsegmentation

### 2. 核心概念

- **Config 文件**：继承式配置，base config 定义通用设置，子配置覆盖特定参数。
- **Backbone**：编码器网络（ResNet、Swin-Transformer、MiT 等），从 `mmcls` 或 `mmseg.models.backbones` 加载。
- **Decode Head**：解码器/分类头（ASPPHead、FCNHead、UperHead、SegFormerHead 等）。
- **Auxiliary Head**：辅助分类头，位于 backbone 中间层，提供额外梯度监督。
- **Data Pipeline**：数据预处理流水线（LoadImage → Resize → Flip → Crop → Normalize → Format）。

### 3. 遥感数据集适配

MMSegmentation 默认支持 Cityscapes、ADE20K 等自然图像数据集。适配遥感数据集需要自定义：

```python
# 自定义数据集配置示例（伪代码，示意 config 结构）
dataset_type = 'LoveDADataset'
data_root = 'data/loveda/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375],
    to_rgb=True
)
# LoveDA 有 7 类（含 background），类别色需自定义
classes = ('building', 'road', 'water', 'bareland',
           'forest', 'agriculture', 'background')
palette = [[255,0,0],[128,128,128],[0,0,255],[128,128,0],
           [0,255,0],[0,255,255],[255,255,255]]
```

### 4. 预训练权重来源

- MMSegmentation 官方 Model Zoo 提供所有模型在 Cityscapes/ADE20K 上的 ImageNet 预训练权重。
- 遥感任务通常从这些权重初始化 backbone，再在目标数据集上微调。
- 权重命名规则：`pspnet_r50-d8_512x512_80k_loveda-...`（模型_骨干_输入_迭代_数据集）。

### 5. 典型训练命令

```bash
# 单卡训练
python tools/train.py configs/pspnet/pspnet_r50-d8_512x512_80k_loveda.py

# 多卡训练（4 卡 DistributedDataParallel）
python -m torch.distributed.launch --nproc_per_node=4 \
    tools/train.py configs/deeplabv3plus/... --launcher pytorch

# 推理 + 评测
python tools/test.py configs/... checkpoint.pth --eval mIoU
```

## 二、TorchGeo（遥感专用 PyTorch 扩展库）

### 1. 概述

- TorchGeo 是 PyTorch 官方支持的遥感地理空间数据扩展库。
- 内置 LoveDA、NAIP、Chesapeake、EuroSAT、RESISC45 等数据集的 Dataset 类。
- 提供与 `torchvision.transforms` 兼容的遥感专用数据增强。
- GitHub：https://github.com/microsoft/torchgeo

### 2. 核心组件

- **GeoDataset**：基于地理位置索引的数据集，支持空间查询。
- **NonGeoDataset**：普通数据集（按文件名索引），与 torchvision 类似。
- **Pre-trained Models**：提供 ResNet、Vision Transformer 等在遥感数据集上的预训练权重。
- **Samplers**：按地理区域或固定大小采样（RandomGeoSampler、GridGeoSampler）。

### 3. LoveDA 数据加载示例

```python
# 伪代码示意（非完整可运行代码，展示 API 结构）
from torchgeo.datasets import LoveDA
from torch.utils.data import DataLoader

# 自动下载并加载 LoveDA 数据集
train_dataset = LoveDA(
    root="data/loveda",
    split="train",
    scene="urban",       # 或 "rural"
    transforms=transform_pipeline,
    download=True,
)

# 标准 PyTorch DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
)

# 每个样本返回 dict：{"image": Tensor, "mask": Tensor}
for batch in train_loader:
    images = batch["image"]   # [B, C, H, W]
    masks = batch["mask"]     # [B, H, W]，值为 0~6（类别索引）
```

### 4. 预训练模型

TorchGeo 提供在 GeoMatrix / SatMAE / SSL4EO 等遥感数据集上预训练的 backbone：

```python
# 伪代码示意
from torchgeo.models import resnet50  # 遥感预训练版本
model = resnet50(weights="Sentinel2_ALL_MOCO")  # 加载遥感预训练权重
# 修改最后的分类头适配目标任务类别数
```

## 三、PyTorch 原生训练模板

### 1. 最小化分割训练循环

```python
# 伪代码：核心训练循环结构
model = build_model(num_classes=7)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss(ignore_index=255)

for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.cuda(), masks.cuda()
        outputs = model(images)              # [B, C, H, W]
        loss = criterion(outputs, masks)     # masks: [B, H, W]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    miou = evaluate_miou(model, val_loader, num_classes=7)
    print(f"Epoch {epoch}: val mIoU = {miou:.4f}")
```

### 2. mIoU 计算工具（torchmetrics）

```python
# 伪代码：使用 torchmetrics 计算mIoU
from torchmetrics import JaccardIndex  # 即 IoU

metric = JaccardIndex(
    task="multiclass",
    num_classes=7,
    ignore_index=255,
    average="micro",   # macro = mIoU（各类平均），micro = PA
)
metric = metric.cuda()

for images, masks in val_loader:
    outputs = model(images.cuda())
    preds = outputs.argmax(dim=1)      # [B, H, W]
    metric.update(preds, masks.cuda())

miou = metric.compute()  # 返回 mIoU
per_class_iou = metric_per_class.compute()  # 返回各类 IoU
```

### 3. 混合损失实现

```python
# 伪代码：CE + Dice 混合损失
class CEDiceLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=7):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=255)
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes

    def dice_loss(self, pred, target, smooth=1.0):
        # pred: [B, C, H, W] logits → softmax 概率
        prob = torch.softmax(pred, dim=1)
        # one-hot target
        target_oh = F.one_hot(target, self.num_classes).permute(0,3,1,2).float()
        intersection = (prob * target_oh).sum(dim=(0,2,3))
        union = prob.sum(dim=(0,2,3)) + target_oh.sum(dim=(0,2,3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        return self.alpha * self.ce(pred, target) + \
               self.beta * self.dice_loss(pred, target)
```

## 四、其他常用工具库

### 1. 数据标注与可视化

| 工具 | 用途 | 特点 |
| --- | --- | --- |
| **QGIS** | 遥感影像标注与制图 | 开源 GIS 软件，支持多波段影像查看、矢量/栅格标注、坐标系转换 |
| **labelme** | 多边形标注工具 | 轻量级，支持语义/实例分割 mask 标注，输出 JSON |
| **EO-learn** | 遥感数据处理流水线 | Python 库，封装 Sentinel Hub API，自动化数据获取与预处理 |
| **rasterio** | 栅格影像 I/O | Python 库，读写 GeoTIFF，处理坐标系和重投影 |
| **GDAL** | 遥感影像底层处理 | C++ 库（Python 绑定），格式转换、裁剪、重采样、波段运算 |

### 2. 遥感深度学习专用库

| 库 | 主要功能 | GitHub |
| --- | --- | --- |
| **TorchGeo** | 数据加载 + 预训练模型 | microsoft/torchgeo |
| **rastervision** | 端到端遥感 ML 流水线 | azavea/rastervision |
| **solaris** | 遥感目标检测 + 分割后处理 | CosmiQ/solaris |
| **eo-learn** | 卫星影像数据获取与处理 | sentinel-hub/eo-learn |
| **OpenMMLab** | 通用 CV 框架套件 | open-mmlab（mmseg/mmdet/mmcls） |

### 3. 预训练权重来源

| 来源 | 数据集 | 适用性 |
| --- | --- | --- |
| ImageNet | ImageNet-1k | 通用基线，所有 backbone 默认 |
| MMSeg Model Zoo | Cityscapes / ADE20K | 分割任务专用预训练 |
| TorchGeo Pretrained | Sentinel-2 / NAIP | 遥感域专用，涨点明显 |
| SatMAE / ScaleMAE | fMoW / 多尺度遥感 | 自监督遥感预训练 |
| RemoteCLIP / GeoRSCLIP | 大规模遥感图文对 | 视觉-语言预训练 |

## 五、数据处理流水线

### 1. GeoTIFF → 训练数据

```
原始 GeoTIFF (多波段，带坐标系)
  → rasterio 读取
  → 坐标系转换（如需要，GDAL warp）
  → 波段选择（如 RGBN：选 1,2,3,8 波段）
  → 归一化（或量化到 0-255）
  → 裁剪为固定大小 Patch（512×512 或 1024×1024）
  → 保存为 PNG/NPY/直接内存加载
```

### 2. 标注格式转换

遥感标注常见格式：
- **PNG 索引图**：像素值为类别索引（0~C-1），255 为 ignore_index。LoveDA、Potsdam 使用此格式。
- **COCO JSON**：多边形标注，适用于实例分割（iSAID）。
- **Shapefile / GeoJSON**：矢量标注，需栅格化为 mask。

```python
# 伪代码：Shapefile → PNG Mask 栅格化
import rasterio
from rasterio.features import rasterize

# 读取原始影像的坐标参考
with rasterio.open("image.tif") as src:
    transform = src.transform
    shape = src.shape

# 读取 Shapefile 多边形标注
geometries = [(geom, class_id) for geom, class_id in shapes]

# 栅格化为 mask
mask = rasterize(
    geometries,
    out_shape=shape,
    transform=transform,
    fill=255,          # 背景为 ignore_index
    dtype=np.uint8,
)
# 保存为 PNG
from PIL import Image
Image.fromarray(mask).save("mask.png")
```

## 六、实验管理与可复现性

### 1. 实验记录工具

| 工具 | 功能 | 集成方式 |
| --- | --- | --- |
| **Weights & Biases (wandb)** | 训练曲线、超参搜索、模型对比 | `wandb.init()` → 自动记录 loss/mIoU |
| **TensorBoard** | 训练可视化（loss 曲线、图像预览） | `SummaryWriter` → `add_scalar/add_image` |
| **MLflow** | 实验追踪 + 模型注册 | `mlflow.log_metric()` |

### 2. 可复现性 Checklist

- 固定随机种子（`torch.manual_seed`, `np.random.seed`, `random.seed`）
- 开启 cudnn 确定性模式（`torch.backends.cudnn.deterministic = True`）
- 记录完整的 config 文件（模型/数据/训练/增强）
- 保存每个实验的 checkpoint（不只保存 best）
- 跑 3~5 次取均值 ± 方差，避免单次随机波动误导结论
- 记录硬件信息（GPU 型号、CUDA 版本、PyTorch 版本）
