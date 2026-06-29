# 遥感分割模型部署与工程化

本文件介绍遥感语义分割模型从训练完成到生产部署的全流程工程实践，涵盖模型导出、推理优化、大图推理方案、显存管理、速度基准测试等，用于知识库问答和工程落地参考。

## 一、模型导出：PyTorch → ONNX

### 1. ONNX 导出基础

ONNX（Open Neural Network Exchange）是跨框架模型交换的中间表示格式。将 PyTorch 模型导出为 ONNX 后，可在 ONNX Runtime / TensorRT / OpenVINO 等推理引擎上运行。

```python
# 伪代码：PyTorch 模型导出为 ONNX
import torch

model = build_model(num_classes=7)
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

dummy_input = torch.randn(1, 3, 512, 512)  # 固定输入尺寸

torch.onnx.export(
    model,
    dummy_input,
    "deeplabv3plus.onnx",
    opset_version=14,           # 推荐使用 opset 14+
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={              # 支持动态 batch/尺寸
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
)
```

### 2. 导出常见问题

| 问题 | 原因 | 解决方案 |
| --- | --- | --- |
| ONNX 导出报错 | 模型含不支持的操作 | 替换为 ONNX 支持的算子，或升级 opset 版本 |
| 导出后精度下降 | 算子实现差异 | 对比 PyTorch 与 ONNX 输出，定位差异层 |
| 动态尺寸推理慢 | 动态 shape 优化难 | 固定输入尺寸导出，或用 TensorRT profile |
| Transformer 导出失败 | 复杂注意力实现 | 使用 mmcv/onnxruntime 扩展算子 |

### 3. 验证 ONNX 模型一致性

```python
# 伪代码：验证 ONNX 与 PyTorch 输出一致性
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("deeplabv3plus.onnx")
input_np = np.random.randn(1, 3, 512, 512).astype(np.float32)

# ONNX 推理
onnx_output = session.run(["output"], {"input": input_np})[0]

# PyTorch 推理
with torch.no_grad():
    torch_output = model(torch.from_numpy(input_np)).numpy()

# 对比（允许小误差）
max_diff = np.abs(onnx_output - torch_output).max()
print(f"最大输出差异: {max_diff:.8f}")  # 应 <1e-4
```

## 二、推理优化：TensorRT

### 1. TensorRT 概述

- NVIDIA 的高性能深度学习推理优化库
- 仅支持 NVIDIA GPU
- 典型加速：比 PyTorch 推理快 2~5 倍
- 核心优化：算子融合（Layer Fusion）、精度校准（FP16/INT8）、内核自动调优（Kernel Auto-Tuning）

### 2. ONNX → TensorRT Engine

```bash
# 方式一：trtexec 命令行工具（最简单）
trtexec --onnx=deeplabv3plus.onnx \
        --saveEngine=deeplabv3plus_trt.engine \
        --fp16                          # 启用 FP16 精度

# 方式二：Python API（可控制更多参数）
# 伪代码
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)
parser = trt.OnnxParser(network, logger)
parser.parse_from_file("deeplabv3plus.onnx")
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)    # FP16 量化
engine_bytes = builder.build_serialized_network(network, config)
```

### 3. FP16 vs INT8 量化对比

| 精度 | 推理速度 | 显存占用 | 精度损失 | 适用场景 |
| --- | --- | --- | --- | --- |
| FP32 | 1×（基准） | 1× | 无 | 精度要求最高 |
| FP16 | ~2× | ~0.5× | 可忽略（<0.1% mIoU） | **推荐默认** |
| INT8 | ~3~4× | ~0.25× | 0.5~2% mIoU | 极致速度要求 |

### 4. INT8 校准（Calibration）

INT8 量化需要一批校准数据确定量化参数：

```python
# 伪代码：INT8 校准流程
class CalibrationDataset:
    def __init__(self, image_dir, batch_size=8):
        self.images = load_images(image_dir)
        self.batch_size = batch_size

    def get_batch(self):
        # 返回归一化后的影像 batch
        return preprocess_batch(self.images[:self.batch_size])

# 使用校准器构建 INT8 engine
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Int8Calibrator(CalibrationDataset("data/val/"))
```

校准数据量建议：500~2000 张代表性影像。校准数据应覆盖实际部署场景的影像分布。

## 三、大图推理工程方案

### 1. 滑窗推理（Sliding Window Inference）

遥感影像单张可达 6000×6000（Potsdam）甚至更大，GPU 显存无法直接处理。标准做法是滑窗推理：

```python
# 伪代码：滑窗推理核心逻辑
def sliding_window_inference(model, large_image, window_size=512, stride=384):
    """
    large_image: [H, W, C]，尺寸可能 6000×6000
    window_size: 裁剪窗口大小
    stride: 滑动步长（< window_size 时有重叠）
    """
    H, W = large_image.shape[:2]
    # 初始化累加器和计数器（用于重叠区域加权平均）
    result_accumulator = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            # 确保不越界
            y_end = min(y + window_size, H)
            x_end = min(x + window_size, W)
            y_start = max(y_end - window_size, 0)
            x_start = max(x_end - window_size, 0)

            # 裁剪 patch
            patch = large_image[y_start:y_end, x_start:x_end]
            # 模型推理（自动 padding 到 window_size）
            pred = model.predict(patch)  # [C, h, w]

            # 累加预测结果（重叠区域取平均）
            result_accumulator[:, y_start:y_end, x_start:x_end] += pred
            count_map[y_start:y_end, x_start:x_end] += 1

    # 归一化（重叠区域除以计数）
    final_pred = result_accumulator / count_map[None, ...]
    return final_pred.argmax(axis=0)  # [H, W] 最终分割 mask
```

### 2. stride / overlap 选择

| overlap 比例 | stride | 推理次数 | 效果 |
| --- | --- | --- | --- |
| 0%（无重叠） | = window_size | 最少 | 边界伪影严重，不推荐 |
| 25% | 0.75 × window_size | ~1.78× | 平衡速度与质量 |
| 50% | 0.5 × window_size | ~4× | 边界伪影少，常用 |
| 75% | 0.25 × window_size | ~16× | 效果最好，但极慢 |

推荐：**stride = window_size × 0.5~0.75**（即 25%~50% overlap），是工程中最常用的配置。

### 3. 边界伪影消除

- **问题**：裁剪边界处目标被切断，预测结果有接缝
- **对策 1：加权融合**。重叠区域用高斯权重加权（中心权重高，边缘权重低），而非简单平均
- **对策 2：多方向裁剪**。同一区域用不同偏移裁剪多次，取平均
- **对策 3：镜像填充**。裁剪到固定尺寸时，边界做镜像 padding 而非零填充

### 4. 显存估算

| 模型 | 输入 512×512 FP16 | 输入 1024×1024 FP16 |
| --- | --- | --- |
| U-Net (ResNet-50) | ~1.5 GB | ~4 GB |
| DeepLabV3+ (ResNet-101) | ~2.5 GB | ~7 GB |
| Swin-L + UperHead | ~4 GB | ~12 GB |
| SegFormer-B5 | ~3 GB | ~9 GB |

建议：推理时显存占用不超过 GPU 总显存的 70%，留余量给系统和数据 I/O。

## 四、推理速度基准参考

以下为常见模型在 NVIDIA GPU 上的推理速度参考（512×512，FP16，batch=1，不含后处理）：

| 模型 | RTX 3090 (ms) | RTX 4090 (ms) | A100 (ms) | Jetson Orin (ms) |
| --- | --- | --- | --- | --- |
| U-Net (ResNet-50) | ~12 | ~7 | ~6 | ~45 |
| DeepLabV3+ (ResNet-101) | ~22 | ~14 | ~11 | ~80 |
| SegFormer-B0 | ~8 | ~5 | ~4 | ~30 |
| SegFormer-B5 | ~35 | ~22 | ~18 | ~120 |
| Swin-T + UperHead | ~18 | ~11 | ~9 | ~65 |
| Swin-L + UperHead | ~65 | ~40 | ~32 | ~220 |

说明：以上为代表性范围，实际速度取决于 TensorRT 版本、CUDA 版本、输入尺寸和系统负载。RTX 4090 比 3090 通常快 40~60%。

## 五、模型压缩与蒸馏

### 1. 知识蒸馏（Knowledge Distillation）

- 思路：用大模型（Teacher）指导小模型（Student）训练，Student 模拟 Teacher 的输出分布
- 典型场景：Swin-L → SegFormer-B0，在保持 90%+ 精度的同时将参数减少 10×

```python
# 伪代码：分割模型蒸馏训练
def distillation_train_step(student, teacher, images, masks):
    student_logits = student(images)
    with torch.no_grad():
        teacher_logits = teacher(images)

    # 硬标签损失（GT）
    hard_loss = cross_entropy(student_logits, masks)
    # 软标签损失（Teacher 输出）
    soft_loss = kl_divergence(
        log_softmax(student_logits / T),
        softmax(teacher_logits / T)
    ) * (T ** 2)
    # 总损失
    total_loss = alpha * hard_loss + beta * soft_loss
    # 常用：alpha=0.5, beta=0.5, T=4
```

### 2. 模型剪枝（Pruning）

- 结构化剪枝：删除整个通道/层，速度提升实际可见
- 非结构化剪枝：逐权重置零，需专用稀疏推理引擎
- 遥感分割常用：通道剪枝 30~50%，精度损失 <1% mIoU

### 3. 模型量化总结

| 量化方式 | 工具 | 精度损失 | 速度提升 |
| --- | --- | --- | ---|
| 训练后量化 PTQ (FP16) | TensorRT / ONNX Runtime | <0.1% | ~2× |
| 训练后量化 PTQ (INT8) | TensorRT + 校准数据 | 0.5~2% | ~3~4× |
| 量化感知训练 QAT (INT8) | PyTorch QAT + TensorRT | <0.5% | ~3~4× |

推荐流程：FP16 → 测试精度 → 不满意则 INT8 PTQ → 仍不满意则 QAT 微调。

## 六、端到端部署架构

### 1. 单机部署（适用于内网/实验室）

```
[用户客户端] → FastAPI / Flask 服务 → ONNX Runtime / TensorRT 推理 → 返回结果
```

- 优点：简单、低延迟
- 适用：并发 <10 的场景

### 2. 高并发部署（适用于生产环境）

```
[用户客户端] → Nginx 负载均衡 → 多个推理 Worker（Docker 容器）
                                    → GPU 推理
                                    → 结果合并
                                → Redis 缓存（对相同影像缓存结果）
```

- 推理 Worker 数量 = GPU 数量 × 每卡并发数
- 大图推理任务可拆分为多个 tile，分布式推理后合并

### 3. 边缘部署（无人机/星载）

- 硬件：NVIDIA Jetson Orin / Nano，或国产 AI 芯片（华为昇腾、瑞芯微 RK3588）
- 模型：SegFormer-B0 / MobileNetV3 + DeepLabV3 头，参数 <10M
- 精度：INT8 量化
- 延迟目标：<100ms / tile（512×512）
- 关键约束：功耗 <15W、重量 <200g（无人机载荷）

## 七、部署常见问题排查

| 问题 | 可能原因 | 解决方案 |
| --- | --- | --- |
| ONNX 推理结果与 PyTorch 不一致 | 算子实现差异 / BN 模式 | 检查 eval() 模式、对比中间层输出 |
| TensorRT 推理比 ONNX Runtime 慢 | 未做内核调优 / 动态 shape | 固定输入尺寸、开启 builder optimization |
| 大图推理有接缝 | overlap 不够 / 边界处理差 | 增加 overlap 到 50%、用高斯加权融合 |
| INT8 后精度大幅下降 | 校准数据不充分 | 增加校准数据量到 1000+、改用 QAT |
| OOM（显存不足） | batch/输入太大 | 降低 batch、用 FP16、减小输入尺寸 |
| 推理速度时快时慢 | CPU I/O 瓶颈 | 预加载数据到内存、用 pinned memory |
