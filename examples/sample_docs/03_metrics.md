# 遥感语义分割评价指标

本文件系统梳理语义分割常用评价指标的定义、公式、适用场景与遥感领域的使用惯例，用于知识库问答。

## 一、混淆矩阵（基础）

对于 C 类语义分割，构建 C × C 混淆矩阵 M，其中：

- M[i][j]：真实类别为 i、预测类别为 j 的像素数
- 对角线 M[i][i]：预测正确的像素数
- 行合计：真实类别 i 的总像素数
- 列合计：预测为类别 j 的总像素数

所有下面的指标都基于混淆矩阵推导。

## 二、像素级指标

### 1. PA (Pixel Accuracy, 像素精度)

- 定义：所有预测正确的像素 / 总像素数
- 公式：PA = Σ_i M[i][i] / Σ_i Σ_j M[i][j]
- 特点：最直观，但会被占主导的大类（背景、林地）掩盖，对小类不敏感。
- 遥感中的局限：当"背景"或"裸地"占 70% 以上时，PA 即使很高，小目标（车辆、建筑）可能完全分错。

### 2. MPA (Mean Pixel Accuracy, 平均像素精度)

- 定义：每个类别像素精度的算术平均
- 公式：MPA = (1/C) Σ_i [ M[i][i] / Σ_j M[i][j] ]
- 特点：对每个类别同等加权，比 PA 更公平，但仍未考虑预测的"误检"（列方向）。

## 三、IoU 系列指标（最常用）

### 3. IoU (Intersection over Union, 交并比 / Jaccard Index)

- 定义：对类别 i，IoU_i = 真实 i 与预测 i 的交集 / 并集
- 公式：IoU_i = M[i][i] / ( Σ_j M[i][j] + Σ_j M[j][i] - M[i][i] )
- 范围：[0, 1]，越大越好
- 特点：同时惩罚漏检与误检，是分割任务最核心的指标。

### 4. mIoU (Mean IoU, 平均交并比)

- 定义：所有类别 IoU 的算术平均
- 公式：mIoU = (1/C) Σ_i IoU_i
- 遥感惯例：mIoU 是遥感语义分割论文与竞赛的**第一指标**，LoveDA、DeepGlobe、ISPRS benchmark 均以 mIoU 排名。
- 注意：mIoU 对小类不友好（小类 IoU 波动大），需要配合 per-class IoU 或 FWIoU 一起看。

### 5. FWIoU (Frequency Weighted IoU, 频率加权 IoU)

- 定义：按真实类别像素占比加权的 IoU
- 公式：FWIoU = Σ_i [ (Σ_j M[i][j] / N) × IoU_i ]，N 为总像素数
- 特点：对大类赋更高权重，介于 PA 与 mIoU 之间。

## 四、精确率 / 召回率 / F1

### 6. Precision (精确率)

- 对类别 i：Precision_i = M[i][i] / Σ_j M[j][i]
- 含义：预测为 i 的像素中，真正是 i 的比例（管"误检"）

### 7. Recall (召回率)

- 对类别 i：Recall_i = M[i][i] / Σ_j M[i][j]
- 含义：真实是 i 的像素中，被正确预测的比例（管"漏检"）

### 8. F1 Score

- 对类别 i：F1_i = 2 × Precision_i × Recall_i / (Precision_i + Recall_i)
- macro-F1：各类 F1 的算术平均
- 遥感惯例：ISPRS Potsdam / Vaihingen benchmark 使用 F1 作为主要指标之一。

## 五、整体精度与 Kappa

### 9. OA (Overall Accuracy)

- 等同于 PA，常在遥感制图（land cover mapping）领域使用。

### 10. Kappa 系数

- 定义：考虑随机一致性的修正 OA
- 公式：Kappa = (p_o - p_e) / (1 - p_e)
  - p_o = OA
  - p_e = Σ_i [ (真实 i 占比) × (预测 i 占比) ]（随机一致概率）
- 范围：[-1, 1]，>0.8 一致性良好
- 遥感惯例：在传统遥感分类（最大似然、随机森林）中常用，深度学习论文中使用减少，但在大尺度地表覆盖制图中仍出现。

## 六、边界与实例级指标

### 11. BF Score (Boundary F1)

- 评估分割边界质量，对建筑/道路这类对边界精度敏感的任务重要。

### 12. mAP (mean Average Precision)

- 用于实例分割（如 iSAID），按置信度阈值做 PR 曲线积分。

## 七、遥感场景下的指标使用建议

| 任务类型 | 主指标 | 辅助指标 | 注意事项 |
| --- | --- | --- | --- |
| 通用语义分割 | mIoU | per-class IoU、PA | 必须看小类 IoU，别只看均值 |
| 城市地物（建筑/道路/车辆） | mIoU + F1 | BF Score | 关注边界与小目标 |
| 大尺度地表覆盖 | mIoU + OA + Kappa | FWIoU | 类别极度不均衡，看 FWIoU 更公平 |
| 实例分割（iSAID） | mAP@0.5 / mAP@0.5:0.95 | per-class AP | 小目标 AP 低是常态 |
| 跨域泛化（LoveDA） | 跨域 mIoU gap | per-class drop | 关注跨域后哪类掉点最严重 |

## 八、常见误区

1. 只报 mIoU 不报 per-class IoU：会掩盖小类完全失效的情况。
2. 评测时忽略"ignore_index"：背景或边界像素的处理方式不同会显著影响 mIoU，需明确标注协议。
3. 在线增广评测时不固定随机种子：分割对初始化敏感，建议跑 3~5 次取均值 ± 方差。
4. 测试时不用 TTA（Test Time Augmentation）：水平/垂直翻转 + 多尺度推理通常能提升 1~3 mIoU，论文常用。
