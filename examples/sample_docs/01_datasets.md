# 遥感语义分割主流数据集

本文件汇总遥感语义分割领域常用公开数据集，涵盖数据规模、空间分辨率、类别定义、影像来源等关键信息，用于知识库问答。

## 1. LoveDA

- 全称：Land Cover Domain Adaptation dataset
- 发布：南京大学 RS-IDE 实验室，2021 年
- 影像来源：Google Earth 高分影像
- 空间分辨率：0.3 m
- 影像数量：训练集 2522 张、验证集 1669 张、测试集 5987 张（公开测试集无标注）
- 影像尺寸：1024 × 1024 像素
- 类别数：7 类
- 类别定义：建筑（building）、道路（road）、水体（water）、裸地（bareland）、林地（forest）、农业（agriculture）、背景（background）
- 特点：包含 Urban（城市）与 Rural（乡村）两个域，专门用于研究跨域泛化（Domain Generalization / Adaptation）问题；城市与乡村场景的地物分布差异显著，直接跨域推理通常会出现明显性能下降。
- 评价指标：mIoU（Mean Intersection over Union）、OA（Overall Accuracy）
- 官方地址：https://github.com/Junjue-Wang/LoveDA

## 2. iSAID

- 全称：Instance Segmentation in Aerial Images Dataset
- 发布：DLR / TetrasAI，2019 年
- 影像来源：DOTA（大型航空影像数据集）裁剪
- 空间分辨率：0.15 ~ 0.5 m 不等
- 影像数量：从 2806 张大型航空影像中裁剪出 1411 张子图，共 655,451 个实例标注
- 影像尺寸：约 800 × 800 ~ 1300 × 1300
- 类别数：15 类
- 类别定义：船只（ship）、存储罐（storage tank）、网球场（tennis court）、篮球场（basketball court）、操场（ground track field）、港口（harbor）、桥梁（bridge）、大型车辆（large vehicle）、小型车辆（small vehicle）、直升机（helicopter）、飞机（plane）、环岛（roundabout）、足球场（soccer ball field）、游泳池（swimming pool）、风力发电机（windmill）
- 特点：是目前最大的航空影像实例分割数据集，同时也支持语义分割评测；标注密度高、目标尺度变化极大（同一类目标可能跨越数十像素到上千像素）。
- 评价指标：mAP（实例分割）、mIoU（语义分割）
- 官方地址：https://github.com/CAPTAIN-WHU/iSAID

## 3. DeepGlobe

- 发布：CVPR 2018 Workshop Challenge
- 影像来源：WorldView-3 卫星
- 空间分辨率：0.5 m
- DeepGlobe Land Cover 子集：
  - 影像数量：训练集 803 张、验证集 171 张、测试集 172 张
  - 影像尺寸：2448 × 2448 像素（原始），常裁剪为 512 × 512 训练
  - 类别数：7 类
  - 类别定义：林地（urban）、农业（agriculture land）、牧场（rangeland）、植被（forest land）、水体（water）、裸地（barren land）、未知（unknown）
- 特点：影像尺寸大、分辨率高，类别分布极度不均衡（urban 与 barren 占比差异大），是评测分割模型在大尺度场景下稳定性的标杆。
- 评价指标：mIoU
- 官方地址：http://deepglobe.org/

## 4. Potsdam / Vaihingen（ISPRS 2D 语义分割）

- 发布：ISPRS（国际摄影测量与遥感协会）Benchmark
- 影像来源：机载 DSM + 真正射影像（True Orthophoto，TOP）
- 空间分辨率：Potsdam 0.05 m，Vaihingen 0.09 m
- 影像尺寸：Potsdam 6000 × 6000（共 38 块），Vaihingen 原始平均约 2500 × 2000（共 33 块）
- 模态：IRRG（近红外+红+绿）或 RGB + DSM 高程
- 类别数：6 类（含背景）
- 类别定义：不透水面（Impervious Surface）、建筑（Building）、低矮植被（Low Vegetation）、树（Tree）、车（Car）、背景（Clutter / Background）
- 特点：超高分辨率城市密集区数据，是城市地物提取的权威 benchmark；DSM 高程通道可提升建筑与树的区分。
- 评价指标：F1、OA、mIoU
- 官方地址：https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx

## 5. UCMerced LandUse（21 类场景分类，常用于预训练）

- 发布：UC Merced，2010 年
- 影像来源：USGS 城市区域正射影像
- 空间分辨率：0.3 m
- 影像数量：21 类 × 100 张 = 2100 张
- 影像尺寸：256 × 256
- 用途：场景级分类（非像素级分割），但常被用作遥感 backbone 预训练或特征学习数据。
- 类别：农田、飞机、棒球场、海滩、建筑、丛林、密集住宅、森林、高速公路、高尔夫球场、港口、十字路口、中等住宅、移动房屋公园、立交桥、停车场、河流、跑道、稀疏住宅、储油罐、网球场。

## 6. 选择数据集的工程建议

- 城市地物提取（建筑/道路/车辆）：优先 LoveDA、Potsdam、iSAID
- 大尺度地表覆盖（林地/水体/农田）：优先 DeepGlobe Land Cover
- 跨域泛化研究：优先 LoveDA（Urban→Rural）
- 实例分割与语义分割联合任务：优先 iSAID
- 模型预训练 backbone：可用 UCMerced 或 ImageNet 预训练权重，再用目标数据集微调
