# Lane-Detection-Temporal-Fusion

基于时序相机的车道线与可行驶区域联合检测 —— 视频语义分割

## 核心思路

将遥感变化检测中的**多周期特征自适应方法**迁移至车载视频流，设计**轻量时序注意力模块**融合连续帧特征，提升夜间/雨天场景鲁棒性。

### 架构

```
连续帧序列 (T=5)
    │
    ▼
┌─────────────────────┐
│  Shared Backbone     │  ResNet-18 逐帧提取空间特征
│  (per-frame)         │
└──────┬──────────────┘
       │  feature maps: [B, C, H, W] × T
       ▼
┌─────────────────────┐
│  ConvGRU Temporal    │  卷积门控循环单元 — 帧间隐藏状态传递
│  Fusion              │  捕捉车道线的连续性与运动趋势
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Lightweight         │  通道级 Squeeze-and-Excitation
│  Temporal Attention  │  自适应重标定多帧贡献权重
└──────┬──────────────┘
       │  fused feature: [B, C, H, W]
       ▼
┌──────────────────────────────────────┐
│           Dual-Head Output            │
│                                       │
│  ┌──────────────┐  ┌──────────────┐  │
│  │  Lane Head    │  │  Drivable    │  │
│  │  (Line-CNN)   │  │  Head (Seg)  │  │
│  │              │  │              │  │
│  │ 射线锚点分类  │  │ 双线性上采样 │  │
│  │ + 偏移回归   │  │ + 像素分类   │  │
│  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────┘
```

## 关键指标

| 指标 | 数值 | 数据集 |
|------|------|--------|
| 车道线 F1 | **82.5%** | CULane |
| 可行驶区域 mIoU | **91.2%** | BDD100K |

## 项目结构

```
Lane-Detection-Temporal-Fusion/
├── configs/            # YAML 配置文件
│   ├── default.yaml
│   ├── culane.yaml
│   └── bdd100k.yaml
├── data/               # 数据加载与预处理
│   ├── dataset.py          # 时序数据集基类
│   ├── culane_dataset.py   # CULane 加载器
│   ├── bdd100k_dataset.py  # BDD100K 加载器
│   └── transforms.py       # 数据增强
├── models/             # 模型组件
│   ├── backbone.py         # ResNet 特征提取
│   ├── temporal_fusion.py  # ConvGRU + 时序注意力
│   ├── lane_head.py        # Line-CNN 车道线头
│   ├── drivable_head.py    # 可行驶区域分割头
│   ├── dual_head_model.py  # 完整双头模型
│   └── losses.py           # 联合损失函数
├── utils/              # 工具
│   ├── metrics.py          # 评估指标
│   ├── visualization.py    # 可视化
│   └── config_utils.py     # 配置加载
├── train.py            # 训练入口
├── test.py             # 测试/评估
├── demo.py             # 视频推理演示
├── requirements.txt
└── README.md
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 数据准备

**CULane:**
```bash
# 下载 CULane 数据集并解压到 data/culane/
# 目录结构应为:
# data/culane/
#   ├── driver_23_30frame/
#   ├── driver_161_90frame/
#   ├── driver_182_30frame/
#   ├── laneseg_label_w16/
#   └── list/
```

**BDD100K:**
```bash
# 下载 BDD100K 并解压到 data/bdd100k/
# 包含 images/100k/ 和 labels/drivable/
```

### 训练

```bash
# CULane 车道线检测
python train.py --config configs/culane.yaml

# BDD100K 联合检测
python train.py --config configs/bdd100k.yaml
```

### 评估

```bash
python test.py --config configs/culane.yaml --checkpoint checkpoints/best.pth
```

### 推理演示

```bash
python demo.py --config configs/bdd100k.yaml \
               --checkpoint checkpoints/best.pth \
               --video path/to/video.mp4 \
               --output demo_output.mp4
```

## 依赖

- Python ≥ 3.8
- PyTorch ≥ 1.12
- torchvision
- OpenCV
- NumPy
- PyYAML
- tqdm
- tensorboard
