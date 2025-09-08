# 机器人视觉伺服系统

基于Auboi5机械臂的智能视觉伺服控制系统，整合了YOLO目标检测和RealSense深度相机，实现精确的机器人视觉控制。

## 🚀 项目特色

- **双模式视觉伺服**: 支持PBVS（基于位置）和IBVS（基于图像）两种控制方法
- **智能目标检测**: 集成YOLO深度学习模型，实现实时目标识别
- **深度感知**: 使用RealSense D435i相机获取精确的3D空间信息
- **安全可靠**: 内置多重安全机制，包括紧急停止和碰撞检测
- **易于使用**: 提供完整的环境配置和一键启动脚本

## 📋 系统要求

### 硬件要求
- **机械臂**: Auboi5六轴机械臂
- **相机**: Intel RealSense D435i深度相机
- **计算机**: 
  - CPU: Intel i5或更高
  - 内存: 8GB RAM（推荐16GB）
  - GPU: NVIDIA GPU（用于YOLO推理，可选）
  - 存储: 至少5GB可用空间

### 软件环境
- **操作系统**: Ubuntu 18.04/20.04/22.04
- **Python**: 3.8.x（通过conda管理）
- **网络**: 与Auboi5机械臂的网络连接

## 📖 目录

1. [🚀 项目特色](#-项目特色)
2. [📋 系统要求](#-系统要求)
3. [⚡ 快速开始](#-快速开始)
4. [🔧 详细安装指南](#-详细安装指南)
5. [🎯 使用说明](#-使用说明)
6. [📁 代码文件说明](#-代码文件说明)
7. [🧠 视觉伺服方法](#-视觉伺服方法)
8. [⚙️ 参数配置](#️-参数配置)
9. [🛠️ 故障排除](#️-故障排除)
10. [📚 API参考](#-api参考)
11. [🔬 扩展开发](#-扩展开发)

## ⚡ 快速开始

### 第一次使用？请按以下步骤操作：

```bash
# 1. 克隆项目
git clone https://github.com/als-ichkann/vision.git
cd vision

# 2. 设置环境（一键配置）
./setup_environment.sh

# 3. 连接硬件设备
# - 确保Auboi5机械臂已连接并启动
# - 连接RealSense D435i相机到USB 3.0接口

# 4. 运行系统测试
python test_system.py --all

# 5. 启动视觉伺服系统
python robot_visual_servoing_integrated.py
```

### 🎮 操作控制
- **'q'**: 退出程序
- **'e'**: 紧急停止机械臂
- **'r'**: (IBVS模式) 重新设置期望特征

## 🔧 详细安装指南

### 步骤1: 环境准备

```bash
# 检查系统版本
lsb_release -a

# 安装基础依赖
sudo apt update
sudo apt install git curl wget build-essential
```

### 步骤2: 安装Miniconda

```bash
# 下载并安装Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 重启终端或执行
source ~/.bashrc
```

### 步骤3: 创建Python环境

```bash
# 创建专用环境
conda create -n vision python=3.8 -y
conda activate vision

# 安装依赖包
pip install -r requirements.txt
```

### 步骤4: 硬件连接

1. **机械臂连接**：
   ```bash
   # 确保机械臂在同一网络中
   ping <机械臂IP地址>
   ```

2. **相机连接**：
   ```bash
   # 测试RealSense相机
   python -c "import pyrealsense2 as rs; print('相机连接成功')"
   ```

### 步骤5: 系统验证

```bash
# 运行完整测试
python test_system.py --all

# 测试相机
python test_system.py --camera

# 测试YOLO模型
python test_system.py --yolo
```

## 🎯 使用说明

### 基本使用流程

#### 1. 启动系统
```bash
# 激活环境
conda activate vision

# 启动主程序
python robot_visual_servoing_integrated.py
```

#### 2. 选择控制模式

程序启动后，您将看到以下选项：

```
=== 机器人视觉伺服系统 ===
控制方法:
1. PBVS - 基于位置的视觉伺服
2. IBVS - 基于图像的视觉伺服

请选择控制方法 (1-2):
```

- **选择1 (PBVS)**: 适用于需要精确3D位置控制的任务
- **选择2 (IBVS)**: 适用于图像空间的跟踪和定位任务

#### 3. 配置机器人连接

```
机器人IP地址 (默认: localhost): 192.168.1.100
机器人端口 (默认: 8899): 8899
```

#### 4. 系统运行

系统启动后将显示：
- 实时相机画面
- 目标检测结果
- 机械臂状态信息
- 控制误差数值

### 高级使用

#### 相机标定
```bash
# 首次使用建议进行相机标定
python camera_calibration.py

# 按照提示使用棋盘格进行标定
# 标定结果将保存到 camera_calibration.json
```

#### YOLO模型训练
```bash
# 使用自定义数据集训练YOLO模型
python yolo_train.py

# 训练结果保存在 yolo_train/weights/ 目录
```

#### 目标检测测试
```bash
# 测试YOLO检测效果
python yolo_detect.py
```

## 系统概述

本系统实现了两种主要的视觉伺服方法：

- **PBVS (Position-Based Visual Servoing)**: 基于位置的视觉伺服 - 在3D空间中进行精确位置控制
- **IBVS (Image-Based Visual Servoing)**: 基于图像的视觉伺服 - 直接在图像空间中进行伺服控制

系统集成了Auboi5机器人控制、YOLO目标检测和RealSense深度相机，可以实现对目标物体的精确视觉伺服控制。

## 环境设置

### 已完成的设置

✅ **Miniconda已安装**: `/home/wang-yb/miniconda3/`  
✅ **conda环境已创建**: `vision` (Python 3.8)  
✅ **依赖包已安装**: 所有requirements.txt中的包  
✅ **环境测试通过**: 所有关键依赖包可正常导入  

### 快速启动环境

#### 方法1：使用设置脚本（推荐）

```bash
# 在项目目录中运行
./setup_environment.sh
```

#### 方法2：手动激活环境

```bash
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision

# 验证环境
python --version  # 应该显示Python 3.8.x
which python      # 应该显示conda环境中的Python路径
```

### 环境信息

- **环境名称**: `vision`
- **Python版本**: 3.8.20
- **conda位置**: `/home/wang-yb/miniconda3/`
- **环境位置**: `/home/wang-yb/miniconda3/envs/vision/`

### 已安装的核心依赖

- `opencv-python>=4.5.0` - 计算机视觉库
- `numpy>=1.21.0` - 数值计算库
- `pyrealsense2>=2.50.0` - RealSense相机SDK
- `ultralytics>=8.0.0` - YOLO目标检测
- `torch>=1.9.0` - PyTorch深度学习框架
- `torchvision>=0.10.0` - PyTorch视觉工具
- `scipy>=1.7.0` - 科学计算库
- `matplotlib>=3.3.0` - 绘图库
- `Pillow>=8.3.0` - 图像处理库

### 避免ROS2冲突

- 使用独立的conda环境 `vision`
- 与系统Python和ROS2 Python完全隔离
- 每个环境有独立的包管理

## 📁 代码文件说明

### 核心驱动程序

#### 1. `robot_visual_servoing_integrated.py`
**主要功能**: 整合的视觉伺服驱动系统
- **作用**: 系统的主入口，集成了PBVS和IBVS两种视觉伺服方法
- **核心类**: `RobotVisualServoing` - 完整的视觉伺服控制系统
- **主要功能**:
  - 机器人连接和控制
  - RealSense相机管理
  - YOLO目标检测
  - PBVS和IBVS控制算法
  - 实时可视化和用户交互
- **运行方式**: `python robot_visual_servoing_integrated.py`
- **支持模式**: 
  - 模式1: PBVS（基于位置的视觉伺服）
  - 模式2: IBVS（基于图像的视觉伺服）

#### 2. `robotcontrol.py`
**主要功能**: Auboi5机器人控制接口
- **作用**: 提供与Auboi5机械臂的底层通信接口
- **核心类**: `Auboi5Robot` - 机器人控制器
- **主要功能**:
  - 机器人连接和初始化
  - 运动控制（关节空间和笛卡尔空间）
  - 安全监控和错误处理
  - 日志记录和状态监控
- **依赖库**: `libpyauboi5` - Auboi5机器人SDK
- **使用场景**: 被其他模块调用，不直接运行

### 仿真系统模块

#### 3. `robot_visual_servoing_simulation.py`
**主要功能**: 集成视觉伺服仿真系统
- **作用**: 支持仿真和真实机械臂模式切换的视觉伺服系统
- **核心类**: `RobotVisualServoing` - 统一的视觉伺服控制系统
- **主要功能**:
  - 仿真/真实模式无缝切换
  - PBVS和IBVS控制算法
  - 与现有代码完全兼容
- **运行方式**: `python robot_visual_servoing_simulation.py`

#### 3. `camera_calibration.py`
**主要功能**: 相机标定工具
- **作用**: 标定RealSense相机的内参矩阵和畸变系数
- **核心类**: `CameraCalibration` - 相机标定工具
- **主要功能**:
  - 棋盘格图像采集
  - 相机内参标定
  - 畸变系数计算
  - 标定结果保存和加载
- **输出文件**: `camera_calibration.json` - 标定参数文件
- **运行方式**: `python camera_calibration.py`
- **使用流程**:
  1. 准备9x6内角点的标准棋盘格
  2. 在不同角度拍摄20张图像
  3. 自动计算并保存标定参数

### 目标检测模块

#### 4. `yolo_detect.py`
**主要功能**: YOLO目标检测演示
- **作用**: 使用训练好的YOLO模型进行实时目标检测
- **主要功能**:
  - 加载YOLO模型
  - RealSense相机图像获取
  - 实时目标检测和显示
  - 深度信息获取
- **模型文件**: 使用 `yolo_train/weights/best.pt`
- **运行方式**: `python yolo_detect.py`
- **使用场景**: 测试YOLO模型性能，验证检测效果

#### 5. `yolo_train.py`
**主要功能**: YOLO模型训练脚本
- **作用**: 训练自定义的YOLO目标检测模型
- **主要功能**:
  - 加载预训练YOLOv5模型
  - 使用自定义数据集训练
  - 模型权重保存
- **配置文件**: 需要配置数据集路径（当前指向番茄检测数据集）
- **输出目录**: `yolo_train/` - 训练结果和权重文件
- **运行方式**: `python yolo_train.py`
- **训练参数**: 100轮训练，640像素图像尺寸

### 测试和验证模块

#### 6. `test_system.py`
**主要功能**: 系统功能测试
- **作用**: 快速测试系统各个组件的工作状态
- **主要功能**:
  - 测试相机连接
  - 测试YOLO模型加载
  - 测试机器人连接
  - 系统完整性验证
- **运行方式**: `python test_system.py --all`
- **测试项目**:
  - RealSense相机连接和图像获取
  - YOLO模型加载和推理
  - 依赖库完整性检查

### 配置和权重文件

#### 7. `requirements.txt`
**主要功能**: Python依赖包列表
- **作用**: 定义项目所需的所有Python包及版本
- **包含的主要包**:
  - opencv-python, numpy, scipy - 计算机视觉和数值计算
  - pyrealsense2 - RealSense相机支持
  - ultralytics - YOLO目标检测
  - torch, torchvision - 深度学习框架
  - matplotlib, Pillow - 图像处理和可视化
- **使用方式**: `pip install -r requirements.txt`

#### 8. `setup_environment.sh`
**主要功能**: 环境设置脚本
- **作用**: 自动激活conda环境并进行环境检查
- **主要功能**:
  - 激活vision conda环境
  - 检查Python版本
  - 验证关键依赖包
- **运行方式**: `./setup_environment.sh`

#### 9. `yolov5nu.pt`
**主要功能**: 预训练YOLO模型
- **作用**: YOLOv5的预训练权重文件
- **使用场景**: 作为训练的起始点或直接用于检测

#### 10. `yolo_train/` 目录
**主要功能**: YOLO训练输出目录
- **作用**: 存储YOLO模型训练的所有结果
- **主要文件**:
  - `weights/best.pt` - 训练得到的最佳模型权重
  - `weights/last.pt` - 最后一轮的模型权重
  - `results.csv` - 训练过程数据
  - `*.png` - 训练过程可视化图表
  - `args.yaml` - 训练参数配置

## 功能特性

### ✅ 核心功能
- 基于Auboi5机器人的完整控制接口
- RealSense D435i相机支持
- YOLO v8目标检测
- 实时视觉伺服控制
- RGB-D图像对齐和深度感知

### ✅ 安全功能
- 速度限制和紧急停止
- 目标丢失检测和处理
- 碰撞检测和工作空间限制
- 实时状态监控

### ✅ 用户友好
- 简化的一体化设计
- 实时可视化显示
- 交互式参数调整
- 详细的状态反馈

## 快速开始

### 1. 环境准备
```bash
# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vision

# 或使用设置脚本
./setup_environment.sh
```

### 2. 系统测试
```bash
# 测试相机连接
python -c "import pyrealsense2 as rs; print('RealSense相机可用')"

# 测试YOLO模型
python -c "from ultralytics import YOLO; print('YOLO模型可用')"

# 运行完整系统测试
python test_system.py --all
```

### 3. 相机标定（首次使用）
```bash
python camera_calibration.py
```

### 4. 运行视觉伺服

**真实机械臂模式**:
```bash
# 运行主程序
python robot_visual_servoing_integrated.py
```


### 5. 选择模式
- **模式1**: PBVS - 适用于精确位置控制
- **模式2**: IBVS - 适用于图像空间控制

### 6. 操作控制
- **'q'**: 退出程序
- **'e'**: 紧急停止
- **'r'**: (IBVS模式) 重新设置期望特征

## 🧠 视觉伺服方法

### PBVS (基于位置的视觉伺服)

#### 工作原理
1. 使用相机和深度信息获取目标的3D位置和姿态
2. 在笛卡尔空间中定义位姿误差
3. 通过控制律将误差转换为机器人运动命令
4. 控制机器人末端执行器到达期望的3D位置

#### 控制律
```
位置误差: e_p = P_desired - P_current
旋转误差: e_r = log(R_desired × R_current^T)
控制速度: v = λ_pos × e_p, ω = λ_rot × e_r
```

#### 优点
- 直观易懂，误差定义在3D空间
- 轨迹规划简单直接
- 适合精确位置控制任务

#### 缺点
- 需要准确的深度信息
- 对相机标定精度要求较高
- 需要目标的3D几何模型

#### 适用场景
- 机器人抓取任务
- 精确装配作业
- 工件定位操作

### IBVS (基于图像的视觉伺服)

#### 工作原理
1. 直接使用图像特征作为反馈信号
2. 在图像空间中定义特征误差
3. 通过图像雅可比矩阵将图像误差转换为机器人运动
4. 保持目标在图像中的期望位置和尺寸

#### 控制律
```
特征误差: e_s = s_desired - s_current
图像雅可比: L = ∂s/∂v (特征对机器人速度的偏导)
控制速度: v = -λ × L^+ × e_s
```

#### 优点
- 不需要精确的3D模型
- 对相机标定误差不敏感
- 收敛性好，稳定性强

#### 缺点
- 轨迹可能不是直线
- 图像雅可比矩阵计算复杂
- 可能出现局部最小值

#### 适用场景
- 目标跟踪任务
- 视觉导航
- 图像稳定控制

## 系统架构

```
robot_visual_servoing_integrated.py (主驱动程序)
│
├── RobotVisualServoing (主类)
│   ├── 机器人控制接口 (robotcontrol.py)
│   │   ├── connect_robot()
│   │   ├── send_velocity_command()
│   │   └── emergency_stop_robot()
│   │
│   ├── 视觉系统
│   │   ├── setup_camera()
│   │   ├── get_frames()
│   │   └── detect_target() (YOLO)
│   │
│   ├── PBVS模块
│   │   ├── estimate_target_pose_pbvs()
│   │   ├── compute_pose_error_pbvs()
│   │   ├── compute_control_law_pbvs()
│   │   └── visualize_pbvs()
│   │
│   └── IBVS模块
│       ├── extract_image_features_ibvs()
│       ├── compute_control_law_ibvs()
│       └── visualize_ibvs()
│
├── 辅助工具
│   ├── camera_calibration.py (相机标定)
│   ├── yolo_train.py (模型训练)
│   ├── yolo_detect.py (检测测试)
│   └── test_system.py (系统测试)
│
└── 基础模块
    ├── visual_servoing.py (基础实现)
    └── advanced_visual_servoing.py (高级功能)
```

## ⚙️ 参数配置

### 控制参数
```python
# 基本控制参数
lambda_pos = 0.3        # 位置/特征控制增益 (0.1-0.5)
lambda_rot = 0.2        # 旋转控制增益 (0.1-0.3)

# 速度限制
max_linear_velocity = 0.05      # 最大线速度 m/s
max_angular_velocity = 0.3      # 最大角速度 rad/s

# 安全参数
max_target_lost = 20    # 最大目标丢失次数
```

### PBVS专用参数
```python
# 期望位置 (相机坐标系)
desired_position = [0.0, 0.0, 0.4]  # [x, y, z] 米

# 目标3D模型点 (假设10cm×10cm正方形)
object_3d_points = [
    [-0.05, -0.05, 0], [0.05, -0.05, 0],
    [0.05, 0.05, 0], [-0.05, 0.05, 0]
]

# 死区设置
position_deadzone = 0.01        # 位置死区 1cm
orientation_deadzone = 0.05     # 姿态死区 ~3度
```

### IBVS专用参数
```python
# 期望图像特征 [x, y, width, height]
desired_features = [320, 240, 120, 120]  # 图像中心，120×120像素

# 死区设置
feature_deadzone = 8            # 特征死区 8像素
```

### 相机参数 (RealSense D435i默认)
```python
camera_matrix = [
    [615.0, 0, 320.0],
    [0, 615.0, 240.0],
    [0, 0, 1]
]
dist_coeffs = [0.1, -0.2, 0, 0, 0]
```

## 使用指南

### 基本操作流程

#### 1. 系统准备
```bash
# 检查硬件连接
- 确保机器人电源开启
- 连接RealSense相机
- 检查网络连接

# 激活环境并启动系统
conda activate vision
python robot_visual_servoing_integrated.py
```

#### 2. 参数设置
```python
# 根据具体任务调整参数
vs = RobotVisualServoing(
    robot_ip='192.168.1.100',  # 机器人IP
    robot_port=8899,           # 机器人端口
    model_path='yolo_train/weights/best.pt'  # YOLO模型
)

# 调整控制参数
vs.lambda_pos = 0.2  # 降低增益获得更稳定的控制
vs.max_linear_velocity = 0.03  # 降低速度提高安全性
```

#### 3. 运行控制
```python
# PBVS模式
vs.run_pbvs()

# IBVS模式  
vs.run_ibvs()
```

### 高级使用技巧

#### 1. 参数调优策略
```python
# 保守参数 - 稳定但较慢
conservative_params = {
    'lambda_pos': 0.1,
    'lambda_rot': 0.05,
    'max_linear_velocity': 0.02
}

# 标准参数 - 平衡性能
standard_params = {
    'lambda_pos': 0.3,
    'lambda_rot': 0.2,
    'max_linear_velocity': 0.05
}

# 激进参数 - 快速但可能不稳定
aggressive_params = {
    'lambda_pos': 0.5,
    'lambda_rot': 0.3,
    'max_linear_velocity': 0.08
}
```

#### 2. 目标检测优化
```python
# 提高检测稳定性
- 确保良好的光照条件
- 目标与背景有足够对比度
- 避免目标部分遮挡
- 保持相机稳定

# YOLO模型优化
- 使用针对特定目标训练的模型
- 调整置信度阈值
- 考虑使用更大的模型获得更高精度
```

#### 3. 安全操作建议
```python
# 工作空间设置
- 设置合理的工作空间边界
- 确保紧急停止按钮可及
- 监控机器人运动状态
- 准备手动接管

# 测试流程
1. 先在仿真环境中测试
2. 使用较低的速度参数开始
3. 逐步增加控制增益
4. 监控系统稳定性
```

## 🛠️ 故障排除

### 🔍 常见问题及解决方案

> 💡 **提示**: 遇到问题时，首先运行 `python test_system.py --all` 进行系统诊断

### ❓ 快速问题检查清单

在报告问题前，请检查以下项目：

```bash
# 1. 检查环境激活
conda info --envs
echo $CONDA_DEFAULT_ENV  # 应该显示 "vision"

# 2. 检查硬件连接
python -c "import pyrealsense2 as rs; print('相机正常')"  # 测试相机
ping <机械臂IP>  # 测试机械臂网络连接

# 3. 检查依赖包
python -c "import cv2, numpy, torch; print('依赖包正常')"

# 4. 检查YOLO模型
ls -la yolo_train/weights/best.pt  # 检查模型文件是否存在
```

#### 1. 机器人连接问题
**问题**: 无法连接到机器人
```bash
解决方案:
- 检查机器人IP地址和端口
- 确认机器人服务器运行状态
- 检查网络连接和防火墙设置
- 验证机器人控制权限
```

#### 2. 相机初始化失败
**问题**: RealSense相机无法初始化
```bash
解决方案:
- 检查相机USB连接
- 更新RealSense驱动程序
- 确认相机设备权限
- 尝试重新插拔相机
```

#### 3. 目标检测不稳定
**问题**: YOLO检测结果不稳定
```bash
解决方案:
- 改善光照条件
- 调整相机位置和角度
- 重新训练YOLO模型
- 调整检测置信度阈值
- 增加目标与背景的对比度
```

#### 4. 控制系统震荡
**问题**: 机器人运动不稳定，出现震荡
```bash
解决方案:
- 降低控制增益 (lambda_pos, lambda_rot)
- 增加死区范围
- 检查相机帧率和延迟
- 优化控制循环频率
- 检查机器人动力学参数
```

#### 5. 深度信息不准确
**问题**: 深度测量存在误差
```bash
解决方案:
- 检查目标表面材质 (避免反光、透明)
- 调整RealSense曝光参数
- 使用多点深度值平均
- 考虑环境光照影响
- 校准深度相机
```

#### 6. conda环境问题
**问题**: conda命令未找到或环境冲突
```bash
解决方案:
# 重新初始化conda
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc

# 检查环境是否正确激活
conda info --envs
which python

# 与ROS2冲突时的处理
- 确保在conda环境中运行项目
- 检查PYTHONPATH环境变量
- 使用 conda deactivate 退出环境后再使用ROS2
```

### 调试技巧

#### 1. 可视化调试
```python
# 观察关键信息
- 检测框位置和大小
- 期望位置显示
- 误差数值变化
- 控制命令大小

# 数据记录
- 保存误差历史
- 记录控制命令
- 分析收敛性能
```

#### 2. 分步测试
```python
# 测试步骤
1. 仅测试目标检测
2. 验证位姿估计精度
3. 检查控制律计算
4. 测试机器人响应
5. 整体系统集成
```

## 📚 API参考

### 主类: RobotVisualServoing

#### 构造函数
```python
def __init__(self, robot_ip='localhost', robot_port=8899, 
             model_path='yolo_train/weights/best.pt'):
    """
    初始化视觉伺服系统
    
    Args:
        robot_ip (str): 机器人IP地址
        robot_port (int): 机器人端口
        model_path (str): YOLO模型路径
    """
```

#### 核心方法
```python
def connect_robot(self) -> bool:
    """连接机器人，返回连接状态"""

def get_frames(self) -> Tuple[np.ndarray, np.ndarray]:
    """获取RGB-D图像帧"""

def detect_target(self, color_image) -> Optional[Dict]:
    """检测目标，返回检测结果"""

def send_velocity_command(self, velocity) -> bool:
    """发送速度命令到机器人"""

def emergency_stop_robot(self):
    """紧急停止机器人"""

def run_pbvs(self):
    """运行PBVS控制循环"""

def run_ibvs(self):
    """运行IBVS控制循环"""

def cleanup(self):
    """清理系统资源"""
```

## 🔬 扩展开发

### 1. 添加新的视觉伺服方法

```python
class CustomVisualServoing(RobotVisualServoing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自定义初始化
    
    def compute_custom_control_law(self, features):
        """实现自定义控制律"""
        # 自定义算法实现
        return control_velocity
    
    def run_custom(self):
        """运行自定义控制循环"""
        # 实现控制循环
        pass
```

### 2. 集成其他机器人平台

```python
class ROSVisualServoing(RobotVisualServoing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ROS初始化
        import rospy
        from geometry_msgs.msg import Twist
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    
    def send_velocity_command(self, velocity):
        """发送ROS速度命令"""
        twist = Twist()
        twist.linear.x = velocity[0]
        twist.linear.y = velocity[1]
        twist.linear.z = velocity[2]
        twist.angular.x = velocity[3]
        twist.angular.y = velocity[4]
        twist.angular.z = velocity[5]
        self.cmd_pub.publish(twist)
        return True
```

## 实际应用示例

### 机器人抓取任务
1. 使用YOLO检测目标物体
2. 使用PBVS将机器人移动到目标上方
3. 执行抓取动作

### 目标跟踪任务
1. 使用IBVS保持目标在图像中心
2. 维持固定的观察距离和角度

## 🎯 典型使用场景

### 场景1: 物体抓取
```python
# 1. 启动PBVS模式
# 2. 检测目标物体
# 3. 机械臂移动到目标上方
# 4. 执行抓取动作
```

### 场景2: 质量检测
```python
# 1. 使用IBVS保持固定视角
# 2. 拍摄高质量图像
# 3. 进行缺陷检测
```

### 场景3: 装配作业
```python
# 1. PBVS精确定位
# 2. 零件对齐
# 3. 装配操作
```

## 📞 技术支持

### 问题反馈
- **GitHub Issues**: [提交问题](https://github.com/als-ichkann/vision/issues)
- **功能请求**: 通过GitHub Issues提交

### 系统要求确认
- 确保硬件满足最低配置要求
- 网络连接稳定
- 所有依赖包正确安装

## 📖 参考文献

1. Chaumette, F., & Hutchinson, S. (2006). Visual servo control. IEEE Robotics & Automation Magazine.
2. Corke, P. (2017). Robotics, vision and control: fundamental algorithms In MATLAB.
3. Marchand, E., Spindler, F., & Chaumette, F. (2005). ViSP for visual servoing: a generic software platform.

## 📄 许可证和贡献

### 许可证
本项目基于现有的robotcontrol.py接口开发，请遵循相应的许可证要求。

### 🤝 贡献指南
欢迎提交问题报告和功能请求！在贡献代码前，请确保：

1. ✅ 代码符合PEP 8规范
2. ✅ 添加适当的注释和文档
3. ✅ 通过所有测试用例
4. ✅ 提供使用示例

### 🔄 版本更新
- 查看 [Releases](https://github.com/als-ichkann/vision/releases) 获取最新版本
- 定期更新依赖包以获得最佳性能

---

## ⚠️ 重要提醒

**🔒 安全第一**: 使用本系统前请确保充分了解机器人安全操作规程，并在安全的环境中进行测试。

**🐍 环境管理**: 每次运行项目前，请确保已激活conda环境：
```bash
conda activate vision
```

**📧 联系方式**: 如有技术问题，请通过GitHub Issues联系我们。

---
*最后更新: 2024年*