# YOLO Dora Rust 项目

这是一个基于 Dora 框架和 Rust 编写的实时目标检测可视化系统，包含三个节点：摄像头节点、检测节点和可视化节点。

## 项目结构

项目包含三个主要节点：

1. **摄像头节点 (camera_node)** - 模拟摄像头数据输入
2. **检测节点 (detector_node)** - 模拟目标检测算法处理
3. **可视化节点 (visualizer_node)** - 在图像上绘制检测框并显示

## 功能特点

- 实时处理摄像头数据流
- 模拟目标检测（人、车等物体）
- 在图像上绘制检测框和标签
- 支持多种图像分辨率
- 使用 Dora 框架进行节点间通信

## 技术栈

- **语言**: Rust
- **框架**: Dora (数据流框架)
- **图像处理**: OpenCV (Rust 绑定)
- **数据格式**: Arrow 数组

## 数据流

1. **摄像头节点** → 发送图像帧数据到检测节点
2. **检测节点** → 处理图像并发送检测结果到可视化节点
3. **可视化节点** → 接收图像和检测结果，绘制检测框并显示

## 检测输出格式

每个检测结果包含：
- `class_name`: 物体类别名称（如 "person", "car"）
- `confidence`: 检测置信度 (0-1)
- `x, y`: 检测框中心的相对坐标
- `width, height`: 检测框的相对宽高

## 安装依赖

安装 Rust

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
安装 OpenCV 依赖
Ubuntu/Debian:

sudo apt-get install libopencv-dev build-essential cmake pkg-config libgtk-3-dev


## 编译和运行

### 编译项目

```bash
# 编译所有节点（发布版本）
cargo build --release
dora run complete-yolo-dataflow.yaml
