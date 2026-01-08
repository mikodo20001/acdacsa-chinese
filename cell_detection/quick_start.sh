#!/bin/bash
# YOLOv8医学细胞检测系统 - 快速启动脚本
# Quick Start Script for Cell Detection System

echo "============================================================"
echo "     YOLOv8 医学细胞检测系统 - 快速启动"
echo "     Medical Cell Detection System - Quick Start"
echo "============================================================"
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python 3.8或更高版本"
    exit 1
fi

echo "✓ Python已安装: $(python3 --version)"
echo ""

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
    echo "✓ 虚拟环境创建完成"
else
    echo "✓ 虚拟环境已存在"
fi
echo ""

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate
echo ""

# 安装依赖
echo "📥 安装依赖包..."
pip install --upgrade pip
pip install -r requirements.txt
echo ""

# 创建必要的目录
echo "📁 创建目录结构..."
mkdir -p data/{raw,processed,train,val,test}
mkdir -p outputs/{runs,evaluation,visualizations,predictions}
mkdir -p models
echo "✓ 目录创建完成"
echo ""

# 检查安装
echo "🔍 检查安装状态..."
python3 -c "import torch; print('✓ PyTorch:', torch.__version__)"
python3 -c "from ultralytics import YOLO; print('✓ YOLOv8已安装')"
python3 -c "import cv2; print('✓ OpenCV:', cv2.__version__)"
echo ""

echo "============================================================"
echo "✅ 环境配置完成！"
echo "============================================================"
echo ""
echo "下一步操作："
echo "1. 准备数据并使用LabelMe进行标注"
echo "2. 运行主程序: python run.py"
echo "3. 或启动图形界面: python gui/cell_detection_gui.py"
echo ""
echo "查看文档："
echo "- README.md - 项目说明"
echo "- docs/使用手册.md - 详细使用指南"
echo ""
