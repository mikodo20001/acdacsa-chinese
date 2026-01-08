"""
YOLOv8医学细胞检测系统 - 主运行脚本
一键启动各种功能的统一入口

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import sys
import argparse
from pathlib import Path


def print_banner():
    """打印横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         YOLOv8 医学细胞检测与分类系统                        ║
║         Medical Cell Detection & Classification System        ║
║                                                               ║
║         基于YOLOv8深度学习框架                                ║
║         精确识别 RBC、WBC、Platelet 三类细胞                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_menu():
    """打印主菜单"""
    menu = """
请选择要执行的功能:

【数据处理】
  1. LabelMe标注转YOLO格式
  2. 数据集划分（8:1:1）
  3. 数据可视化分析

【模型训练】
  4. 开始训练（默认配置）
  5. 开始训练（自定义参数）
  6. 恢复训练

【模型评估】
  7. 模型评估
  8. 查看训练结果

【模型推理】
  9. 单张图像检测
  10. 批量图像检测
  11. 启动图形界面

【其他】
  12. 查看使用手册
  13. 检查环境配置
  0. 退出

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(menu)


def convert_labelme():
    """转换LabelMe标注"""
    print("\n【LabelMe转YOLO格式】")
    labelme_dir = input("请输入LabelMe标注目录 [默认: data/raw]: ").strip() or "data/raw"
    output_dir = input("请输入输出目录 [默认: data/processed]: ").strip() or "data/processed"

    cmd = f'python scripts/labelme_to_yolo.py --labelme_dir {labelme_dir} --output_dir {output_dir}'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def split_dataset():
    """划分数据集"""
    print("\n【数据集划分】")
    source_dir = input("请输入源数据目录 [默认: data/processed]: ").strip() or "data/processed"
    output_dir = input("请输入输出目录 [默认: data]: ").strip() or "data"

    cmd = f'python scripts/split_dataset.py --source_dir {source_dir} --output_dir {output_dir}'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def visualize_data():
    """数据可视化"""
    print("\n【数据可视化】")
    labels_dir = input("请输入标注目录 [默认: data/train/labels]: ").strip() or "data/train/labels"

    cmd = f'python scripts/visualize.py --labels_dir {labels_dir}'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def train_default():
    """默认配置训练"""
    print("\n【开始训练 - 默认配置】")
    print("使用配置:")
    print("  - 模型: yolov8s.pt")
    print("  - Epochs: 100")
    print("  - Batch: 16")
    print("  - Image Size: 640")

    confirm = input("\n确认开始训练? (y/n): ").strip().lower()
    if confirm == 'y':
        cmd = 'python scripts/train.py --data configs/dataset.yaml'
        print(f"\n执行命令: {cmd}\n")
        import os
        os.system(cmd)


def train_custom():
    """自定义参数训练"""
    print("\n【开始训练 - 自定义参数】")

    model = input("选择模型 [yolov8n/s/m/l/x, 默认: s]: ").strip() or "s"
    epochs = input("训练轮数 [默认: 100]: ").strip() or "100"
    batch = input("批次大小 [默认: 16]: ").strip() or "16"
    imgsz = input("图像大小 [默认: 640]: ").strip() or "640"

    cmd = f'python scripts/train.py --data configs/dataset.yaml --model yolov8{model}.pt --epochs {epochs} --batch {batch} --imgsz {imgsz}'
    print(f"\n执行命令: {cmd}\n")

    confirm = input("确认开始训练? (y/n): ").strip().lower()
    if confirm == 'y':
        import os
        os.system(cmd)


def resume_training():
    """恢复训练"""
    print("\n【恢复训练】")
    checkpoint = input("请输入检查点路径 [默认: outputs/runs/cell_detection/weights/last.pt]: ").strip()
    checkpoint = checkpoint or "outputs/runs/cell_detection/weights/last.pt"

    cmd = f'python scripts/train.py --data configs/dataset.yaml --resume {checkpoint}'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def evaluate_model():
    """评估模型"""
    print("\n【模型评估】")
    model = input("请输入模型路径 [默认: outputs/runs/cell_detection/weights/best.pt]: ").strip()
    model = model or "outputs/runs/cell_detection/weights/best.pt"

    split = input("选择数据集 [train/val/test, 默认: test]: ").strip() or "test"

    cmd = f'python scripts/evaluate.py --model {model} --data configs/dataset.yaml --split {split}'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def view_results():
    """查看训练结果"""
    print("\n【查看训练结果】")
    print("训练结果保存在: outputs/runs/cell_detection/")
    print("\n包含以下文件:")
    print("  - results.png: 训练曲线")
    print("  - confusion_matrix.png: 混淆矩阵")
    print("  - F1_curve.png, P_curve.png, R_curve.png: 性能曲线")
    print("\n正在打开结果目录...")

    import os
    import platform
    results_dir = "outputs/runs/cell_detection"

    if platform.system() == 'Windows':
        os.system(f'explorer {results_dir}')
    elif platform.system() == 'Darwin':  # macOS
        os.system(f'open {results_dir}')
    else:  # Linux
        os.system(f'xdg-open {results_dir}')


def inference_single():
    """单张图像检测"""
    print("\n【单张图像检测】")
    model = input("请输入模型路径 [默认: outputs/runs/cell_detection/weights/best.pt]: ").strip()
    model = model or "outputs/runs/cell_detection/weights/best.pt"

    source = input("请输入图像路径: ").strip()
    if not source:
        print("错误: 必须指定图像路径")
        return

    conf = input("置信度阈值 [默认: 0.25]: ").strip() or "0.25"

    cmd = f'python scripts/inference.py --model {model} --source {source} --conf {conf} --show'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def inference_batch():
    """批量图像检测"""
    print("\n【批量图像检测】")
    model = input("请输入模型路径 [默认: outputs/runs/cell_detection/weights/best.pt]: ").strip()
    model = model or "outputs/runs/cell_detection/weights/best.pt"

    source = input("请输入图像目录: ").strip()
    if not source:
        print("错误: 必须指定图像目录")
        return

    output = input("输出目录 [默认: outputs/predictions]: ").strip() or "outputs/predictions"
    conf = input("置信度阈值 [默认: 0.25]: ").strip() or "0.25"

    cmd = f'python scripts/inference.py --model {model} --source {source} --output {output} --conf {conf}'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def launch_gui():
    """启动图形界面"""
    print("\n【启动图形界面】")
    print("正在启动PyQt5图形界面...")

    cmd = 'python gui/cell_detection_gui.py'
    print(f"\n执行命令: {cmd}\n")
    import os
    os.system(cmd)


def view_manual():
    """查看使用手册"""
    print("\n【使用手册】")
    print("使用手册位置: docs/使用手册.md")
    print("README文档位置: README.md")
    print("\n请使用Markdown阅读器查看文档")


def check_environment():
    """检查环境配置"""
    print("\n【环境配置检查】")
    print("正在检查Python环境...\n")

    import sys
    print(f"Python版本: {sys.version}")

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")

    try:
        from ultralytics import YOLO
        print("✓ YOLOv8已安装")
    except ImportError:
        print("❌ Ultralytics未安装")

    try:
        import cv2
        print(f"✓ OpenCV版本: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV未安装")

    try:
        from PyQt5 import QtCore
        print(f"✓ PyQt5版本: {QtCore.QT_VERSION_STR}")
    except ImportError:
        print("❌ PyQt5未安装")

    print("\n环境检查完成!")


def main():
    """主函数"""
    print_banner()

    while True:
        print_menu()
        choice = input("请输入选项 [0-13]: ").strip()

        if choice == '0':
            print("\n感谢使用! 再见!")
            break
        elif choice == '1':
            convert_labelme()
        elif choice == '2':
            split_dataset()
        elif choice == '3':
            visualize_data()
        elif choice == '4':
            train_default()
        elif choice == '5':
            train_custom()
        elif choice == '6':
            resume_training()
        elif choice == '7':
            evaluate_model()
        elif choice == '8':
            view_results()
        elif choice == '9':
            inference_single()
        elif choice == '10':
            inference_batch()
        elif choice == '11':
            launch_gui()
        elif choice == '12':
            view_manual()
        elif choice == '13':
            check_environment()
        else:
            print("\n❌ 无效选项，请重新选择\n")

        input("\n按Enter键继续...")
        print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
