"""
YOLOv8医学细胞检测训练脚本
使用YOLOv8s架构进行迁移学习

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


class CellDetectionTrainer:
    """细胞检测模型训练器"""

    def __init__(self, data_config, model_config=None):
        """
        初始化训练器

        Args:
            data_config: 数据集配置文件路径
            model_config: 模型配置字典（可选）
        """
        self.data_config = data_config
        self.model_config = model_config or {}

        # 检查CUDA是否可用
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA版本: {torch.version.cuda}")

    def load_model(self, model_name='yolov8s.pt'):
        """
        加载预训练模型

        Args:
            model_name: 模型名称（yolov8n/s/m/l/x.pt）

        Returns:
            YOLO模型实例
        """
        print(f"\n加载模型: {model_name}")
        model = YOLO(model_name)
        return model

    def train(self, model_name='yolov8s.pt', **kwargs):
        """
        训练模型

        Args:
            model_name: 预训练模型名称
            **kwargs: 其他训练参数
        """
        # 合并配置
        config = {
            'epochs': 100,
            'batch': 16,
            'imgsz': 640,
            'workers': 8,
            'device': self.device,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'project': '../outputs/runs',
            'name': 'cell_detection',
            'save': True,
            'val': True,
            'plots': True,
            'verbose': True,
        }
        config.update(self.model_config)
        config.update(kwargs)

        # 加载模型
        model = self.load_model(model_name)

        # 打印训练配置
        print("\n" + "=" * 60)
        print("训练配置")
        print("=" * 60)
        for key, value in config.items():
            if key != 'data':
                print(f"{key}: {value}")
        print("=" * 60 + "\n")

        # 开始训练
        print("开始训练...\n")
        results = model.train(
            data=self.data_config,
            **config
        )

        return results, model

    def resume_training(self, checkpoint_path):
        """
        从检查点恢复训练

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"\n从检查点恢复训练: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        results = model.train(resume=True)
        return results, model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8医学细胞检测训练')

    # 必需参数
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径')

    # 模型参数
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                        help='预训练模型（n=nano, s=small, m=medium, l=large, x=xlarge）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮次（默认100）')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小（默认16）')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小（默认640）')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数（默认8）')

    # 优化器参数
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='优化器类型（默认AdamW）')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率（默认0.01）')
    parser.add_argument('--lrf', type=float, default=0.01,
                        help='最终学习率比例（默认0.01）')

    # 输出参数
    parser.add_argument('--project', type=str, default='../outputs/runs',
                        help='项目保存路径')
    parser.add_argument('--name', type=str, default='cell_detection',
                        help='实验名称')

    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" " * 15 + "YOLOv8医学细胞检测训练系统")
    print("=" * 70)

    # 检查数据配置文件是否存在
    if not os.path.exists(args.data):
        print(f"错误: 数据配置文件不存在: {args.data}")
        return

    # 如果是恢复训练
    if args.resume:
        trainer = CellDetectionTrainer(data_config=args.data)
        results, model = trainer.resume_training(args.resume)
    else:
        # 构建模型配置
        model_config = {
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'workers': args.workers,
            'optimizer': args.optimizer,
            'lr0': args.lr0,
            'lrf': args.lrf,
            'project': args.project,
            'name': args.name,
        }

        # 创建训练器
        trainer = CellDetectionTrainer(
            data_config=args.data,
            model_config=model_config
        )

        # 开始训练
        results, model = trainer.train(model_name=args.model)

    # 训练完成
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"\n模型保存位置: {args.project}/{args.name}")
    print("最佳权重: weights/best.pt")
    print("最终权重: weights/last.pt")
    print("\n可视化结果保存在输出目录中")


if __name__ == '__main__':
    # 示例用法
    import sys
    if len(sys.argv) == 1:
        print("=" * 70)
        print(" " * 15 + "YOLOv8医学细胞检测训练系统")
        print("=" * 70)
        print("\n使用方法:")
        print("python train.py --data <数据配置文件>")
        print("\n基础示例:")
        print("python train.py --data ../configs/dataset.yaml")
        print("\n完整示例:")
        print("python train.py --data ../configs/dataset.yaml \\")
        print("                --model yolov8s.pt \\")
        print("                --epochs 100 \\")
        print("                --batch 16 \\")
        print("                --imgsz 640 \\")
        print("                --optimizer AdamW \\")
        print("                --lr0 0.01")
        print("\n恢复训练:")
        print("python train.py --data ../configs/dataset.yaml --resume runs/train/weights/last.pt")
        print("\n模型大小选择:")
        print("yolov8n.pt - Nano   (最快，精度较低)")
        print("yolov8s.pt - Small  (推荐，平衡性能)")
        print("yolov8m.pt - Medium (较慢，精度较高)")
        print("yolov8l.pt - Large  (慢，高精度)")
        print("yolov8x.pt - XLarge (最慢，最高精度)")
    else:
        main()
