"""
YOLOv8模型评估脚本
评估训练好的模型在测试集上的性能

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model_path, data_config):
        """
        初始化评估器

        Args:
            model_path: 模型权重文件路径
            data_config: 数据集配置文件路径
        """
        self.model_path = model_path
        self.data_config = data_config

        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(data_config):
            raise FileNotFoundError(f"数据配置文件不存在: {data_config}")

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)

        # 检查设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}\n")

    def evaluate(self, split='test', save_dir='../outputs/evaluation'):
        """
        评估模型

        Args:
            split: 数据集划分（train/val/test）
            save_dir: 结果保存目录

        Returns:
            评估结果
        """
        print("=" * 60)
        print(f"评估模型在 {split} 集上的性能")
        print("=" * 60)

        # 运行评估
        results = self.model.val(
            data=self.data_config,
            split=split,
            device=self.device,
            save_json=True,
            save_hybrid=True,
            project=save_dir,
            name=f'{split}_evaluation'
        )

        return results

    def print_metrics(self, results):
        """
        打印评估指标

        Args:
            results: 评估结果
        """
        print("\n" + "=" * 60)
        print("评估指标")
        print("=" * 60)

        # 提取关键指标
        metrics = results.results_dict

        print(f"\n整体性能:")
        print(f"  mAP@0.5      : {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision    : {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall       : {metrics.get('metrics/recall(B)', 0):.4f}")

        # 各类别性能
        if hasattr(results, 'ap_class_index') and hasattr(results, 'ap50'):
            print(f"\n各类别性能 (mAP@0.5):")
            class_names = ['RBC', 'WBC', 'Platelet']
            for i, ap in enumerate(results.ap50):
                if i < len(class_names):
                    print(f"  {class_names[i]:10s}: {ap:.4f}")

        print("\n" + "=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8模型评估')

    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='评估的数据集划分（默认test）')
    parser.add_argument('--save_dir', type=str, default='../outputs/evaluation',
                        help='结果保存目录')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" " * 20 + "模型评估系统")
    print("=" * 70)

    # 创建评估器
    evaluator = ModelEvaluator(
        model_path=args.model,
        data_config=args.data
    )

    # 运行评估
    results = evaluator.evaluate(
        split=args.split,
        save_dir=args.save_dir
    )

    # 打印指标
    evaluator.print_metrics(results)

    print(f"\n评估结果已保存到: {args.save_dir}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("=" * 70)
        print(" " * 20 + "模型评估系统")
        print("=" * 70)
        print("\n使用方法:")
        print("python evaluate.py --model <模型路径> --data <数据配置>")
        print("\n示例:")
        print("python evaluate.py \\")
        print("    --model ../outputs/runs/cell_detection/weights/best.pt \\")
        print("    --data ../configs/dataset.yaml \\")
        print("    --split test")
        print("\n可选参数:")
        print("--split [train|val|test]  # 选择评估的数据集（默认test）")
        print("--save_dir <目录>         # 指定结果保存目录")
    else:
        main()
