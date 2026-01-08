"""
训练结果可视化脚本
生成类别分布、位置分布、损失曲线等可视化图表

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ResultVisualizer:
    """结果可视化工具"""

    def __init__(self, data_dir, output_dir):
        """
        初始化可视化工具

        Args:
            data_dir: 数据目录（包含标注文件）
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.class_names = ['RBC', 'WBC', 'Platelet']
        self.class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    def parse_yolo_labels(self, labels_dir):
        """
        解析YOLO标注文件

        Args:
            labels_dir: 标注文件目录

        Returns:
            数据字典
        """
        data = defaultdict(list)
        label_files = list(Path(labels_dir).glob('*.txt'))

        for label_file in tqdm(label_files, desc="解析标注文件"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])

                        data['class_id'].append(class_id)
                        data['class_name'].append(self.class_names[class_id])
                        data['x_center'].append(x_center)
                        data['y_center'].append(y_center)
                        data['width'].append(width)
                        data['height'].append(height)

        return pd.DataFrame(data)

    def plot_class_distribution(self, df, title='类别分布'):
        """
        绘制类别分布直方图

        Args:
            df: 数据框
            title: 图表标题
        """
        plt.figure(figsize=(10, 6))

        class_counts = df['class_name'].value_counts()

        # 确保按顺序显示
        class_counts = class_counts.reindex(self.class_names, fill_value=0)

        bars = plt.bar(range(len(class_counts)), class_counts.values,
                       color=self.class_colors, alpha=0.8, edgecolor='black')

        plt.xlabel('细胞类别', fontsize=12)
        plt.ylabel('检测数量', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_counts)), class_counts.index, fontsize=11)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{int(count)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ 类别分布图已保存: {self.output_dir / 'class_distribution.png'}")
        plt.close()

    def plot_position_scatter(self, df, title='细胞位置分布'):
        """
        绘制细胞位置散点图

        Args:
            df: 数据框
            title: 图表标题
        """
        plt.figure(figsize=(10, 10))

        for i, class_name in enumerate(self.class_names):
            class_data = df[df['class_name'] == class_name]
            plt.scatter(class_data['x_center'], class_data['y_center'],
                       c=self.class_colors[i], label=class_name,
                       alpha=0.5, s=20, edgecolors='none')

        plt.xlabel('X坐标（归一化）', fontsize=12)
        plt.ylabel('Y坐标（归一化）', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='upper right')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.gca().invert_yaxis()  # 翻转Y轴使其符合图像坐标系
        plt.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'position_scatter.png', dpi=300, bbox_inches='tight')
        print(f"✓ 位置散点图已保存: {self.output_dir / 'position_scatter.png'}")
        plt.close()

    def plot_bbox_size_distribution(self, df, title='边界框尺寸分布'):
        """
        绘制边界框宽高分布图

        Args:
            df: 数据框
            title: 图表标题
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 宽度分布
        for i, class_name in enumerate(self.class_names):
            class_data = df[df['class_name'] == class_name]
            axes[0].hist(class_data['width'], bins=30, alpha=0.6,
                        color=self.class_colors[i], label=class_name, edgecolor='black')

        axes[0].set_xlabel('宽度（归一化）', fontsize=12)
        axes[0].set_ylabel('频数', fontsize=12)
        axes[0].set_title('边界框宽度分布', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3, linestyle='--')

        # 高度分布
        for i, class_name in enumerate(self.class_names):
            class_data = df[df['class_name'] == class_name]
            axes[1].hist(class_data['height'], bins=30, alpha=0.6,
                        color=self.class_colors[i], label=class_name, edgecolor='black')

        axes[1].set_xlabel('高度（归一化）', fontsize=12)
        axes[1].set_ylabel('频数', fontsize=12)
        axes[1].set_title('边界框高度分布', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'bbox_size_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ 边界框尺寸分布图已保存: {self.output_dir / 'bbox_size_distribution.png'}")
        plt.close()

    def plot_aspect_ratio(self, df, title='边界框宽高比分布'):
        """
        绘制边界框宽高比分布

        Args:
            df: 数据框
            title: 图表标题
        """
        df['aspect_ratio'] = df['width'] / (df['height'] + 1e-6)

        plt.figure(figsize=(12, 6))

        for i, class_name in enumerate(self.class_names):
            class_data = df[df['class_name'] == class_name]
            plt.hist(class_data['aspect_ratio'], bins=30, alpha=0.6,
                    color=self.class_colors[i], label=class_name, edgecolor='black')

        plt.xlabel('宽高比', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.xlim(0, 3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'aspect_ratio_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ 宽高比分布图已保存: {self.output_dir / 'aspect_ratio_distribution.png'}")
        plt.close()

    def generate_statistics_report(self, df):
        """
        生成统计报告

        Args:
            df: 数据框
        """
        report_path = self.output_dir / 'statistics_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(" " * 20 + "数据集统计报告\n")
            f.write("=" * 60 + "\n\n")

            # 总体统计
            f.write(f"总检测框数量: {len(df)}\n\n")

            # 各类别统计
            f.write("各类别统计:\n")
            f.write("-" * 60 + "\n")
            for class_name in self.class_names:
                class_data = df[df['class_name'] == class_name]
                count = len(class_data)
                percentage = count / len(df) * 100
                f.write(f"{class_name:10s}: {count:5d} ({percentage:5.2f}%)\n")

            f.write("\n" + "-" * 60 + "\n\n")

            # 尺寸统计
            f.write("边界框尺寸统计:\n")
            f.write("-" * 60 + "\n")
            for class_name in self.class_names:
                class_data = df[df['class_name'] == class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  平均宽度: {class_data['width'].mean():.4f}\n")
                f.write(f"  平均高度: {class_data['height'].mean():.4f}\n")
                f.write(f"  平均宽高比: {(class_data['width'] / class_data['height']).mean():.4f}\n")

            f.write("\n" + "=" * 60 + "\n")

        print(f"✓ 统计报告已保存: {report_path}")

    def visualize_all(self, labels_dir):
        """
        生成所有可视化图表

        Args:
            labels_dir: 标注文件目录
        """
        print("\n" + "=" * 60)
        print(" " * 20 + "数据可视化")
        print("=" * 60 + "\n")

        # 解析标注文件
        df = self.parse_yolo_labels(labels_dir)

        if len(df) == 0:
            print("错误: 未找到有效的标注数据")
            return

        print(f"\n解析到 {len(df)} 个检测框\n")

        # 生成各种图表
        self.plot_class_distribution(df)
        self.plot_position_scatter(df)
        self.plot_bbox_size_distribution(df)
        self.plot_aspect_ratio(df)
        self.generate_statistics_report(df)

        print("\n" + "=" * 60)
        print("可视化完成!")
        print(f"所有图表已保存到: {self.output_dir}")
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='数据可视化工具')

    parser.add_argument('--labels_dir', type=str, required=True,
                        help='标注文件目录（包含.txt文件）')
    parser.add_argument('--output_dir', type=str, default='../outputs/visualizations',
                        help='输出目录')

    return parser.parse_args()


def main():
    args = parse_args()

    # 创建可视化工具
    visualizer = ResultVisualizer(
        data_dir=args.labels_dir,
        output_dir=args.output_dir
    )

    # 生成所有可视化
    visualizer.visualize_all(args.labels_dir)


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("=" * 70)
        print(" " * 20 + "数据可视化工具")
        print("=" * 70)
        print("\n使用方法:")
        print("python visualize.py --labels_dir <标注目录>")
        print("\n示例:")
        print("python visualize.py --labels_dir ../data/train/labels")
        print("\n可选参数:")
        print("--output_dir <目录>  # 指定输出目录（默认../outputs/visualizations）")
    else:
        main()
