"""
数据集划分脚本
按照8:1:1比例划分训练集、验证集和测试集

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse


class DatasetSplitter:
    """数据集划分工具"""

    def __init__(self, source_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """
        初始化数据集划分器

        Args:
            source_dir: 源数据目录（包含images和labels文件夹）
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # 验证比例和
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "训练集、验证集、测试集比例之和必须为1"

        # 设置随机种子
        random.seed(seed)

        # 源目录
        self.source_images_dir = self.source_dir / 'images'
        self.source_labels_dir = self.source_dir / 'labels'

        # 目标目录
        self.train_dir = self.output_dir / 'train'
        self.val_dir = self.output_dir / 'val'
        self.test_dir = self.output_dir / 'test'

        # 创建目录结构
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            (split_dir / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'labels').mkdir(parents=True, exist_ok=True)

    def get_image_files(self):
        """获取所有图像文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(self.source_images_dir.glob(f'*{ext}'))
            image_files.extend(self.source_images_dir.glob(f'*{ext.upper()}'))

        return sorted(list(set(image_files)))

    def split_files(self, files):
        """
        划分文件列表

        Args:
            files: 文件列表

        Returns:
            train_files, val_files, test_files
        """
        # 打乱文件顺序
        random.shuffle(files)

        total = len(files)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        return train_files, val_files, test_files

    def copy_files(self, files, target_dir, split_name):
        """
        复制文件到目标目录

        Args:
            files: 文件列表
            target_dir: 目标目录
            split_name: 数据集名称（train/val/test）
        """
        target_images_dir = target_dir / 'images'
        target_labels_dir = target_dir / 'labels'

        for img_file in tqdm(files, desc=f"复制{split_name}集"):
            # 复制图像
            target_img_path = target_images_dir / img_file.name
            shutil.copy2(img_file, target_img_path)

            # 复制对应的标注文件
            label_file = self.source_labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                target_label_path = target_labels_dir / label_file.name
                shutil.copy2(label_file, target_label_path)
            else:
                print(f"警告: 未找到标注文件 {label_file}")

    def split(self):
        """执行数据集划分"""
        print("=" * 60)
        print("数据集划分工具")
        print("=" * 60)

        # 获取所有图像文件
        image_files = self.get_image_files()
        print(f"\n找到 {len(image_files)} 个图像文件")

        if len(image_files) == 0:
            print("错误: 未找到任何图像文件")
            return

        # 划分文件
        train_files, val_files, test_files = self.split_files(image_files)

        print(f"\n数据集划分比例: {self.train_ratio}:{self.val_ratio}:{self.test_ratio}")
        print(f"训练集: {len(train_files)} 张")
        print(f"验证集: {len(val_files)} 张")
        print(f"测试集: {len(test_files)} 张")

        # 复制文件
        print("\n开始复制文件...")
        self.copy_files(train_files, self.train_dir, "训练")
        self.copy_files(val_files, self.val_dir, "验证")
        self.copy_files(test_files, self.test_dir, "测试")

        print("\n数据集划分完成!")
        print(f"输出目录: {self.output_dir}")
        self.print_summary()

    def print_summary(self):
        """打印数据集统计信息"""
        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)

        for split_name, split_dir in [("训练集", self.train_dir),
                                       ("验证集", self.val_dir),
                                       ("测试集", self.test_dir)]:
            images = list((split_dir / 'images').glob('*'))
            labels = list((split_dir / 'labels').glob('*.txt'))
            print(f"{split_name}: {len(images)} 张图像, {len(labels)} 个标注文件")


def main():
    parser = argparse.ArgumentParser(description='数据集划分工具（8:1:1）')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='源数据目录（包含images和labels）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例（默认0.8）')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例（默认0.1）')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='测试集比例（默认0.1）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（默认42）')

    args = parser.parse_args()

    # 创建划分器并执行
    splitter = DatasetSplitter(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    splitter.split()


if __name__ == '__main__':
    # 示例用法
    import sys
    if len(sys.argv) == 1:
        print("=" * 60)
        print("数据集划分工具")
        print("=" * 60)
        print("\n使用方法:")
        print("python split_dataset.py --source_dir <源目录> --output_dir <输出目录>")
        print("\n示例:")
        print("python split_dataset.py --source_dir ../data/processed --output_dir ../data")
        print("\n可选参数:")
        print("--train_ratio 0.8  # 训练集比例")
        print("--val_ratio 0.1    # 验证集比例")
        print("--test_ratio 0.1   # 测试集比例")
        print("--seed 42          # 随机种子")
    else:
        main()
