"""
LabelMe标注格式转YOLO格式脚本
将LabelMe的JSON标注文件转换为YOLO所需的txt格式

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


class LabelMe2YOLO:
    """LabelMe到YOLO格式转换器"""

    def __init__(self, labelme_dir, output_dir, classes):
        """
        初始化转换器

        Args:
            labelme_dir: LabelMe标注文件目录
            output_dir: 输出目录
            classes: 类别列表 ['RBC', 'WBC', 'Platelet']
        """
        self.labelme_dir = Path(labelme_dir)
        self.output_dir = Path(output_dir)
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # 创建输出目录
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

    def convert_polygon_to_bbox(self, points):
        """
        将多边形坐标转换为边界框

        Args:
            points: 多边形点坐标列表 [[x1,y1], [x2,y2], ...]

        Returns:
            x_min, y_min, x_max, y_max
        """
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def normalize_bbox(self, bbox, img_width, img_height):
        """
        归一化边界框坐标到[0, 1]范围

        Args:
            bbox: (x_min, y_min, x_max, y_max)
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            x_center, y_center, width, height (归一化后)
        """
        x_min, y_min, x_max, y_max = bbox

        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min

        # 归一化
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        return x_center_norm, y_center_norm, width_norm, height_norm

    def convert_single_file(self, json_path):
        """
        转换单个LabelMe JSON文件

        Args:
            json_path: JSON文件路径

        Returns:
            success: 是否转换成功
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像信息
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            image_filename = data['imagePath']

            # 准备YOLO标注内容
            yolo_annotations = []

            # 遍历所有标注对象
            for shape in data['shapes']:
                label = shape['label']
                points = shape['points']

                # 检查类别是否有效
                if label not in self.class_to_idx:
                    print(f"警告: 未知类别 '{label}' 在文件 {json_path}")
                    continue

                class_idx = self.class_to_idx[label]

                # 转换坐标
                bbox = self.convert_polygon_to_bbox(points)
                x_center, y_center, width, height = self.normalize_bbox(
                    bbox, img_width, img_height
                )

                # YOLO格式: class_id x_center y_center width height
                yolo_annotations.append(
                    f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            # 保存标注文件
            txt_filename = json_path.stem + '.txt'
            txt_path = self.labels_dir / txt_filename
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_annotations))

            # 复制图像文件
            source_img_path = json_path.parent / image_filename
            if source_img_path.exists():
                target_img_path = self.images_dir / image_filename
                shutil.copy2(source_img_path, target_img_path)
            else:
                print(f"警告: 图像文件不存在 {source_img_path}")
                return False

            return True

        except Exception as e:
            print(f"错误: 转换文件 {json_path} 失败: {str(e)}")
            return False

    def convert_all(self):
        """转换目录中的所有JSON文件"""
        json_files = list(self.labelme_dir.glob('*.json'))

        if not json_files:
            print(f"警告: 在 {self.labelme_dir} 中未找到JSON文件")
            return

        print(f"找到 {len(json_files)} 个标注文件")
        print(f"开始转换...")

        success_count = 0
        for json_path in tqdm(json_files, desc="转换进度"):
            if self.convert_single_file(json_path):
                success_count += 1

        print(f"\n转换完成!")
        print(f"成功: {success_count}/{len(json_files)}")
        print(f"图像保存在: {self.images_dir}")
        print(f"标注保存在: {self.labels_dir}")


def main():
    parser = argparse.ArgumentParser(description='LabelMe到YOLO格式转换')
    parser.add_argument('--labelme_dir', type=str, required=True,
                        help='LabelMe标注文件目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='YOLO格式输出目录')
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['RBC', 'WBC', 'Platelet'],
                        help='类别列表（按索引顺序）')

    args = parser.parse_args()

    # 创建转换器并执行转换
    converter = LabelMe2YOLO(
        labelme_dir=args.labelme_dir,
        output_dir=args.output_dir,
        classes=args.classes
    )
    converter.convert_all()


if __name__ == '__main__':
    # 示例用法
    print("=" * 60)
    print("LabelMe到YOLO格式转换工具")
    print("=" * 60)

    # 如果直接运行（不带参数），使用默认路径
    import sys
    if len(sys.argv) == 1:
        print("\n使用方法:")
        print("python labelme_to_yolo.py --labelme_dir <标注目录> --output_dir <输出目录>")
        print("\n示例:")
        print("python labelme_to_yolo.py --labelme_dir ../data/raw --output_dir ../data/processed")
    else:
        main()
