"""
YOLOv8模型推理脚本
使用训练好的模型对新图像进行细胞检测

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
import json


class CellDetector:
    """细胞检测器"""

    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        初始化检测器

        Args:
            model_path: 模型权重文件路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)

        # 类别名称
        self.class_names = ['RBC', 'WBC', 'Platelet']
        self.class_colors = {
            'RBC': (255, 107, 107),      # 红色
            'WBC': (78, 205, 196),       # 青色
            'Platelet': (69, 183, 209)   # 蓝色
        }

        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}\n")

    def detect_single_image(self, image_path, save_path=None, show=False):
        """
        检测单张图像

        Args:
            image_path: 图像路径
            save_path: 保存路径（可选）
            show: 是否显示结果

        Returns:
            检测结果
        """
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None

        # 进行检测
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]

        # 绘制检测结果
        annotated_image = self.draw_detections(image, results)

        # 保存结果
        if save_path:
            cv2.imwrite(str(save_path), annotated_image)

        # 显示结果
        if show:
            cv2.imshow('Cell Detection', annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results

    def draw_detections(self, image, results):
        """
        在图像上绘制检测框

        Args:
            image: 原始图像
            results: 检测结果

        Returns:
            标注后的图像
        """
        annotated_image = image.copy()

        # 获取检测框
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # 提取信息
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                color = self.class_colors[class_name]

                # 绘制边界框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 准备标签文本
                label = f'{class_name} {confidence:.2f}'

                # 计算标签背景框大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # 绘制标签背景
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # 绘制标签文本
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        return annotated_image

    def detect_batch(self, input_dir, output_dir):
        """
        批量检测图像

        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f'*{ext}'))
            image_files.extend(input_dir.glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"错误: 在 {input_dir} 中未找到图像文件")
            return

        print(f"找到 {len(image_files)} 张图像")
        print("开始批量检测...\n")

        # 统计信息
        detection_stats = {class_name: 0 for class_name in self.class_names}
        total_detections = 0

        # 批量处理
        for image_file in tqdm(image_files, desc="检测进度"):
            # 检测
            save_path = output_dir / image_file.name
            results = self.detect_single_image(image_file, save_path=save_path)

            # 统计
            if results and results.boxes is not None:
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    detection_stats[class_name] += 1
                    total_detections += 1

        # 打印统计信息
        self.print_batch_statistics(detection_stats, total_detections, len(image_files))

    def print_batch_statistics(self, detection_stats, total_detections, num_images):
        """
        打印批量检测统计信息

        Args:
            detection_stats: 检测统计字典
            total_detections: 总检测数
            num_images: 图像总数
        """
        print("\n" + "=" * 60)
        print("批量检测统计")
        print("=" * 60)
        print(f"\n处理图像数: {num_images}")
        print(f"总检测数量: {total_detections}")
        print(f"平均每张: {total_detections / num_images:.2f}\n")

        print("各类别检测数量:")
        for class_name, count in detection_stats.items():
            percentage = count / total_detections * 100 if total_detections > 0 else 0
            print(f"  {class_name:10s}: {count:5d} ({percentage:5.2f}%)")

        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv8细胞检测推理')

    parser.add_argument('--model', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--source', type=str, required=True,
                        help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='../outputs/predictions',
                        help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值（默认0.25）')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS的IOU阈值（默认0.45）')
    parser.add_argument('--show', action='store_true',
                        help='显示检测结果')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print(" " * 20 + "细胞检测推理系统")
    print("=" * 70 + "\n")

    # 创建检测器
    detector = CellDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # 判断是单张图像还是目录
    source_path = Path(args.source)

    if source_path.is_file():
        # 单张图像检测
        print(f"检测单张图像: {source_path}\n")
        output_path = Path(args.output) / source_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        detector.detect_single_image(
            source_path,
            save_path=output_path,
            show=args.show
        )

        print(f"\n检测完成!")
        print(f"结果已保存到: {output_path}")

    elif source_path.is_dir():
        # 批量检测
        detector.detect_batch(source_path, args.output)
        print(f"\n批量检测完成!")
        print(f"结果已保存到: {args.output}")

    else:
        print(f"错误: 无效的输入路径 {source_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print("=" * 70)
        print(" " * 20 + "细胞检测推理系统")
        print("=" * 70)
        print("\n使用方法:")
        print("python inference.py --model <模型路径> --source <图像路径或目录>")
        print("\n单张图像检测示例:")
        print("python inference.py \\")
        print("    --model ../outputs/runs/cell_detection/weights/best.pt \\")
        print("    --source ../data/test/images/cell_001.jpg \\")
        print("    --output ../outputs/predictions \\")
        print("    --conf 0.5")
        print("\n批量检测示例:")
        print("python inference.py \\")
        print("    --model ../outputs/runs/cell_detection/weights/best.pt \\")
        print("    --source ../data/test/images \\")
        print("    --output ../outputs/predictions")
        print("\n可选参数:")
        print("--conf 0.25   # 置信度阈值（默认0.25）")
        print("--iou 0.45    # NMS的IOU阈值（默认0.45）")
        print("--show        # 显示检测结果窗口")
    else:
        main()
