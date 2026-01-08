"""
YOLOv8细胞检测图形界面
基于PyQt5的可视化检测工具

Author: Medical Cell Detection System
Date: 2026-01-08
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QGroupBox,
    QTextEdit, QSplitter, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

# 添加父目录到路径以导入检测器
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))
from inference import CellDetector


class DetectionThread(QThread):
    """后台检测线程"""
    finished = pyqtSignal(object, object)  # results, annotated_image
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, detector, image_path):
        super().__init__()
        self.detector = detector
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit(50)
            results = self.detector.detect_single_image(self.image_path)

            if results is None:
                self.error.emit("检测失败")
                return

            # 读取原始图像并绘制结果
            image = cv2.imread(str(self.image_path))
            annotated_image = self.detector.draw_detections(image, results)

            self.progress.emit(100)
            self.finished.emit(results, annotated_image)

        except Exception as e:
            self.error.emit(f"检测出错: {str(e)}")


class CellDetectionGUI(QMainWindow):
    """细胞检测图形界面"""

    def __init__(self):
        super().__init__()
        self.detector = None
        self.current_image = None
        self.current_results = None
        self.detection_thread = None

        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle('医学细胞检测系统 - YOLOv8')
        self.setGeometry(100, 100, 1400, 800)

        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 主布局
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)

        # 标题
        title_label = QLabel('YOLOv8医学细胞检测系统')
        title_font = QFont('Arial', 18, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        main_layout.addWidget(title_label)

        # 创建水平分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板 - 控制区
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        # 右侧面板 - 图像显示区
        right_panel = self.create_image_panel()
        splitter.addWidget(right_panel)

        # 设置分割器比例
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # 状态栏
        self.statusBar().showMessage('就绪')

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 模型加载组
        model_group = QGroupBox('模型设置')
        model_layout = QVBoxLayout()

        self.model_path_label = QLabel('未加载模型')
        self.model_path_label.setWordWrap(True)
        model_layout.addWidget(self.model_path_label)

        load_model_btn = QPushButton('加载模型')
        load_model_btn.clicked.connect(self.load_model)
        load_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        model_layout.addWidget(load_model_btn)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # 参数设置组
        params_group = QGroupBox('检测参数')
        params_layout = QVBoxLayout()

        # 置信度阈值
        conf_label = QLabel('置信度阈值: 0.25')
        params_layout.addWidget(conf_label)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(
            lambda v: conf_label.setText(f'置信度阈值: {v/100:.2f}')
        )
        params_layout.addWidget(self.conf_slider)

        # IOU阈值
        iou_label = QLabel('IOU阈值: 0.45')
        params_layout.addWidget(iou_label)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(1)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(
            lambda v: iou_label.setText(f'IOU阈值: {v/100:.2f}')
        )
        params_layout.addWidget(self.iou_slider)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # 图像操作组
        image_group = QGroupBox('图像操作')
        image_layout = QVBoxLayout()

        load_image_btn = QPushButton('加载图像')
        load_image_btn.clicked.connect(self.load_image)
        load_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        image_layout.addWidget(load_image_btn)

        detect_btn = QPushButton('开始检测')
        detect_btn.clicked.connect(self.run_detection)
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        image_layout.addWidget(detect_btn)

        save_btn = QPushButton('保存结果')
        save_btn.clicked.connect(self.save_result)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #d68910;
            }
        """)
        image_layout.addWidget(save_btn)

        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 结果显示
        results_group = QGroupBox('检测结果')
        results_layout = QVBoxLayout()

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        layout.addStretch()

        return panel

    def create_image_panel(self):
        """创建图像显示面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # 图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                background-color: #ecf0f1;
                min-height: 500px;
            }
        """)
        self.image_label.setText('请加载图像')
        layout.addWidget(self.image_label)

        return panel

    def load_model(self):
        """加载模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择模型文件',
            '',
            'PyTorch模型 (*.pt);;所有文件 (*.*)'
        )

        if model_path:
            try:
                conf = self.conf_slider.value() / 100
                iou = self.iou_slider.value() / 100

                self.detector = CellDetector(
                    model_path=model_path,
                    conf_threshold=conf,
                    iou_threshold=iou
                )

                self.model_path_label.setText(f'已加载: {Path(model_path).name}')
                self.statusBar().showMessage('模型加载成功')
                QMessageBox.information(self, '成功', '模型加载成功!')

            except Exception as e:
                QMessageBox.critical(self, '错误', f'模型加载失败: {str(e)}')

    def load_image(self):
        """加载图像"""
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            '选择图像文件',
            '',
            '图像文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)'
        )

        if image_path:
            self.current_image = image_path
            self.display_image(image_path)
            self.statusBar().showMessage(f'已加载图像: {Path(image_path).name}')

    def display_image(self, image_path):
        """显示图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return

        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整图像大小以适应标签
        h, w, ch = image_rgb.shape
        max_width = self.image_label.width() - 20
        max_height = self.image_label.height() - 20

        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        image_resized = cv2.resize(image_rgb, (new_w, new_h))

        # 转换为QPixmap并显示
        bytes_per_line = ch * new_w
        q_image = QImage(image_resized.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def run_detection(self):
        """运行检测"""
        if self.detector is None:
            QMessageBox.warning(self, '警告', '请先加载模型!')
            return

        if self.current_image is None:
            QMessageBox.warning(self, '警告', '请先加载图像!')
            return

        # 更新检测器参数
        self.detector.conf_threshold = self.conf_slider.value() / 100
        self.detector.iou_threshold = self.iou_slider.value() / 100

        # 启动后台检测线程
        self.progress_bar.setValue(0)
        self.statusBar().showMessage('正在检测...')

        self.detection_thread = DetectionThread(self.detector, self.current_image)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.progress.connect(self.progress_bar.setValue)
        self.detection_thread.error.connect(self.on_detection_error)
        self.detection_thread.start()

    def on_detection_finished(self, results, annotated_image):
        """检测完成回调"""
        self.current_results = (results, annotated_image)

        # 显示标注后的图像
        self.display_annotated_image(annotated_image)

        # 显示检测结果统计
        self.display_results(results)

        self.statusBar().showMessage('检测完成')
        self.progress_bar.setValue(100)

    def on_detection_error(self, error_msg):
        """检测错误回调"""
        QMessageBox.critical(self, '错误', error_msg)
        self.statusBar().showMessage('检测失败')
        self.progress_bar.setValue(0)

    def display_annotated_image(self, annotated_image):
        """显示标注后的图像"""
        # 转换颜色空间
        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # 调整大小
        h, w, ch = image_rgb.shape
        max_width = self.image_label.width() - 20
        max_height = self.image_label.height() - 20

        scale = min(max_width / w, max_height / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        image_resized = cv2.resize(image_rgb, (new_w, new_h))

        # 显示
        bytes_per_line = ch * new_w
        q_image = QImage(image_resized.data, new_w, new_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def display_results(self, results):
        """显示检测结果"""
        text = "=" * 50 + "\n"
        text += "检测结果统计\n"
        text += "=" * 50 + "\n\n"

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            text += "未检测到细胞\n"
            self.results_text.setText(text)
            return

        # 统计各类别数量
        class_counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0}
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = self.detector.class_names[class_id]
            class_counts[class_name] += 1

        total = len(boxes)
        text += f"总检测数量: {total}\n\n"
        text += "各类别统计:\n"
        text += "-" * 50 + "\n"

        for class_name, count in class_counts.items():
            percentage = count / total * 100 if total > 0 else 0
            text += f"{class_name:10s}: {count:4d} ({percentage:5.2f}%)\n"

        text += "\n" + "=" * 50 + "\n"
        text += "详细检测信息:\n"
        text += "=" * 50 + "\n\n"

        for i, box in enumerate(boxes, 1):
            class_id = int(box.cls[0])
            class_name = self.detector.class_names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            text += f"[{i}] {class_name} - 置信度: {confidence:.3f}\n"
            text += f"    位置: ({x1}, {y1}) - ({x2}, {y2})\n\n"

        self.results_text.setText(text)

    def save_result(self):
        """保存结果"""
        if self.current_results is None:
            QMessageBox.warning(self, '警告', '没有可保存的结果!')
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            '保存结果',
            '',
            '图像文件 (*.jpg *.png);;所有文件 (*.*)'
        )

        if save_path:
            _, annotated_image = self.current_results
            cv2.imwrite(save_path, annotated_image)
            QMessageBox.information(self, '成功', f'结果已保存到:\n{save_path}')


def main():
    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle('Fusion')

    window = CellDetectionGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
