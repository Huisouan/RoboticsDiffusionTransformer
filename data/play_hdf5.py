import sys
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtGui import QImage, QPixmap

class HDF5Viewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDF5 Robot Data Viewer")
        self.setGeometry(100, 100, 1600, 900)
        
        self.current_frame = 0
        self.hdf5_data = None
        self.total_frames = 0
        self.camera_names = []
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # UI布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # 视频显示区域（3摄像头）
        self.video_layout = QHBoxLayout()
        self.video_labels = [QLabel(f"Camera {i}") for i in range(3)]
        for label in self.video_labels:
            label.setFixedSize(640, 480)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_layout.addWidget(label)
        self.main_layout.addLayout(self.video_layout)
        
        # 控制区域（仅保留核心按钮）
        self.btn_load = QPushButton("Load HDF5 File")
        self.btn_play_pause = QPushButton("Play/Pause")
        self.btn_play_pause.setCheckable(True)
        
        self.btn_load.clicked.connect(self.load_file)
        self.btn_play_pause.clicked.connect(self.toggle_play)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_play_pause)
        self.main_layout.addLayout(control_layout)
        
        # 数据曲线区域
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.main_layout.addWidget(self.canvas)
        
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play_pause.setText("Play")
        else:
            self.timer.start(33)  # ~30 FPS
            self.btn_play_pause.setText("Pause")
            
    def update_frame(self):
        
        if self.current_frame < self.total_frames:
            for i, cam_name in enumerate(self.camera_names):
                frame_data = self.hdf5_data[f'observation/images/{cam_name}'][self.current_frame]
                
                # CHW → HWC 转置
                frame_data = frame_data.transpose(1, 2, 0)  # (3,480,640) → (480,640,3)
                
                # 颜色空间转换（假设原始数据是 RGB）
                frame = cv2.cvtColor(
                    frame_data,
                    cv2.COLOR_RGB2BGR  # RGB → BGR（OpenCV 格式）
                ).astype(np.uint8).copy(order='C')
                
                # 创建 QImage
                height, width, channels = frame.shape
                bytes_per_line = channels * width
                q_image = QImage(
                    frame.data.tobytes(),
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_BGR888  # 匹配 BGR 格式
                )
                self.video_labels[i].setPixmap(QPixmap.fromImage(q_image))
            self.current_frame += 1
        else:
            self.timer.stop()
            self.btn_play_pause.setText("Play")
            self.current_frame = 0
            
    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择HDF5文件", "", "HDF5 Files (*.hdf5)")
        if file_path:
            self.hdf5_data = h5py.File(file_path, 'r')
            self.init_video()
            self.init_plot()
            
    def init_video(self):
        if not self.hdf5_data:
            return
        
        images_group = self.hdf5_data['observation/images']
        self.camera_names = list(images_group.keys())
        self.total_frames = images_group[self.camera_names[0]].shape[0]
        
        # 自适应摄像头数量
        if len(self.camera_names) != len(self.video_labels):
            # 清空原有布局并重新创建
            self.video_layout.deleteLater()
            self.video_labels = [QLabel(f"Camera {i}") for i in range(len(self.camera_names))]
            for label in self.video_labels:
                label.setFixedSize(640, 480)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.video_layout.addWidget(label)
            self.main_layout.insertLayout(0, self.video_layout)
            
    def init_plot(self):
        if not self.hdf5_data:
            return
        
        qpos = self.hdf5_data['observation/qpos'][:]
        self.plot_data = qpos
        
        self.ax.clear()
        self.ax.plot(self.plot_data)
        self.ax.set_title("Joint States")
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Value")
        self.canvas.draw()
        
    def reset(self):
        self.current_frame = 0
        self.timer.stop()
        self.update_frame()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())