import sys
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QFileDialog
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
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
        self.preprocessed_images = {}  # 预处理后的帧数据缓存
        
        # 性能优化：异步加载线程
        self.frame_loader = FrameLoaderThread()
        self.frame_loader.frame_ready.connect(self.update_video_labels)
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # UI布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # 视频显示区域
        self.video_layout = QHBoxLayout()
        self.video_labels = []
        self.q_images = []  # 预分配QImage对象
        self.create_video_labels(3)  # 初始3个摄像头
        
        self.main_layout.addLayout(self.video_layout)
        
        # 控制区域
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
        
    def create_video_labels(self, count):
        """动态创建视频标签"""
        self.video_layout.deleteLater()
        self.video_labels = []
        self.q_images = []
        self.video_layout = QHBoxLayout()
        
        for i in range(count):
            label = QLabel(f"Camera {i}")
            label.setFixedSize(640, 480)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_layout.addWidget(label)
            self.video_labels.append(label)
            
            # 预创建QImage对象
            q_img = QImage(640, 480, QImage.Format.Format_BGR888)
            self.q_images.append(q_img)
        
        self.main_layout.insertLayout(0, self.video_layout)
        
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn_play_pause.setText("Play")
        else:
            self.timer.start(33)  # ~30 FPS
            self.btn_play_pause.setText("Pause")
            
    def update_frame(self):
        if self.current_frame < self.total_frames:
            # 异步加载下一帧
            self.frame_loader.set_frame(self.current_frame + 1, self.camera_names)
            self.frame_loader.start()
            
            # 更新当前帧显示
            self.show_current_frame()
            self.current_frame += 1
        else:
            self.timer.stop()
            self.btn_play_pause.setText("Play")
            
    def show_current_frame(self):
        """直接显示当前帧（无需转置）"""
        for i, cam_name in enumerate(self.camera_names):
            frame = self.preprocessed_images[cam_name][self.current_frame]
            self.q_images[i].loadFromData(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.strides[0],
                QImage.Format.Format_BGR888
            )
            self.video_labels[i].setPixmap(QPixmap.fromImage(self.q_images[i]))
            
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
        
        # 预处理所有帧数据（关键优化）
        self.preprocessed_images = {}
        for cam_name in self.camera_names:
            # 预转置为HWC格式并确保连续内存
            data = self.hdf5_data[f'observation/images/{cam_name}'][:]
            processed = np.ascontiguousarray(data.transpose(0, 2, 3, 1), dtype=np.uint8)
            self.preprocessed_images[cam_name] = processed
            
        # 动态调整视频标签数量
        self.create_video_labels(len(self.camera_names))
        
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
        self.show_current_frame()

class FrameLoaderThread(QThread):
    frame_ready = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.target_frame = 0
        self.camera_names = []
        
    def set_frame(self, frame_idx, cams):
        self.target_frame = frame_idx
        self.camera_names = cams

    def run(self):
        # 异步预加载下一帧数据（假设已在预处理）
        # 这里仅作为占位符，实际预处理已在load阶段完成
        self.frame_ready.emit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = HDF5Viewer()
    viewer.show()
    sys.exit(app.exec())