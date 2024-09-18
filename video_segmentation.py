import sys
import av
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import SAM
import cv2

class VideoSegmentation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("빠른 비디오 세그멘테이션")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        self.model = SAM('sam2_b.pt').to('cuda')
        self.video_path = "data\KakaoTalk_20240821_112445447.mp4"  # 여기에 실제 비디오 경로를 입력하세요
        self.container = av.open(self.video_path)
        self.stream = self.container.streams.video[0]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 약 30 FPS

        self.click_point = None

    def update_frame(self):
        try:
            frame = next(self.container.decode(video=0))
            image = frame.to_ndarray(format='rgb24')

            if self.click_point is not None:
                input_point = np.array([[self.click_point[0], self.click_point[1]]])
                input_label = np.array([1])

                results = self.model.predict(image, points=input_point, labels=input_label)

                if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    for mask in masks:
                        colored_mask = np.zeros_like(image)
                        colored_mask[mask.astype(bool)] = [0, 255, 0]
                        image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)

        except StopIteration:
            self.timer.stop()
            print("비디오의 끝에 도달했습니다.")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.click_point = (event.x(), event.y())
            print(f"클릭한 위치: ({event.x()}, {event.y()})")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.click_point = None
        elif event.key() == Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSegmentation()
    window.show()
    sys.exit(app.exec_())