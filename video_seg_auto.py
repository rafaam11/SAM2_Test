import sys
import av
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import SAM
import cv2
import torch

# CUDA 최적화를 위해 설정
torch.backends.cudnn.benchmark = True

class VideoSegmentation(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("최적화된 비디오 세그멘테이션")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # 더 작은 모델 사용 (가능한 경우)
        self.model = SAM('sam2_b.pt').to('cuda')
        # self.model.eval()  # 모델을 평가 모드로 설정

        self.video_path = "data/sav_1.mp4"  # 여기에 실제 비디오 경로를 입력하세요
        self.container = av.open(self.video_path)
        self.stream = self.container.streams.video[0]

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # 약 30 FPS

        # 최적화를 위한 설정
        self.frame_skip = 2  # 매 2번째 프레임만 처리
        self.current_frame = 0
        self.resize_width, self.resize_height = 640, 360  # 프레임 크기 조정

    def update_frame(self):
        try:
            self.current_frame += 1
            if self.current_frame % self.frame_skip != 0:
                return

            frame = next(self.container.decode(video=0))
            image = frame.to_ndarray(format='rgb24')

            # 프레임 크기 축소
            resized_image = cv2.resize(image, (self.resize_width, self.resize_height))

            # 세그멘테이션 수행
            with torch.no_grad():
                results = self.model.predict(resized_image)

            if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                masks = results[0].masks.data.cpu().numpy()

                # 상위 3개의 마스크 사용
                top_masks = masks[:3] if len(masks) >= 3 else masks
                colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # 각각 빨강, 초록, 파랑

                for i, mask in enumerate(top_masks):
                    colored_mask = np.zeros_like(resized_image)
                    colored_mask[mask.astype(bool)] = colors[i]
                    resized_image = cv2.addWeighted(resized_image, 1, colored_mask, 0.5, 0)

            # 원본 크기로 복구하여 출력
            output_image = cv2.resize(resized_image, (image.shape[1], image.shape[0]))

            # 결과를 QLabel에 표시
            height, width, channel = output_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(output_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)

        except StopIteration:
            self.timer.stop()
            print("비디오의 끝에 도달했습니다.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSegmentation()
    window.show()
    sys.exit(app.exec_())
