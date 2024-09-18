from ultralytics import SAM
import cv2
import numpy as np
import torch

# 전역 변수
image = None
model = None

def on_mouse_click(event, x, y, flags, param):
    global image, model
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭한 위치를 점 프롬프트로 사용
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1은 전경을 의미

        # 세그멘테이션 수행
        results = model.predict(image, points=input_point, labels=input_label)
        
        print(f"클릭한 위치: ({x}, {y})")
        print(f"예측 결과: {results}")

        # 결과 시각화
        vis_image = image.copy()
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for mask in masks:
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask.astype(bool)] = [0, 255, 0]  # 초록색으로 마스크 표시
                vis_image = cv2.addWeighted(vis_image, 1, colored_mask, 0.5, 0)
        else:
            print("마스크를 찾을 수 없습니다.")

        # 클릭한 점 표시
        cv2.circle(vis_image, (x, y), 5, (0, 0, 255), -1)
        
        # 결과 표시
        cv2.imshow("Segmentation Result", vis_image)

# 메인 함수
def main():
    global image, model

    # 모델 로드
    model = SAM('sam2_b.pt').to('cuda')

    # 이미지 로드
    image_path = "data/20230223_170757.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 로드할 수 없습니다: {image_path}")
        return

    # 이미지 크기 조정
    max_height = 800
    height, width = image.shape[:2]
    if height > max_height:
        scale = max_height / height
        new_width = int(width * scale)
        image = cv2.resize(image, (new_width, max_height))

    # 윈도우 생성 및 마우스 콜백 설정
    cv2.namedWindow("Segmentation Result")
    cv2.setMouseCallback("Segmentation Result", on_mouse_click)

    # 초기 이미지 표시
    cv2.imshow("Segmentation Result", image)

    print("이미지를 클릭하여 세그멘테이션을 시작하세요. ESC 키를 눌러 종료합니다.")

    # 키 입력 대기
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 키
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()