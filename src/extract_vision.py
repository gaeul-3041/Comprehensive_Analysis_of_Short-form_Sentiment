from ultralytics import YOLO
import os
import json
import argparse
from collections import Counter

# 영어 → 한국어 감정 매핑
EMOTION_KR_MAP = {
    "angry": "분노",
    "disgust": "혐오",
    "fear": "두려움",
    "happy": "기쁨",
    "neutral": "중립",
    "sad": "슬픔",
    "surprise": "놀람"
}

def run_emotion_distribution_inference(image_dir, output_path, model_path="best_v8n.pt"):
    print(f"🔍 감정 탐지 확률 기반 분석 시작: {image_dir}")

    model = YOLO(model_path)
    emotion_labels = list(model.names.values())
    emotion_predictions = []

    frame_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    if not frame_files:
        print("❌ 분석할 이미지가 없습니다.")
        return

    for file in sorted(frame_files):
        image_path = os.path.join(image_dir, file)
        results = model.predict(source=image_path, imgsz=640, conf=0.1, save=False, verbose=False)

        boxes = results[0].boxes
        if boxes is None or boxes.cls is None or len(boxes.cls) == 0:
            print(f"⚠️ 감정 탐지 실패: {file}")
            continue

        emotions_in_frame = []
        for cls_id in boxes.cls:
            label = emotion_labels[int(cls_id.item())]
            emotion_predictions.append(label)
            emotions_in_frame.append(EMOTION_KR_MAP.get(label, label))

        print(f"✅ {file} 에서 감정 탐지 성공: {', '.join(emotions_in_frame)}")

    if not emotion_predictions:
        print("❌ 감정 예측 결과 없음")
        return

    # 감정 빈도 계산 및 확률 (합 = 1.0)
    count = Counter(emotion_predictions)
    total = sum(count.values())
    distribution = {
        EMOTION_KR_MAP[label]: round(freq / total, 3)
        for label, freq in count.items()
        if label in EMOTION_KR_MAP
    }

    # dominant 감정도 한국어로 변환
    dominant_en = max(count, key=count.get)
    dominant_kr = EMOTION_KR_MAP.get(dominant_en, dominant_en)

    result = {
        "dominant_emotion": dominant_kr,
        "emotion_distribution": distribution,
        "source": "vision-detection"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ 감정 분포 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 감정 탐지 확률 기반 분석")
    parser.add_argument("--image", type=str, required=True, help="프레임 이미지 디렉토리 경로")
    parser.add_argument("--output", type=str, required=True, help="결과 저장 JSON 경로")
    parser.add_argument("--model", type=str, default="best_v8n.pt", help="YOLOv8 모델 경로")
    args = parser.parse_args()

    run_emotion_distribution_inference(args.image, args.output, args.model)
