#!/usr/bin/env python3
# detect_objects.py – YOLO 감지 결과 + 감성 사전 기반 감정 시각화 (한글 지원)

import os
import argparse
import json
from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def load_sentiment_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        k: list(v["emotions"].keys())[0] if isinstance(v["emotions"], dict) else v["emotions"][0]
        for k, v in data.items()
    }

def detect_objects(input_dir, output_path, sentiment_path=None, model_path='yolo11x.pt', conf=0.25):
    model = YOLO(model_path)
    results_dict = {}

    sentiment_map = load_sentiment_map(sentiment_path) if sentiment_path else {}

    # 시각화된 결과 저장 폴더
    annotated_dir = os.path.join(os.path.dirname(output_path), "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)

    # 폰트 경로 설정
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # Linux fallback

    font = ImageFont.truetype(font_path, 16)

    for file in sorted(os.listdir(input_dir)):
        if not file.lower().endswith(('.jpg', '.png')):
            continue

        image_path = os.path.join(input_dir, file)
        print(f"🔍 Detecting objects in: {image_path}")
        results = model(image_path, conf=conf)
        result = results[0]

        image_cv = cv2.imread(image_path)
        h, w = image_cv.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        boxes = result.boxes
        labels = result.names
        objects = []

        for box in boxes:
            cls = int(box.cls.item())
            conf_score = round(box.conf.item(), 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = labels[cls]
            sentiment = sentiment_map.get(label, "중립")

            # 결과 저장용
            objects.append({
                "label": label,
                "confidence": conf_score,
                "bbox": [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(x2 / w, 4),
                    round(y2 / h, 4)
                ],
                "sentiment": sentiment
            })

            # 박스 및 텍스트 출력 (한글 포함)
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, max(0, y1 - 20)), f"{label} ({conf_score}) - {sentiment}", font=font, fill=(0, 255, 0))

        # 저장
        annotated = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(annotated_dir, file), annotated)
        results_dict[file] = objects

    # JSON 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Detection results saved to {output_path}")
    print(f"🖼️ Annotated frames saved to {annotated_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 객체 감지 + 감정 시각화 (한글 포함)")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 프레임 디렉토리")
    parser.add_argument("--output", type=str, default="data/objects/detections.json", help="출력 JSON 경로")
    parser.add_argument("--sentimap", type=str, default=None, help="감성 사전 JSON 경로")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="YOLO 모델 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="감지 confidence threshold")
    args = parser.parse_args()

    detect_objects(args.input_dir, args.output, args.sentimap, args.model, args.conf)
