
import os
import argparse
import json
import cv2
from ultralytics import YOLO
from transformers import pipeline
from collections import Counter

EN_KO_EMOTION_MAP = {
    "admiration": "감탄", "amusement": "재미", "anger": "분노", "annoyance": "짜증",
    "approval": "수긍", "caring": "보살핌", "confusion": "혼란", "curiosity": "호기심",
    "desire": "욕망", "disappointment": "실망", "disapproval": "거부감", "disgust": "혐오",
    "embarrassment": "당황", "excitement": "흥분", "fear": "두려움", "gratitude": "감사",
    "grief": "슬픔", "joy": "기쁨", "love": "사랑", "nervousness": "긴장", "optimism": "낙관",
    "pride": "자부심", "realization": "깨달음", "relief": "안도", "remorse": "후회",
    "sadness": "슬픔", "surprise": "놀람", "neutral": "중립"
}

def analyze_emotion(object_labels):
    if not object_labels:
        return {
            "dominant_emotion": "중립",
            "emotion_distribution": {"중립": 1.0},
            "source": "object_gpt"
        }

    # 새로운 문장형 프롬프트
    prompt = f"A visual scene with the following objects: {', '.join(object_labels)}. What kind of emotion does this scene convey?"
    results = classifier(prompt)[0]

    emotions = {e["label"]: float(e["score"]) for e in results}
    total = sum(emotions.values())
    normed = {EN_KO_EMOTION_MAP.get(k, k): round(v / total, 6) for k, v in emotions.items()}
    dominant = max(normed, key=normed.get)

    return {
        "dominant_emotion": dominant,
        "emotion_distribution": normed,
        "source": "object_gpt"
    }

def detect_objects(input_dir, output_path, model_path='yolov8n.pt', conf=0.25):
    model = YOLO(model_path)
    bbox_results = {}
    frame_objects = {}

    annotated_dir = os.path.join(os.path.dirname(output_path), "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])

    for idx, file in enumerate(files):
        image_path = os.path.join(input_dir, file)
        print(f"\n📸 [{idx+1}/{len(files)}] {file} 감지 중...")

        results = model(image_path, conf=conf)
        result = results[0]

        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        boxes = result.boxes
        class_names = result.names
        bbox_objects = []
        object_labels = []

        for box in boxes:
            cls = int(box.cls.item())
            conf_score = round(box.conf.item(), 2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = class_names[cls]

            object_labels.append(label)

            bbox_objects.append({
                "label": label,
                "confidence": conf_score,
                "bbox": [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(x2 / w, 4),
                    round(y2 / h, 4)
                ]
            })

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{label} ({conf_score})"
            cv2.putText(image, text, (int(x1), max(int(y1) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        bbox_results[file] = bbox_objects
        frame_objects[file] = object_labels
        cv2.imwrite(os.path.join(annotated_dir, file), image)

        print(f"🔍 감지된 객체: {', '.join(object_labels) if object_labels else '(없음)'}")
        result = analyze_emotion(object_labels)
        print(f"🧠 감정 분석 → 주요 감정: {result['dominant_emotion']}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 객체 감지 결과 저장 → {output_path}")
    print(f"🖼️ Annotated frames 저장 → {annotated_dir}")

    video_id = os.path.splitext(os.path.basename(output_path))[0].replace("_detections", "")
    emotion_path = os.path.join("data/results", f"{video_id}_object.json")

    total_emotions = Counter()
    count = 0
    for labels in frame_objects.values():
        result = analyze_emotion(labels)
        for emo, val in result["emotion_distribution"].items():
            total_emotions[emo] += val
        count += 1

    if count > 0:
        final_dist = {k: round(v / count, 4) for k, v in total_emotions.items()}
        dominant = max(final_dist, key=final_dist.get)
    else:
        final_dist = {"중립": 1.0}
        dominant = "중립"

    final_output = {
        "dominant_emotion": dominant,
        "emotion_distribution": final_dist,
        "source": "object_gpt"
    }

    os.makedirs(os.path.dirname(emotion_path), exist_ok=True)
    with open(emotion_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    print(f"🧠 감정 분석 결과 저장 → {emotion_path}")

if __name__ == "__main__":
    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model="fyaronskiy/ModernBERT-large-english-go-emotions",
        tokenizer="fyaronskiy/ModernBERT-large-english-go-emotions",
        top_k=None
    )

    parser = argparse.ArgumentParser(description="YOLO 객체 감지 + GPT 기반 감정 추론")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/objects/detections.json")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    detect_objects(args.input_dir, args.output, args.model, args.conf)
