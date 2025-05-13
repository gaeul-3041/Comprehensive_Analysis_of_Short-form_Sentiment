import os
import argparse
import json
from ultralytics import YOLO
import cv2
import itertools
from collections import defaultdict

def filter_to_goemotions(emotions_dict):
    return emotions_dict  # 전체 감정 허용

def load_sentiment_maps(base_path, vg_path):
    with open(base_path, 'r', encoding='utf-8') as f1:
        base_raw = json.load(f1)
    with open(vg_path, 'r', encoding='utf-8') as f2:
        vg = json.load(f2)
    base = {label.lower().strip(): val for label, val in base_raw.items()}
    return base, vg

def extract_emotions_from_entry(entry):
    if not isinstance(entry, dict):
        return {}
    return {k: float(v) for k, v in entry.items() if isinstance(v, (int, float))}

def detect_objects(input_dir, output_path, sentiment_base_path, sentiment_vg_path, model_path='yolov8n.pt', conf=0.25):
    model = YOLO(model_path)
    bbox_results = {}
    frame_emotion_list = []

    sentiment_map, vg_map = load_sentiment_maps(sentiment_base_path, sentiment_vg_path)
    annotated_dir = os.path.join(os.path.dirname(output_path), "annotated_frames")
    os.makedirs(annotated_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if not file.lower().endswith(('.jpg', '.png')):
                continue
            image_path = os.path.join(root, file)
            print(f"🔍 Detecting objects in: {image_path}")
            results = model(image_path, conf=conf)
            result = results[0]

            image = cv2.imread(image_path)
            h, w = image.shape[:2]

            boxes = result.boxes
            class_names = result.names
            bbox_objects = []
            frame_emotions = defaultdict(float)
            detected_labels = []

            for box in boxes:
                cls = int(box.cls.item())
                conf_score = round(box.conf.item(), 2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label_raw = class_names[cls]
                label = label_raw.lower().strip()

                sentiment_entry = sentiment_map.get(label, {})
                sentiments = extract_emotions_from_entry(sentiment_entry)

                if label not in sentiment_map:
                    print(f"⚠️ 감정 사전에 없는 라벨: {label_raw}")
                elif not sentiments:
                    print(f"⚠️ 감정 정보 없음: {label_raw} → 사전은 있음, 감정 비어 있음")

                detected_labels.append(label_raw)

                # 바운딩 박스 정보 저장
                bbox_objects.append({
                    "label": label_raw,
                    "confidence": conf_score,
                    "bbox": [
                        round(x1 / w, 4),
                        round(y1 / h, 4),
                        round(x2 / w, 4),
                        round(y2 / h, 4)
                    ]
                })

                for emo, score in sentiments.items():
                    frame_emotions[emo] += score

                # 시각화
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{label_raw} ({conf_score})"
                cv2.putText(image, text, (int(x1), max(int(y1) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 관계 감정 추가
            for r in range(2, 4):
                for combo in itertools.combinations(sorted(set(detected_labels)), r):
                    key = "+".join(combo)
                    if key in vg_map:
                        rel_sentiments = filter_to_goemotions(vg_map[key].get("emotions", {}))
                        for emo, score in rel_sentiments.items():
                            frame_emotions[emo] += score

            # 프레임 감정 정규화 후 저장
            total = sum(frame_emotions.values())
            if total > 0:
                normed = {k: v / total for k, v in frame_emotions.items()}
                frame_emotion_list.append(normed)

            bbox_results[file] = bbox_objects
            cv2.imwrite(os.path.join(annotated_dir, file), image)

    # 바운딩 박스 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_results, f, indent=2, ensure_ascii=False)

    # 프레임 평균 감정 결과 저장
    aggregated = defaultdict(float)
    for emo_dist in frame_emotion_list:
        for emo, val in emo_dist.items():
            aggregated[emo] += val

    frame_count = len(frame_emotion_list)
    final_dist = {k: v / frame_count for k, v in aggregated.items()} if frame_count else {}

    dominant = max(final_dist, key=final_dist.get) if final_dist else "중립"
    object_emotion_output = {
        "dominant_emotion": dominant,
        "emotion_distribution": final_dist,
        "source": "object"
    }

    emotion_path = os.path.join("data/results", os.path.basename(output_path).replace("_detections", "_object"))
    os.makedirs(os.path.dirname(emotion_path), exist_ok=True)
    with open(emotion_path, 'w', encoding='utf-8') as f:
        json.dump(object_emotion_output, f, indent=2, ensure_ascii=False)

    print(f"✅ 바운딩 박스 저장 → {output_path}")
    print(f"✅ 감정 결과 저장 → {emotion_path}")
    print(f"🖼️ Annotated frames 저장 → {annotated_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 객체 감지 + 감정 사전 기반 감정 태깅")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 프레임 디렉토리")
    parser.add_argument("--output", type=str, default="data/objects/detections.json", help="출력 JSON 경로")
    parser.add_argument("--sentimap", type=str, required=True, help="기본 감성 사전 경로")
    parser.add_argument("--vgmap", type=str, required=True, help="VG 감정 관계 사전 경로")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 모델 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="감지 confidence threshold")
    args = parser.parse_args()

    detect_objects(args.input_dir, args.output, args.sentimap, args.vgmap, args.model, args.conf)
