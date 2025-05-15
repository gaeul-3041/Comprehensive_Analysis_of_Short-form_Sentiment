import os
import argparse
import json
from ultralytics import YOLO
import cv2
import itertools
from collections import defaultdict

def load_sentiment_maps(base_path, vg_path):
    with open(base_path, 'r', encoding='utf-8') as f1:
        base_raw = json.load(f1)
    with open(vg_path, 'r', encoding='utf-8') as f2:
        vg = json.load(f2)
    base = {label.lower().strip(): val for label, val in base_raw.items()}
    return base, vg

def extract_emotions_from_entry(entry):
    return {k: float(v) for k, v in entry.items() if isinstance(v, (int, float))}

def generate_visual_summary(objects):
    label_count = defaultdict(int)
    people_positions = []
    summary_lines = []

    for obj in objects:
        label = obj['label']
        label_count[label] += 1

        if label == 'person':
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            people_positions.append(center_x)

    num_people = label_count.get('person', 0)
    if num_people == 0:
        summary_lines.append("No people are visible in the scene.")
    elif num_people == 1:
        pos = "center" if 0.3 < people_positions[0] < 0.7 else "edge"
        summary_lines.append(f"A single person appears near the {pos}.")
    elif num_people > 1:
        spread = max(people_positions) - min(people_positions)
        dist = "close together" if spread < 0.3 else "scattered apart"
        summary_lines.append(f"{num_people} people are standing {dist} across the scene.")

    for label, count in label_count.items():
        if label == "person":
            continue
        if count == 1:
            summary_lines.append(f"A {label} is visible.")
        else:
            summary_lines.append(f"{count} {label}s are present.")

    return " ".join(summary_lines)

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

                detected_labels.append(label_raw)

                bbox_objects.append({
                    "label": label_raw,
                    "confidence": conf_score,
                    "bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
                })

                for emo, score in sentiments.items():
                    frame_emotions[emo] += score

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{label_raw} ({conf_score})"
                cv2.putText(image, text, (int(x1), max(int(y1) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for r in range(2, 4):
                for combo in itertools.combinations(sorted(set(detected_labels)), r):
                    key = "+".join(combo)
                    if key in vg_map:
                        rel_sentiments = extract_emotions_from_entry(vg_map[key].get("emotions", {}))
                        for emo, score in rel_sentiments.items():
                            frame_emotions[emo] += score

            total = sum(frame_emotions.values())
            if total > 0:
                normed = {k: v / total for k, v in frame_emotions.items()}
                frame_emotion_list.append(normed)

            bbox_results[file] = {
                "objects": bbox_objects,
                "summary": generate_visual_summary(bbox_objects)
            }
            cv2.imwrite(os.path.join(annotated_dir, file), image)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_results, f, indent=2, ensure_ascii=False)

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
    parser = argparse.ArgumentParser(description="YOLO 객체 감지 + 감정 태깅 + 시각 요약")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 프레임 디렉토리")
    parser.add_argument("--output", type=str, default="data/objects/detections.json", help="출력 JSON 경로")
    parser.add_argument("--sentimap", type=str, required=True, help="기본 감성 사전 경로")
    parser.add_argument("--vgmap", type=str, required=True, help="VG 감정 관계 사전 경로")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 모델 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="감지 confidence threshold")
    args = parser.parse_args()

    detect_objects(args.input_dir, args.output, args.sentimap, args.vgmap, args.model, args.conf)