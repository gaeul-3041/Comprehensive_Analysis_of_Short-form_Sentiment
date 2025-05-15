import os
import argparse
import json
import re
from ultralytics import YOLO
import cv2
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

def run_llm_emotion_estimator(summary):
    model_path = "model/kanana-nano"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

    with open("prompts/emotion_summary_prompt.txt", encoding="utf-8") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{{visual_summary}}", summary)

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_str = re.search(r"\{.*\}", decoded, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        print(f"❌ 감정 JSON 파싱 실패: {e}")
        return {
            "dominant_emotion": "중립",
            "emotion_distribution": {"중립": 1.0},
            "source": "visual_summary",
            "transcript": summary
        }

def fuse_emotion_distribution(chunks):
    fused = defaultdict(float)
    count = 0
    for chunk in chunks.values():
        dist = chunk.get("emotion_distribution", {})
        if dist:
            for emo, val in dist.items():
                fused[emo] += val
            count += 1
    if count > 0:
        fused = {k: v / count for k, v in fused.items()}
        dominant = max(fused, key=fused.get)
    else:
        fused = {"중립": 1.0}
        dominant = "중립"
    return {
        "dominant_emotion": dominant,
        "emotion_distribution": fused,
        "source": "visual_summary_chunks"
    }

def detect_objects(input_dir, output_path, model_path='yolov8n.pt', conf=0.25):
    model = YOLO(model_path)
    bbox_results = {}
    all_summaries = []
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

            for box in boxes:
                cls = int(box.cls.item())
                conf_score = round(box.conf.item(), 2)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label_raw = class_names[cls]
                label = label_raw.lower().strip()

                bbox_objects.append({
                    "label": label_raw,
                    "confidence": conf_score,
                    "bbox": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
                })

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2, ), int(y2)), (0, 255, 0), 2)
                text = f"{label_raw} ({conf_score})"
                cv2.putText(image, text, (int(x1), max(int(y1) - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            summary = generate_visual_summary(bbox_objects)
            if not bbox_objects:
                summary = "No detectable objects were found in this scene."
            all_summaries.append(f"- {summary}")

            bbox_results[file] = {
                "objects": bbox_objects,
                "summary": summary
            }
            cv2.imwrite(os.path.join(annotated_dir, file), image)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(bbox_results, f, indent=2, ensure_ascii=False)

    chunk_size = 10
    scene_emotion_chunks = {}
    emotion_distributions = []

    for i in range(0, len(all_summaries), chunk_size):
        chunk = all_summaries[i:i + chunk_size]
        scene_description = "".join(chunk)
        emotion_json = run_llm_emotion_estimator(scene_description)
        emotion_distributions.append(emotion_json.get("emotion_distribution", {}))
        scene_emotion_chunks[f"chunk_{i//chunk_size+1:02d}"] = {
            "dominant_emotion": emotion_json.get("dominant_emotion", "중립"),
            "emotion_distribution": emotion_json.get("emotion_distribution", {}),
            "transcript": scene_description,
            "source": "visual_summary"
        }

    fused = defaultdict(float)
    count = len(emotion_distributions)
    for dist in emotion_distributions:
        for emo, val in dist.items():
            fused[emo] += val
    if count > 0:
        fused = {k: v / count for k, v in fused.items()}
        dominant = max(fused, key=fused.get)
    else:
        fused = {"중립": 1.0}
        dominant = "중립"

    scene_emotion_path = os.path.join("data/results", os.path.basename(output_path).replace("_detections", "_scene_emotion"))
    with open(scene_emotion_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dominant_emotion": dominant,
            "emotion_distribution": fused,
            "transcript": "\n\n".join(all_summaries),
            "source": "visual_summary",
            "chunks": scene_emotion_chunks
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ 바운딩 박스 저장 → {output_path}")
    print(f"✅ 요약 기반 감정 결과 저장 → {scene_emotion_path}")
    print(f"🖼️ Annotated frames 저장 → {annotated_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 객체 감지 + 시각 요약 + 감정 분포 생성")
    parser.add_argument("--input_dir", type=str, required=True, help="입력 프레임 디렉토리")
    parser.add_argument("--output", type=str, default="data/objects/detections.json", help="출력 JSON 경로")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 모델 경로")
    parser.add_argument("--conf", type=float, default=0.25, help="감지 confidence threshold")
    args = parser.parse_args()

    detect_objects(args.input_dir, args.output, args.model, args.conf)
