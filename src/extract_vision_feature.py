import os
import json
import argparse
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModel
import torch


def generate_description(objects):
    objs = sorted(set(objects))
    if not objs:
        return "이 장면에는 어떤 물체도 보이지 않습니다."
    return f"이 장면에는 {', '.join(objs)}가 있습니다."


# 모델 로드 (kanana-nano-2.1b-embedding)
tokenizer = AutoTokenizer.from_pretrained("kakaocorp/kanana-nano-2.1b-embedding")
model = AutoModel.from_pretrained("kakaocorp/kanana-nano-2.1b-embedding").to("cuda" if torch.cuda.is_available() else "cpu")


def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
    return embeddings[0].cpu().tolist()


def extract_vision_embeddings(input_dir, output_path, model_path="yolov8n.pt", conf=0.25):
    yolo_model = YOLO(model_path)
    frame_results = {}
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))])

    for idx, file in enumerate(files):
        image_path = os.path.join(input_dir, file)
        print(f"📸 [{idx+1}/{len(files)}] {file} 임베딩 추출 중...")

        results = yolo_model(image_path, conf=conf)
        result = results[0]
        class_names = result.names
        boxes = result.boxes
        labels = [class_names[int(b.cls.item())] for b in boxes]

        description = generate_description(labels)
        embedding = get_text_embedding(description)

        frame_results[file] = {
            "description": description,
            "embedding": embedding
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frame_results, f, indent=2, ensure_ascii=False)
    print(f"✅ 임베딩 저장 완료 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 기반 비전 특징 임베딩 추출")
    parser.add_argument("--input_dir", required=True, help="프레임 디렉토리")
    parser.add_argument("--output", required=True, help="결과 저장 경로")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    extract_vision_embeddings(args.input_dir, args.output, args.model, args.conf)