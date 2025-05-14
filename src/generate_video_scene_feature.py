
import os
import json
import argparse
from ultralytics import YOLO
from moviepy.editor import VideoFileClip
from transformers import BertTokenizer, BertModel
import torch
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)[0].cpu().tolist()

def track_video(video_path, model_path="yolov8n.pt", conf=0.25):
    model = YOLO(model_path)
    results = model.track(source=video_path, conf=conf, save=False, verbose=False)
    print("✅ YOLOv8 객체 추적 완료")
    return results[0]  # Assume single video

def summarize_objects(tracks):
    label_counter = Counter()
    for box in tracks.boxes:
        cls = int(box.cls.item())
        label = tracks.names[cls]
        label_counter[label] += 1
    return label_counter

def generate_scene_sentence(counter):
    top = counter.most_common(5)
    labels = [lbl for lbl, _ in top]
    if not labels:
        return "An empty scene."
    elif "person" in labels and "pizza" in labels:
        return "A person is eating pizza."
    elif "person" in labels and "bottle" in labels:
        return "A person is drinking from a bottle."
    elif "pizza" in labels and "table" in labels:
        return "Pizza is served on a table."
    elif "person" in labels:
        return "A person is visible."
    else:
        return "A scene with " + ", ".join(labels) + "."

def extract_scene_from_video(video_path, output_path, model_path, conf):
    print("🚀 영상 기반 장면 임베딩 시작...")
    tracks = track_video(video_path, model_path, conf)
    summary = summarize_objects(tracks)
    sentence = generate_scene_sentence(summary)
    print(f"📝 생성 문장: {sentence}")
    embedding = get_bert_embedding(sentence)

    result = {
        "text": sentence,
        "embedding": embedding
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ 비디오 장면 임베딩 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 트래킹 기반 영상 전체 장면 기술 + 임베딩")
    parser.add_argument("--video", required=True, help="영상 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSON 파일 경로")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    extract_scene_from_video(args.video, args.output, args.model, args.conf)
