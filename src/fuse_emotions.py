import os
import json
import argparse
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_distribution(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()} if total > 0 else {}


# 감정 라벨 고정 (GoEmotions 한글 기준)
EMOTION_LABELS = [
    "감탄", "재미", "분노", "짜증", "수긍", "보살핌", "혼란", "호기심", "욕망", "실망",
    "거부감", "혐오", "당황", "흥분", "두려움", "감사", "슬픔", "기쁨", "사랑", "긴장",
    "낙관", "자부심", "깨달음", "안도", "후회", "중립", "놀람", "비탄"
]


# 감정 분류 모델 로딩
def load_emotion_model():
    model_name = "beomi/KcELECTRA-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(EMOTION_LABELS))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model


# 감정 분포 예측 (문장 단위 처리 후 평균)
def predict_emotion(text, tokenizer, model):
    sentences = [s.strip() for s in text.replace("\n", " ").split('.') if len(s.strip()) > 5]
    all_distributions = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0].tolist()
            all_distributions.append(probs)

    # 평균 분포 계산
    avg_dist = [0.0] * len(EMOTION_LABELS)
    for dist in all_distributions:
        for i, val in enumerate(dist):
            avg_dist[i] += val
    avg_dist = [val / len(all_distributions) for val in avg_dist]

    return {
        "dominant_emotion": EMOTION_LABELS[avg_dist.index(max(avg_dist))],
        "emotion_distribution": {label: float(prob) for label, prob in zip(EMOTION_LABELS, avg_dist)}
    }


# 감정 통합 함수
def fuse_emotions(vision_path, audio_path, ocr_path, output_path):
    sources = {}
    tokenizer, model = load_emotion_model()

    for name, path in [("vision", vision_path), ("audio", audio_path), ("ocr", ocr_path)]:
        data = load_summary(path)
        if not data:
            continue

        if "emotion_distribution" in data:
            sources[name] = data
        else:
            text = data.get("summary") or data.get("text") or data.get("transcript")
            if text:
                prediction = predict_emotion(text, tokenizer, model)
                prediction["text"] = text
                sources[name] = prediction

        if name in sources:
            per_source_output = os.path.join(
                os.path.dirname(output_path), f"{os.path.basename(output_path).split('_')[0]}_{name}_emotion.json"
            )
            with open(per_source_output, "w", encoding="utf-8") as f:
                json.dump(sources[name], f, indent=2, ensure_ascii=False)

    if not sources:
        print("❌ 감정 추출할 정보가 없습니다.")
        return

    # 전체 감정 분포 평균 계산
    fused_dist = defaultdict(float)
    for name, module in sources.items():
        weight = 0.1 if name == "vision" else 1.0
        for emo, val in module["emotion_distribution"].items():
            if emo != "중립":  # 중립 제외
                fused_dist[emo] += val * weight

    fused_dist = {k: v / 2.1 for k, v in fused_dist.items()}
    dominant = max(fused_dist, key=fused_dist.get)

    final_output = {
        "dominant_emotion": dominant,
        "emotion_distribution": normalize_distribution(fused_dist),
        "sources": sources
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"✅ Fused emotion saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="모듈별 감정 분석 결과 통합")
    parser.add_argument("--vision", type=str, required=False, help="시각 정보 요약 JSON")
    parser.add_argument("--audio", type=str, required=True, help="오디오 분석 JSON")
    parser.add_argument("--ocr", type=str, required=True, help="OCR 분석 JSON")
    parser.add_argument("--output", type=str, required=True, help="최종 감정 결과 JSON")
    args = parser.parse_args()

    fuse_emotions(args.vision, args.audio, args.ocr, args.output)