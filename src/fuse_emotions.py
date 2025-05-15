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


# 감정 분류 모델 로딩 (로컬이거나 공개 모델 경로 가능)
def load_emotion_model():
    model_name = "beomi/KcELECTRA-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(EMOTION_LABELS))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer, model


# 감정 분포 예측
def predict_emotion(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].tolist()
    return {
        "dominant_emotion": EMOTION_LABELS[probs.index(max(probs))],
        "emotion_distribution": {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
    }


# 감정 통합 함수
def fuse_emotions(vision_path, audio_path, ocr_path, output_path):
    sources = {}
    tokenizer, model = load_emotion_model()

    for name, path in [("vision", vision_path), ("audio", audio_path), ("ocr", ocr_path)]:
        data = load_summary(path)
        if not data:
            continue

        # 이미 확률 분포가 있는 경우 사용
        if "emotion_distribution" in data:
            sources[name] = data
        else:
            # summary/text/transcript 중 가능한 입력 추출
            text = data.get("summary") or data.get("text") or data.get("transcript")
            if text:
                prediction = predict_emotion(text, tokenizer, model)
                prediction["text"] = text
                sources[name] = prediction

        # 분포가 있다면 별도로 저장
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
    for module in sources.values():
        for emo, val in module["emotion_distribution"].items():
            fused_dist[emo] += val

    num_sources = len(sources)
    fused_dist = {k: v / num_sources for k, v in fused_dist.items()}
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
