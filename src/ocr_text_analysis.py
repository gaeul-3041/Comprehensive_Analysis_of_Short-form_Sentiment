import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import easyocr
from transformers import pipeline
from difflib import SequenceMatcher

# 한글 감정 매핑
EN_KO_EMOTION_MAP = {
    "admiration": "감탄", "amusement": "재미", "anger": "분노", "annoyance": "짜증",
    "approval": "수긍", "caring": "보살핌", "confusion": "혼란", "curiosity": "호기심",
    "desire": "욕망", "disappointment": "실망", "disapproval": "거부감", "disgust": "혐오",
    "embarrassment": "당황", "excitement": "흥분", "fear": "두려움", "gratitude": "감사",
    "grief": "슬픔", "joy": "기쁨", "love": "사랑", "nervousness": "긴장", "optimism": "낙관",
    "pride": "자부심", "realization": "깨달음", "relief": "안도", "remorse": "후회",
    "sadness": "슬픔", "surprise": "놀람", "neutral": "중립"
}

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a, b).ratio() > threshold

def analyze_ocr(input_dir, output_path):
    try:
        print("🚀 영상 텍스트 감정 분석 시작...")
        reader = easyocr.Reader(['en', 'ko'])

        classifier = pipeline(
            "text-classification",
            model="fyaronskiy/ModernBERT-large-english-go-emotions",
            tokenizer="fyaronskiy/ModernBERT-large-english-go-emotions",
            top_k=None
        )

        image_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not image_files:
            raise FileNotFoundError("❌ 분석할 이미지가 존재하지 않습니다.")

        combined_lines = []
        print(f"📂 총 {len(image_files)}장의 프레임에서 OCR 수행")

        for idx, file in enumerate(image_files):
            image_path = os.path.join(input_dir, file)
            result = reader.readtext(image_path, detail=0)
            text = " ".join(result).strip()
            print(f"🖼 [{idx+1}/{len(image_files)}] {file} → \"{text[:40]}{'...' if len(text) > 40 else ''}\"")
            if text and not any(is_similar(text, existing) for existing in combined_lines):
                combined_lines.append(text)

        if not combined_lines:
            raise ValueError("OCR 텍스트가 비어 있습니다.")

        dedup_text = "\n".join(combined_lines)
        print("🧠 감정 분석 시작...")
        results = classifier(dedup_text, top_k=None)

        emotions = {
            EN_KO_EMOTION_MAP.get(e["label"], e["label"]): float(e["score"])
            for e in results
        }
        dominant = max(emotions, key=emotions.get)

        output = {
            "dominant_emotion": dominant,
            "emotion_distribution": emotions,
            "source": "ocr",
            "text": dedup_text
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ OCR 감정 분석 완료 → {output_path}")

    except Exception as e:
        print(f"❌ OCR 감정 분석 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    analyze_ocr(args.input_dir, args.output)
