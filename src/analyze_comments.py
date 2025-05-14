
import argparse
import json
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

def analyze_comments(input_path, output_path):
    with open(input_path, encoding="utf-8") as f:
        comments = json.load(f)

    classifier = pipeline(
        "text-classification",
        model="fyaronskiy/ModernBERT-large-english-go-emotions",
        tokenizer="fyaronskiy/ModernBERT-large-english-go-emotions",
        top_k=None
    )

    emotion_scores = Counter()
    comment_text = ""

    for comment in comments:
        comment_text += comment.strip() + "\n"
        results = classifier(comment.strip())[0]
        for emo in results:
            label = EN_KO_EMOTION_MAP.get(emo["label"], emo["label"])
            emotion_scores[label] += emo["score"]

    total = sum(emotion_scores.values())
    normed = {k: round(v / total, 6) for k, v in emotion_scores.items()}
    dominant = max(normed, key=normed.get)

    result = {
        "text": comment_text.strip(),
        "emotion_distribution": normed,
        "dominant_emotion": dominant,
        "source": "comments"
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ 댓글 감정 결과 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="댓글 리스트 JSON")
    parser.add_argument("--output", required=True, help="출력 결과 JSON")
    args = parser.parse_args()
    analyze_comments(args.input, args.output)
