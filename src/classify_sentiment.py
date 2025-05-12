#!/usr/bin/env python3
# classify_sentiment.py (ver. 감정별 가중치 구조 대응)

import os
import json
import argparse
from collections import defaultdict, Counter

def classify_emotions(detection_path, sentiment_map_path, output_path):
    # 감지 결과 로드
    with open(detection_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    # 감성 사전 로드
    with open(sentiment_map_path, 'r', encoding='utf-8') as f:
        sentiment_map = json.load(f)

    emotion_score = Counter()
    frame_tags = defaultdict(list)

    for frame, objects in detections.items():
        for obj in objects:
            if obj in sentiment_map:
                emo_dict = sentiment_map[obj]["emotions"]
                for emo, weight in emo_dict.items():
                    emotion_score[emo] += weight
                    frame_tags[frame].append(emo)

    total = sum(emotion_score.values())
    emotion_distribution = {
        emo: round(score / total * 100, 2)
        for emo, score in emotion_score.items()
    }
    dominant_emotion = emotion_score.most_common(1)[0][0] if emotion_score else "중립"

    result = {
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "frame_level_tags": frame_tags
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ 감성 분석 결과 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 감지 결과 기반 감정 분석 (감정별 가중치)")
    parser.add_argument("--detection", type=str, required=True, help="YOLO 감지 결과 JSON")
    parser.add_argument("--sentimap", type=str, required=True, help="감성 사전 JSON (감정별 가중치)")
    parser.add_argument("--output", type=str, default="data/results/final_sentiment.json", help="출력 결과 파일")
    args = parser.parse_args()

    classify_emotions(args.detection, args.sentimap, args.output)
