#!/usr/bin/env python3
# visualize_results.py (워드클라우드 생성 및 한글 깨짐 방지 포함)

import os
import json
import argparse
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordcloud(result_path, output_image="data/results/wordcloud.png"):
    # 한글 깨짐 방지 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용
    plt.rcParams['axes.unicode_minus'] = False

    # JSON 감정 분포 로드
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    emotion_distribution = data.get("emotion_distribution", {})
    if not emotion_distribution:
        print("⚠️ 감정 데이터가 비어있습니다.")
        return

    # 워드클라우드 생성
    wordcloud = WordCloud(
        font_path="C:/Windows/Fonts/malgun.ttf",
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(emotion_distribution)

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("감정 워드클라우드", fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_image), exist_ok=True)
    plt.savefig(output_image)
    print(f"✅ 워드클라우드 저장 완료 → {output_image}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="감성 분석 결과로 워드클라우드 생성")
    parser.add_argument("--input", type=str, default="data/results/final_sentiment.json", help="감성 분석 결과 JSON")
    parser.add_argument("--output", type=str, default="data/results/wordcloud.png", help="출력 이미지 경로")
    args = parser.parse_args()

    generate_wordcloud(args.input, args.output)
