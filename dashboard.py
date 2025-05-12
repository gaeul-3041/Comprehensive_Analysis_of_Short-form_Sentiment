#!/usr/bin/env python3
# dashboard.py – 감성 분석 결과 시각화용 대시보드 (뱃지 스타일 포함)

import os
import json
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

# 한글 폰트 설정
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_NAME = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['font.family'] = FONT_NAME
plt.rcParams['axes.unicode_minus'] = False

def load_sentiment_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_wordcloud(emotion_distribution):
    wc = WordCloud(
        font_path=FONT_PATH,
        width=800,
        height=400,
        background_color="white"
    ).generate_from_frequencies(emotion_distribution)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("감정 워드클라우드", fontsize=16)
    return fig

def plot_bar_chart(emotion_distribution, top_n=20):
    sorted_items = sorted(emotion_distribution.items(), key=lambda x: x[1], reverse=True)[:top_n]
    labels, values = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, values, color="skyblue")
    ax.set_title("상위 감정 분포", fontsize=16)
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout()
    return fig

def render_dominant_emotion(dominant):
    st.markdown(
        f"""
<div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;'>
    <span style='font-weight: 600; font-size: 1.1rem;'>🧠 지배적 감정:</span>
    <span style='background-color: #ffeaa7; color: #2d3436; padding: 6px 14px; border-radius: 16px; font-weight: bold; font-size: 1rem;'>
        💡 {dominant}
    </span>
</div>
""",
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(page_title="CASS 감정 대시보드", layout="wide")
    st.title("🎭 CASS: 감성 분석 대시보드")
    st.markdown("영상 기반 객체 감지 + 감성 태깅 + 가중치 기반 분석 결과 시각화")

    result_path = "data/results/final_sentiment.json"
    if not os.path.exists(result_path):
        st.warning("⚠️ 분석 결과가 존재하지 않습니다.")
        return

    data = load_sentiment_data(result_path)
    emotion_distribution = data.get("emotion_distribution", {})
    dominant = data.get("dominant_emotion", "없음")

    render_dominant_emotion(dominant)

    st.subheader("☁️ 감정 워드클라우드")
    st.pyplot(plot_wordcloud(emotion_distribution))

    st.subheader("📊 감정 분포 막대 그래프")
    st.pyplot(plot_bar_chart(emotion_distribution))

if __name__ == "__main__":
    main()
