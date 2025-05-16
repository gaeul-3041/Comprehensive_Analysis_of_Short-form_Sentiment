import os
import json
import platform
import subprocess
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud

# 한글 폰트 설정
if platform.system() == "Windows":
    font_path = "C:/Windows/Fonts/malgun.ttf"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc("font", family=font_name)

st.set_page_config(page_title="CASS 감정 분석 대시보드", layout="wide")
st.markdown("<h1 style='text-align:center;'>🎬 CASS 감정 분석 결과 시각화</h1>", unsafe_allow_html=True)

# 📂 결과 파일 목록
result_files = [f for f in os.listdir("data/results") if f.endswith(".json") and ("sentiment" in f or "object" in f)]
if not result_files:
    st.warning("분석 결과가 존재하지 않습니다. 먼저 main.py를 실행해 주세요.")
    st.stop()

# 📁 선택 박스
selected_file = st.selectbox("📂 분석 결과 파일 선택", sorted(result_files))
filepath = os.path.join("data/results", selected_file)
video_id = selected_file.split("_")[0]

# 🎥 영상 제목 출력
meta_path = filepath.replace("_final_sentiment.json", "_meta.json").replace("_object.json", "_meta.json")
title = None
if os.path.exists(meta_path):
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
        title = meta.get("title")
else:
    try:
        result = subprocess.run(["yt-dlp", "-e", f"https://www.youtube.com/watch?v={video_id}"],
                                capture_output=True, text=True)
        title = result.stdout.strip()
    except Exception:
        pass

if title:
    st.markdown(f"<h3 style='text-align:center;'>🎥 영상 제목: {title}</h3>", unsafe_allow_html=True)

# 🎞 유튜브 영상 임베드
st.components.v1.html(f"""
<div style='text-align:center;'>
<iframe width="640" height="360"
    src="https://www.youtube.com/embed/{video_id}" 
    frameborder="0" allowfullscreen></iframe>
</div>
""", height=380)

# 📄 JSON 감정 결과 로드
with open(filepath, encoding='utf-8') as f:
    data = json.load(f)

# 📊 감정 차트 & 워드클라우드
def plot_emotion_charts(score_dict, color, unit="점유율", title="감정 분포"):
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    # 바 차트: 상위 10개
    bar_items = sorted_items[:10]
    bar_labels = [k for k, _ in bar_items]
    bar_values = [v for _, v in bar_items]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(bar_labels, bar_values, color=color)
        ax.set_ylabel(unit)
        ax.set_title(title)
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with col2:
        wc = WordCloud(
            font_path=font_path,
            background_color="white",
            width=800,
            height=400,
            max_words=200,
            colormap="tab20"
        )
        wc.generate_from_frequencies(dict(sorted_items))
        st.image(wc.to_array(), caption="감정 워드클라우드", use_column_width=True)

# 📦 객체 기반 (프레임별)
if isinstance(list(data.values())[0], list):
    total = {}
    for frames in data.values():
        for obj in frames:
            for emo, score in obj.get("sentiments", {}).items():
                total[emo] = total.get(emo, 0.0) + score

    if not total:
        st.warning("감정 정보가 포함되지 않은 결과입니다.")
        st.stop()

    dominant = max(total, key=total.get)
    st.markdown(f"""
    <div style='text-align:center; margin-top: 10px;'>
        <span style='display:inline-block; padding:10px 20px; border-radius:25px;
                     background:#d9f9e1; color:#1c7c54; font-size:18px; font-weight:bold;'>
            🏆 주요 감정: {dominant}
        </span>
    </div>
    """, unsafe_allow_html=True)

    plot_emotion_charts(total, color="skyblue", unit="점수 합계", title="감정 분포 (누적)")

# 🧠 통합 분석 결과 or ocr/audio
else:
    dominant = data.get("dominant_emotion", "(없음)")
    distribution = data.get("emotion_distribution", {})

    if not distribution:
        st.warning("감정 분포 정보가 없습니다.")
        st.stop()

    st.markdown(f"""
    <div style='text-align:center; margin-top: 10px;'>
        <span style='display:inline-block; padding:10px 20px; border-radius:25px;
                     background:#d9f9e1; color:#1c7c54; font-size:18px; font-weight:bold;'>
            🏆 주요 감정: {dominant}
        </span>
    </div>
    """, unsafe_allow_html=True)

    plot_emotion_charts(distribution, color="coral", unit="점유율", title="감정 분포")
