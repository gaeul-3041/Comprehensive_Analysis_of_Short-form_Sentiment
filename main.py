#!/usr/bin/env python3
# main.py
# YouTube Shorts URL 또는 로컬 영상 입력으로 전체 감성 분석 파이프라인 자동 실행

import os
import argparse
import subprocess

def run_command(command_list, desc):
    print(f"🚀 {desc}...")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"❌ {desc} 실패!")
        exit(1)
    print(f"✅ {desc} 완료!")

def download_youtube_shorts(url, output_dir="data/raw_videos"):
    os.makedirs(output_dir, exist_ok=True)
    video_id = url.strip("/").split("/")[-1]
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    cmd = [
        "yt-dlp", "-f", "mp4",
        "-o", output_path,
        url
    ]
    run_command(cmd, f"YouTube Shorts 다운로드 ({video_id})")
    return output_path

def main(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = f"data/frames/{video_id}"
    detect_output = "data/objects/detections.json"
    sentiment_output = "data/results/final_sentiment.json"
    wordcloud_output = "data/results/wordcloud.png"

    run_command([
        "python", "src/extract_frames.py",
        "--video", video_path,
        "--output_dir", "data/frames",
        "--interval", "1"
    ], "프레임 추출")

    run_command([
        "python", "src/detect_objects.py",
        "--input_dir", frame_dir,
        "--output", detect_output
    ], "YOLO 객체 감지")

    run_command([
        "python", "src/classify_sentiment.py",
        "--detection", detect_output,
        "--sentimap", "config/sentiment_map.json",
        "--output", sentiment_output
    ], "감성 매핑")

    run_command([
        "python", "src/visualize_results.py",
        "--input", sentiment_output,
        "--output", wordcloud_output
    ], "워드클라우드 생성")

    print("\n✨ 모든 파이프라인 작업이 완료되었습니다!")
    print("👉 다음 명령어로 대시보드를 실행해보세요:")
    print("   streamlit run dashboard.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전체 멀티모달 감성 분석 파이프라인 실행기 (YouTube 쇼츠 지원)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="로컬 영상 경로 (.mp4)")
    group.add_argument("--url", type=str, help="YouTube Shorts URL")
    args = parser.parse_args()

    if args.url:
        print(f"🌐 URL로부터 영상 다운로드 시작: {args.url}")
        video_file = download_youtube_shorts(args.url)
    else:
        video_file = args.video

    main(video_file)
