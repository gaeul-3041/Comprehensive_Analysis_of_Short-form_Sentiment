
import os
import argparse
import subprocess

def run_command(command_list, desc):
    print(f"\n🚀 {desc}...")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"❌ {desc} 실패!")
        exit(1)
    print(f"✅ {desc} 완료!")

def download_youtube_video(url, output_dir="data/raw_videos"):
    os.makedirs(output_dir, exist_ok=True)
    video_id = url.strip("/").split("/")[-1]
    output_path = os.path.join(output_dir, f"{video_id}.mp4")

    cmd = [
        "yt-dlp", "-f", "mp4",
        "-o", output_path,
        url
    ]
    run_command(cmd, f"YouTube 영상 다운로드 ({video_id})")
    return output_path

def main(video_path, model_path="yolov8n.pt"):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = f"data/frames/{video_id}"
    detect_output = f"data/objects/{video_id}_detections.json"
    sentiment_output = f"data/results/{video_id}_final_sentiment.json"
    wordcloud_output = f"data/results/{video_id}_wordcloud.png"
    comment_path = f"data/results/{video_id}_comments.json"

    # 1. 프레임 추출
    run_command([
        "python", "src/extract_frames.py",
        "--video", video_path,
        "--output_dir", frame_dir,
        "--interval", "1"
    ], "프레임 추출")

    # 2. 객체 감지 + GPT 기반 감정 추론
    run_command([
        "python", "src/detect_objects.py",
        "--input_dir", frame_dir,
        "--output", detect_output,
        "--model", model_path
    ], "YOLO 객체 감지 + 감정 추론")

    # 3. 오디오 감정 분석
    run_command([
        "python", "src/audio_analysis.py",
        "--input", video_path,
        "--output", f"data/results/{video_id}_audio.json"
    ], "오디오 감정 분석")

    # 4. OCR 텍스트 감정 분석
    run_command([
        "python", "src/ocr_text_analysis.py",
        "--input_dir", frame_dir,
        "--output", f"data/results/{video_id}_ocr.json"
    ], "영상 텍스트 감정 분석")

    # 5. 감정 통합 분석
    run_command([
        "python", "src/fuse_emotions.py",
        "--object", f"data/results/{video_id}_object.json",
        "--audio", f"data/results/{video_id}_audio.json",
        "--ocr", f"data/results/{video_id}_ocr.json",
        "--output", sentiment_output
    ], "감정 통합")

    # 6. 댓글 감정 기준 평가 (존재할 경우)
    if os.path.exists(comment_path):
        run_command([
            "python", "src/evaluate_emotion_similarity.py",
            "--gt", comment_path,
            "--pred", sentiment_output
        ], "댓글 기반 감정 유사도 평가")

    # 7. 워드클라우드 시각화
    run_command([
        "python", "src/visualize_results.py",
        "--input", sentiment_output,
        "--output", wordcloud_output
    ], "감정 결과 시각화")

    print("\n✨ 모든 분석이 완료되었습니다!")
    print("👉 streamlit run dashboard.py 로 대시보드를 실행해보세요!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CASS 멀티모달 감정 분석 파이프라인")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="로컬 영상 경로 (.mp4)")
    group.add_argument("--url", type=str, help="YouTube Shorts URL")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 모델 경로")
    args = parser.parse_args()

    video_file = download_youtube_video(args.url) if args.url else args.video
    main(video_file, args.model)
