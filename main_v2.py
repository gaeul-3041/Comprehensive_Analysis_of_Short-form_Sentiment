
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

def main(video_path, comment_file=None, model_path="yolov8n.pt"):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    features_dir = "data/features"
    results_dir = "data/results"
    frame_dir = f"data/frames/{video_id}"
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. 프레임 추출 (OCR용)
    run_command([
        "python", "src/extract_frames.py",
        "--video", video_path,
        "--output_dir", frame_dir,
        "--interval", "1"
    ], "프레임 추출")

    # 2. Vision 임베딩 (프레임 기반 → 평균 벡터)
    run_command([
        "python", "src/extract_vision_feature.py",
        "--input_dir", frame_dir,
        "--output", f"{features_dir}/{video_id}_vision.json",
        "--model", model_path
    ], "YOLO 객체 기반 문장 임베딩")

    # 3. OCR 임베딩
    run_command([
        "python", "src/extract_ocr_feature.py",
        "--input_dir", frame_dir,
        "--output", f"{features_dir}/{video_id}_ocr.json"
    ], "OCR 임베딩")

    # 4. Audio 임베딩
    run_command([
        "python", "src/extract_audio_feature.py",
        "--input", video_path,
        "--output", f"{features_dir}/{video_id}_audio.json"
    ], "오디오 임베딩")

    # 5. Feature Fusion → 감정 예측
    fused_result = f"{results_dir}/{video_id}_fused_sentiment.json"
    run_command([
        "python", "src/fuse_features.py",
        "--vision", f"{features_dir}/{video_id}_vision_mean.json",
        "--ocr", f"{features_dir}/{video_id}_ocr.json",
        "--audio", f"{features_dir}/{video_id}_audio.json",
        "--output", fused_result
    ], "피처 융합 및 감정 예측")

    # 6. 댓글 감정 분석 (선택)
    comment_json = f"{results_dir}/{video_id}_comments.json"
    if comment_file and os.path.exists(comment_file):
        run_command([
            "python", "src/analyze_comments.py",
            "--input", comment_file,
            "--output", comment_json
        ], "댓글 기반 감정 분석")

    # 7. 평가
    if os.path.exists(comment_json):
        run_command([
            "python", "src/evaluate_emotion_similarity.py",
            "--gt", comment_json,
            "--pred", fused_result
        ], "댓글 기준 유사도 평가")

    print("\n🎉 프레임 기반 Feature Fusion 감정 분석 + 평가 완료!")
    print("👉 streamlit run dashboard.py 로 결과를 시각화해보세요!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="프레임 기반 Feature Fusion 감정 분석 파이프라인")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="로컬 영상 경로 (.mp4)")
    group.add_argument("--url", type=str, help="YouTube Shorts URL")
    parser.add_argument("--comment", type=str, help="댓글 텍스트 파일 (JSON list)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO 모델 경로")
    args = parser.parse_args()

    video_file = download_youtube_video(args.url) if args.url else args.video
    main(video_file, args.comment, args.model)
