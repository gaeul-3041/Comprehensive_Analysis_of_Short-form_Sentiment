import os
import argparse
import subprocess
import json

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

    # 제목 가져오기
    result = subprocess.run(["yt-dlp", "-e", url], capture_output=True, text=True)
    title = result.stdout.strip()
    meta_path = f"data/results/{video_id}_meta.json"
    os.makedirs("data/results", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"title": title}, f, ensure_ascii=False)

    # 영상 다운로드
    cmd = ["yt-dlp", "-f", "mp4", "-o", output_path, url]
    run_command(cmd, f"YouTube 영상 다운로드 ({video_id})")
    return output_path, video_id

def main(video_path, video_id, vgmap_path):
    frame_dir = f"data/frames/{video_id}"
    detect_output = f"data/objects/{video_id}_detections.json"
    audio_output = f"data/results/{video_id}_audio.json"
    sentiment_output = f"data/results/{video_id}_final_sentiment.json"
    wordcloud_output = f"data/results/{video_id}_wordcloud.png"

    run_command([
        "python", "src/extract_frames.py",
        "--video", video_path,
        "--output_dir", frame_dir,
        "--interval", "1"
    ], "프레임 추출")

    run_command([
        "python", "src/detect_objects.py",
        "--input_dir", frame_dir,
        "--output", detect_output,
        "--sentimap", "config/sentiment_map.json",
        "--vgmap", vgmap_path
    ], "YOLO 객체 감지 및 감정 태깅")

    run_command([
        "python", "src/audio_analysis.py",
        "--input", video_path,
        "--output", audio_output
    ], "오디오/대사 감정 분석")

    run_command([
        "python", "src/ocr_text_analysis.py",
        "--input_dir", frame_dir,
        "--output", f"data/results/{video_id}_ocr.json"
    ], "영상 텍스트 감정 분석")

    run_command([
        "python", "src/fuse_emotions.py",
        "--object", f"data/results/{video_id}_object.json",
        "--audio", audio_output,
        "--ocr", f"data/results/{video_id}_ocr.json",
        "--output", sentiment_output
    ], "감정 통합 분석")

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
    parser.add_argument("--vgmap", type=str, default="config/vg_sentiment_map.json", help="VG 감정 관계 사전 경로")
    args = parser.parse_args()

    if args.url:
        video_file, video_id = download_youtube_video(args.url)
    else:
        video_file = args.video
        video_id = os.path.splitext(os.path.basename(video_file))[0]

    main(video_file, video_id, args.vgmap)
