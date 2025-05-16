import os
import argparse
import requests
import cv2
import math
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

def extract_scene_frames(video_path, scene_dir, threshold):
    print(f"\U0001F4FC Scene-based frame extraction from: {video_path}")
    print(f"\U0001F3DA Threshold for scene detection: {threshold}")

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    print(f"✅ {len(scene_list)} scenes detected")

    os.makedirs(scene_dir, exist_ok=True)

    if len(scene_list) >= 3:
        save_images(scene_list, video_manager, num_images=1, output_dir=scene_dir)
        print(f"\U0001F5BC Scene frames saved to: {scene_dir}")
    else:
        print("⚠️ 씬이 충분히 감지되지 않아 fallback 프레임 추출로 대체")
        video_manager.release()
        extract_fallback_frames(video_path, scene_dir, interval_sec=3.0)

def extract_fallback_frames(video_path, output_dir, interval_sec=3.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_secs = math.floor(frame_count / fps)
    print(f"⏱ fallback 프레임 추출: {interval_sec}초 간격 / 총 {total_secs}초")

    count = 0
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out_path = os.path.join(output_dir, f"fallback-frame-{idx:03d}.jpg")
            cv2.imwrite(out_path, frame)
            idx += 1
        count += 1
    cap.release()
    print(f"✅ fallback 프레임 {idx}개 저장 완료 → {output_dir}")

def download_youtube_thumbnail(video_path, thumbnail_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    thumbnail_url = f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"

    print(f"\U0001F304 Downloading thumbnail for {video_id}")
    try:
        response = requests.get(thumbnail_url, timeout=5)
        response.raise_for_status()
        with open(thumbnail_path, "wb") as f:
            f.write(response.content)
        print(f"✅ 썸네일 저장 완료 → {thumbnail_path}")
    except Exception as e:
        print(f"⚠️ 썸네일 다운로드 실패: {e}")

def extract_frames(video_path, output_dir, threshold):
    video_output_dir = output_dir
    scene_dir = output_dir
    thumbnail_path = os.path.join(video_output_dir, "thumbnail.jpg")

    os.makedirs(video_output_dir, exist_ok=True)
    extract_scene_frames(video_path, scene_dir, threshold)
    download_youtube_thumbnail(video_path, thumbnail_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene-based frame & thumbnail extractor with fallback for YouTube Shorts")
    parser.add_argument("--video", type=str, required=True, help="Input video path (.mp4)")
    parser.add_argument("--output_dir", type=str, default="data/frames", help="Directory to save frames and thumbnail")
    parser.add_argument("--threshold", type=float, default=30.0, help="Scene detection threshold")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.threshold)
