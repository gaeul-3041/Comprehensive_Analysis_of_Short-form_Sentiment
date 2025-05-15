import os
import argparse
import requests
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

def extract_scene_frames(video_path, scene_dir, threshold=30.0):
    print(f"\U0001F4FC Scene-based frame extraction from: {video_path}")
    print(f"\U0001F3DA Threshold for scene detection: {threshold}")

    # Scene detection setup
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    scene_list = scene_manager.get_scene_list()
    print(f"✅ {len(scene_list)} scenes detected")

    # Save one frame per scene
    os.makedirs(scene_dir, exist_ok=True)
    save_images(scene_list, video_manager, num_images=1, output_dir=scene_dir)
    print(f"🖼 Scene frames saved to: {scene_dir}")

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

def extract_frames(video_path, output_dir, threshold=30.0):
    video_output_dir = output_dir

    scene_dir = os.path.join(video_output_dir, "scenes")
    thumbnail_path = os.path.join(video_output_dir, "thumbnail.jpg")

    os.makedirs(video_output_dir, exist_ok=True)
    extract_scene_frames(video_path, scene_dir, threshold)
    download_youtube_thumbnail(video_path, thumbnail_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scene-based frame & thumbnail extractor for YouTube Shorts")
    parser.add_argument("--video", type=str, required=True, help="Input video path (.mp4)")
    parser.add_argument("--output_dir", type=str, default="data/frames", help="Directory to save frames and thumbnail")
    parser.add_argument("--threshold", type=float, default=30.0, help="Scene detection threshold")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, args.threshold)
