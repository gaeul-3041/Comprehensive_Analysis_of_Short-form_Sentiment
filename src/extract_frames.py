#!/usr/bin/env python3
import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fps_interval=1):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    save_path = output_dir
    os.makedirs(save_path, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_interval)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📼 Extracting from: {video_path}")
    print(f"🎞 FPS: {fps}, Total Frames: {frame_count}, Saving every {frame_interval} frames")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_filename = os.path.join(save_path, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"✅ Done! Extracted {saved_count} frames to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video at set intervals.")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", type=str, default="data/frames", help="Output directory for frames")
    parser.add_argument("--interval", type=int, default=1, help="Seconds between each frame capture")
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir, fps_interval=args.interval)
