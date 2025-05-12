#!/usr/bin/env python3
import sys
import os
import subprocess

def download_youtube_shorts(video_url, output_dir="data/raw_videos"):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "yt-dlp",
        "-f", "mp4",
        "--write-auto-sub",
        "--sub-lang", "en",
        "--write-thumbnail",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        video_url
    ]
    print("Downloading with command:", " ".join(command))
    subprocess.run(command)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python download_shorts.py [YouTube Shorts URL]")
        sys.exit(1)

    video_url = sys.argv[1]
    download_youtube_shorts(video_url)
