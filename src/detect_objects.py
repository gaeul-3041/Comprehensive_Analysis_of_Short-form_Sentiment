#!/usr/bin/env python3
import os
import argparse
import json
from ultralytics import YOLO
from PIL import Image

def detect_objects(input_dir, output_path, model_path='yolov8n.pt', conf=0.25):
    model = YOLO(model_path)
    results_dict = {}

    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(('.jpg', '.png')):
                image_path = os.path.join(root, file)
                print(f"🔍 Detecting objects in: {image_path}")
                results = model(image_path, conf=conf)
                result = results[0]
                labels = result.names
                classes = result.boxes.cls.tolist()
                detected_objects = [labels[int(cls)] for cls in classes]
                results_dict[file] = detected_objects

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Detection results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in frames using YOLO")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with input frames")
    parser.add_argument("--output", type=str, default="data/objects/detections.json", help="Output JSON path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model (e.g. yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    args = parser.parse_args()

    detect_objects(args.input_dir, args.output, args.model, args.conf)
