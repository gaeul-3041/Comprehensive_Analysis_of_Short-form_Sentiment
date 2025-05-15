import cv2
import os
import json
import argparse
import numpy as np
from onnxruntime import InferenceSession

# dominant 감정만 추출하는 vision 분석

def run_vision_emotion_inference(image_dir, output_path, model_path="best.onnx"):
    print(f"\U0001F50D 얼굴 기반 감정 분류: {image_dir}")

    session = InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_shape = session.get_outputs()[0].shape
    num_emotions = output_shape[1] if output_shape else 7
    EMOTION_LABELS = [
        "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
    ]
    if len(EMOTION_LABELS) != num_emotions:
        print(f"⚠️ Warning: Model output size ({num_emotions}) does not match EMOTION_LABELS. Generating generic labels.")
        EMOTION_LABELS = [f"emotion_{i}" for i in range(num_emotions)]

    dominant_counts = {label: 0 for label in EMOTION_LABELS}
    frame_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))]
    if not frame_files:
        print("❌ 분석할 이미지가 없습니다.")
        return

    for file in sorted(frame_files):
        image_path = os.path.join(image_dir, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️ 이미지 로딩 실패: {file}")
            continue

        resized = cv2.resize(image, (640, 640))
        input_tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0][0]
        pred_idx = int(np.argmax(logits))
        if pred_idx < len(EMOTION_LABELS):
            dominant = EMOTION_LABELS[pred_idx]
            dominant_counts[dominant] += 1
        else:
            print(f"⚠️ Invalid prediction index {pred_idx} for image {file}")

    final_dominant = max(dominant_counts, key=dominant_counts.get)
    result = {
        "dominant_emotion": final_dominant,
        "frame_wise_count": dominant_counts,
        "source": "vision"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ 결과 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 얼굴 감정 모델 기반 분석")
    parser.add_argument("--image", type=str, required=True, help="프레임 이미지 디렉토리 경로")
    parser.add_argument("--output", type=str, required=True, help="결과 저장 JSON 경로")
    parser.add_argument("--model", type=str, default="best.onnx", help="ONNX 모델 경로")
    args = parser.parse_args()

    run_vision_emotion_inference(args.image, args.output, args.model)
