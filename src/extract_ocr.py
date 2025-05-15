import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import easyocr
from difflib import SequenceMatcher

def is_similar(a, b, threshold=0.5):
    return SequenceMatcher(None, a, b).ratio() > threshold

def run_ocr_summary(input_dir, output_path):
    print("🚀 OCR 정보 요약 (요약 없이 원문 그대로)...")
    print(f"🔍 OCR 텍스트 추출 중: {input_dir}")

    reader = easyocr.Reader(['en', 'ko'], gpu=False)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("❌ 분석할 이미지가 없습니다.")
        return

    combined_lines = []
    for file in image_files:
        image_path = os.path.join(input_dir, file)
        result = reader.readtext(image_path, detail=0)
        for text in result:
            line = text.strip()
            if line and not any(is_similar(line, existing) for existing in combined_lines):
                combined_lines.append(line)

    if not combined_lines:
        print("❌ OCR 텍스트 없음.")
        return

    output = {
        "text": "\n".join(combined_lines),
        "source": "ocr"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ OCR 원문 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EasyOCR 기반 OCR 원문 추출기 (요약 없음)")
    parser.add_argument("--input_dir", required=True, help="프레임 이미지 디렉토리")
    parser.add_argument("--output", required=True, help="출력 JSON 경로")
    args = parser.parse_args()

    run_ocr_summary(args.input_dir, args.output)
