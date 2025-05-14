import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import easyocr
import torch
from transformers import AutoTokenizer, AutoModel

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained("kakaocorp/kanana-nano-2.1b-embedding")
model = AutoModel.from_pretrained("kakaocorp/kanana-nano-2.1b-embedding").to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)[0].cpu().tolist()

def extract_ocr(input_dir, output_path):
    reader = easyocr.Reader(['en', 'ko'])
    combined_text = ""

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(input_dir, file)
            result = reader.readtext(image_path, detail=0)
            combined_text += " ".join(result) + "\n"

    text = combined_text.strip()
    if not text:
        text = "No readable text."

    embedding = get_embedding(text)

    result = {
        "text": text,
        "embedding": embedding
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ OCR 임베딩 저장 완료 → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract_ocr(args.input_dir, args.output)