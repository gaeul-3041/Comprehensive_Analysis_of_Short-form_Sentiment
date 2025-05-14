import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import tempfile
from moviepy.editor import VideoFileClip
import whisper
import torch
from transformers import AutoTokenizer, AutoModel


# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로딩
whisper_model = whisper.load_model("base", device=device)
tokenizer = AutoTokenizer.from_pretrained("kakaocorp/kanana-nano-2.1b-embedding")
model = AutoModel.from_pretrained("kakaocorp/kanana-nano-2.1b-embedding").to(device)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)[0].cpu().tolist()

def convert_mp4_to_mp3(input_path):
    mp3_path = tempfile.mktemp(suffix=".mp3")
    clip = VideoFileClip(input_path)
    clip.audio.write_audiofile(mp3_path, logger=None)
    return mp3_path

def extract_audio(input_path, output_path):
    try:
        mp3_path = convert_mp4_to_mp3(input_path)
        result = whisper_model.transcribe(mp3_path)
        text = result["text"].strip()
        os.remove(mp3_path)

        if not text:
            text = "No speech detected."

        embedding = get_embedding(text)

        result = {
            "text": text,
            "embedding": embedding
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✅ 오디오 임베딩 저장 완료 → {output_path}")

    except Exception as e:
        print(f"❌ 오디오 처리 실패: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    extract_audio(args.input, args.output)