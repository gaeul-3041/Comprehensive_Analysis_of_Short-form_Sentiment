import os
import argparse
import json
import whisper

def transcribe_audio(input_path):
    print(f"🔊 Transcribing audio: {input_path}")
    model = whisper.load_model("base")
    result = model.transcribe(input_path, fp16=False)
    return result["text"]

def run_audio_analysis(input_path, output_path):
    transcript = transcribe_audio(input_path)

    result = {
        "summary": transcript.strip(),  # 그대로 summary로 넘김
        "transcript": transcript.strip(),
        "source": "audio"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Audio transcript saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="오디오에서 텍스트 추출 (Whisper)")
    parser.add_argument("--input", type=str, required=True, help="입력 영상 파일 (.mp4)")
    parser.add_argument("--output", type=str, default="data/results/audio_summary.json", help="출력 파일 경로")
    args = parser.parse_args()

    run_audio_analysis(args.input, args.output)
