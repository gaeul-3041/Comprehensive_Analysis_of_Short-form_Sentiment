import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import tempfile
from moviepy.editor import VideoFileClip
from transformers import pipeline
import whisper

EN_KO_EMOTION_MAP = {
    "admiration": "감탄",
    "amusement": "재미",
    "anger": "분노",
    "annoyance": "짜증",
    "approval": "수긍",
    "caring": "보살핌",
    "confusion": "혼란",
    "curiosity": "호기심",
    "desire": "욕망",
    "disappointment": "실망",
    "disapproval": "거부감",
    "disgust": "혐오",
    "embarrassment": "당황",
    "excitement": "흥분",
    "fear": "두려움",
    "gratitude": "감사",
    "grief": "슬픔",
    "joy": "기쁨",
    "love": "사랑",
    "nervousness": "긴장",
    "optimism": "낙관",
    "pride": "자부심",
    "realization": "깨달음",
    "relief": "안도",
    "remorse": "후회",
    "sadness": "슬픔",
    "surprise": "놀람",
    "neutral": "중립"
}


def convert_mp4_to_mp3(input_path):
    """mp4 → mp3 변환"""
    try:
        mp3_path = tempfile.mktemp(suffix=".mp3")
        clip = VideoFileClip(input_path)
        clip.audio.write_audiofile(mp3_path, logger=None)
        return mp3_path
    except Exception as e:
        print(f"❌ mp4 → mp3 변환 실패: {e}")
        return None


def transcribe_audio(input_path, chunk=True):
    """Whisper로 자막 추출 (chunk 여부로 모델 다르게)"""
    try:
        print("🗣️ Whisper로 자막 추출 중...")
        if chunk:
            whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                chunk_length_s=30,
                stride_length_s=5,
                return_timestamps=False
            )
            result = whisper_pipe(input_path)
            transcript = result["text"].strip()
        else:
            model = whisper.load_model("base")
            result = model.transcribe(input_path)
            transcript = result.get("text", "").strip()

        print(f"✅ 텍스트 추출 완료: {transcript[:50]}...")
        return transcript
    except Exception as e:
        print(f"❌ 음성 인식 실패: {e}")
        return None


def analyze_audio(input_path, output_path):
    """오디오 감정 분석 전체 파이프라인"""
    try:
        print("🚀 오디오 감정 분석 시작...")

        # 1차: 대사 위주 판단
        transcript = transcribe_audio(input_path, chunk=False)
        if transcript and len(transcript.strip()) > 10:
            source_label = "speech"
        else:
            source_label = "audio"
            mp3_path = convert_mp4_to_mp3(input_path)
            if not mp3_path or not os.path.exists(mp3_path):
                raise RuntimeError("mp3 파일 생성 실패")
            transcript = transcribe_audio(mp3_path, chunk=True)
            if mp3_path and os.path.exists(mp3_path):
                os.remove(mp3_path)

        if not transcript or len(transcript.strip()) == 0:
            raise RuntimeError("음성 텍스트 추출 실패")

        classifier = pipeline(
            "text-classification",
            model="fyaronskiy/ModernBERT-large-english-go-emotions",
            tokenizer="fyaronskiy/ModernBERT-large-english-go-emotions",
            top_k=None
        )

        results = classifier(transcript, top_k=None)

        # 감정 한글 매핑
        emotions = {
            EN_KO_EMOTION_MAP.get(e["label"], e["label"]): float(e["score"])
            for e in results
        }
        dominant = max(emotions, key=emotions.get)

        output = {
            "dominant_emotion": dominant,
            "emotion_distribution": emotions,
            "transcript": transcript,
            "source": source_label
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"✅ 오디오 감정 분석 완료 → {output_path}")

    except Exception as e:
        print(f"❌ 오디오 감정 분석 실패: {e}")
        fallback = {
            "dominant_emotion": "중립",
            "emotion_distribution": {"중립": 1.0},
            "transcript": "(음성 인식 실패)",
            "source": "audio"
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fallback, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 mp4 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSON 파일 경로")
    args = parser.parse_args()
    analyze_audio(args.input, args.output)
