import os
import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter


def load_embedding(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        return data["embedding"]


def fuse_embeddings(vision_vec, ocr_vec, audio_vec):
    v1 = np.array(vision_vec)
    v2 = np.array(ocr_vec)
    v3 = np.array(audio_vec)
    return np.concatenate([v1, v2, v3], axis=0)


def make_prompt_from_vector(vector):
    return "이 장면의 특징을 종합적으로 고려했을 때 감정을 한 단어로 알려줘.\n감정:"


def classify_emotion_kanana(prompt, tokenizer, model, num_samples=10, top_k=10):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    generations = model.generate(
        input_ids=input_ids,
        do_sample=True,
        top_k=top_k,
        num_return_sequences=num_samples,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id
    )

    emotion_labels = [
        "기쁨", "슬픔", "분노", "놀람", "짜증", "불안", "당황", "상처",
        "행복", "중립", "설렘", "후회", "두려움", "실망", "감동", "사랑",
        "고마움", "혐오", "부끄러움", "혼란", "그리움", "우울", "신남",
        "만족", "무관심", "놀람", "불쾌", "자신감"
    ]

    counts = Counter()
    for output in generations:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        if "감정:" in decoded:
            prediction = decoded.split("감정:")[-1].strip().split("\n")[0]
        else:
            prediction = decoded.strip().split("\n")[-1]
        prediction = prediction.strip().split()[0]
        if prediction in emotion_labels:
            counts[prediction] += 1
        else:
            counts["중립"] += 1

    total = sum(counts.values())
    distribution = {emo: round(count / total, 4) for emo, count in counts.items()}
    dominant = max(distribution, key=distribution.get)
    return dominant, distribution


def main(vision_path, ocr_path, audio_path, output_path):
    print("🚀 피처 융합 및 감정 예측 중...")
    v = load_embedding(vision_path)
    o = load_embedding(ocr_path)
    a = load_embedding(audio_path)
    fused = fuse_embeddings(v, o, a)

    prompt = make_prompt_from_vector(fused)
    model_id = "Gaeul2/kanana-nano-emotion-lora"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

    dominant, distribution = classify_emotion_kanana(prompt, tokenizer, model)
    result = {
        "dominant_emotion": dominant,
        "emotion_distribution": distribution,
        "source": "feature_fusion"
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ 감정 예측 결과 저장 완료 → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="멀티모달 임베딩 융합 + 감정 예측기")
    parser.add_argument("--vision", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    main(args.vision, args.ocr, args.audio, args.output)
