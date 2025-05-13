import os
import json
import argparse
from collections import defaultdict

def load_json_or_default(path, module_name):
    if not os.path.exists(path):
        print(f"⚠️ {module_name} 결과 없음 → 제외 처리")
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
            scores = data.get("emotion_distribution") or data.get("emotion_scores") or {}
            if not scores:
                print(f"⚠️ {module_name} → 감정 점수 없음")
                return None
            return scores
    except Exception as e:
        print(f"❌ {module_name} 로드 실패: {e}")
        return None

def fuse_emotions(module_outputs, weights=None):
    if weights is None:
        weights = {module: 1.0 for module in module_outputs}

    final_scores = defaultdict(float)
    for module, score_dict in module_outputs.items():
        if not score_dict:
            continue
        w = weights.get(module, 1.0)
        for emo, score in score_dict.items():
            if emo == "중립":
                continue
            final_scores[emo] += float(score) * w

    if not final_scores:
        return {
            "dominant_emotion": "중립",
            "emotion_distribution": {"중립": 100.0},
            "source": "fusion"
        }

    total = sum(final_scores.values())
    normalized = {k: round(v / total * 100, 2) for k, v in final_scores.items()}
    dominant = max(normalized, key=normalized.get)

    return {
        "dominant_emotion": dominant,
        "emotion_distribution": normalized,
        "source": "fusion"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="멀티모달 감정 통합")
    parser.add_argument("--object", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--ocr", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    modules = {
        "object": load_json_or_default(args.object, "객체"),
        "audio": load_json_or_default(args.audio, "오디오/대사"),
        "ocr": load_json_or_default(args.ocr, "텍스트")
    }

    result = fuse_emotions({k: v for k, v in modules.items() if v})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ 감정 통합 완료 → {args.output}")
