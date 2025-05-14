
import argparse
import json
from scipy.spatial.distance import cosine
import numpy as np

def load_distribution(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        return data.get("emotion_distribution", {})

def cosine_similarity(dict1, dict2):
    keys = sorted(set(dict1) | set(dict2))
    v1 = np.array([dict1.get(k, 0.0) for k in keys])
    v2 = np.array([dict2.get(k, 0.0) for k in keys])
    return 1 - cosine(v1, v2) if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0 else 0.0

def main(gt_path, pred_path):
    gt = load_distribution(gt_path)
    pred = load_distribution(pred_path)

    sim = cosine_similarity(gt, pred)
    print(f"📐 Cosine 유사도 (댓글 기준 vs 예측): {round(sim, 4)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="감정 분포 간 유사도 평가")
    parser.add_argument("--gt", required=True, help="정답 감정 JSON (댓글 기반)")
    parser.add_argument("--pred", required=True, help="예측 감정 JSON")
    args = parser.parse_args()
    main(args.gt, args.pred)
