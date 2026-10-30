"""YOLO 판정 기준값 고르기 (신뢰도 conf, 겹침 제거 iou, 확인 필요 review_conf).

val 분할(또는 앱 feedback 사진)에서 YOLO 원시 예측을 한 번만 뽑은 뒤, 기준값 조합을 바꿔 가며
'개수 완전 일치 비율'이 가장 높은 값을 고른다 (동률이면 F1). 결과는 가중치 옆 .json 에 저장되고
앱(YoloDetector)이 그대로 읽는다.

review_conf: 신뢰도 구간별 정밀도가 --review-precision 보다 낮은 구간의 박스는
'확인 필요'(점선 표시, 하이브리드 모드면 Claude 재확인)로 표시하도록 정한다.

    python tune.py --weights runs/nuts/weights/best.pt --data val
"""

import argparse
import json
from pathlib import Path

from common import class_names
from evaluate import kcal_per_piece, load_eval_set, predict_all  # backend 경로도 함께 등록됨
from app.metrics import evaluate, greedy_match, postprocess

CONF_GRID = [round(0.05 * i, 2) for i in range(2, 15)]   # 0.10 ~ 0.70
IOU_GRID = [round(0.05 * i, 2) for i in range(6, 15)]    # 0.30 ~ 0.70


def sweep(raw_samples: list[dict], classes: list[str], kcal: dict) -> tuple[dict, dict]:
    """raw_samples: [{"gt", "pred"(원시)}] → (최적 기준값, 그때 지표)"""
    best_key, best = None, None
    for conf in CONF_GRID:
        for iou_thr in IOU_GRID:
            samples = [{"gt": s["gt"], "pred": postprocess(s["pred"], conf, iou_thr)} for s in raw_samples]
            m = evaluate(samples, classes, kcal)
            key = (m["overall"]["count_exact_rate"] or 0, m["overall"]["f1"] or 0, -abs(conf - 0.25))
            if best_key is None or key > best_key:
                best_key, best = key, ({"conf": conf, "iou": iou_thr}, m)
    return best


def review_threshold(samples: list[dict], conf: float, target_precision: float = 0.9) -> float:
    """기준값을 적용한 예측에서, 신뢰도 0.1 구간별 정밀도가 목표보다 낮은 가장 높은 구간의 윗값."""
    hits = []
    for s in samples:
        for pi, gi in greedy_match(s["pred"], s["gt"], 0.5, same_class=True):
            hits.append((s["pred"][pi]["confidence"], gi is not None))
    review = conf
    edge = conf
    while edge < 1.0:
        top = round(edge + 0.1, 2)
        in_bin = [hit for c, hit in hits if edge <= c < top]
        if in_bin and sum(in_bin) / len(in_bin) < target_precision:
            review = min(top, 1.0)
        edge = top
    return round(review, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", default="val", help="val / app / <YOLO 폴더> (test 로는 튜닝하지 마세요)")
    parser.add_argument("--review-precision", type=float, default=0.9)
    args = parser.parse_args()
    if args.data == "test":
        raise SystemExit("test 분할로 튜닝하면 최종 성능이 부풀려집니다. val 을 쓰세요.")

    from app.detector import YoloDetector

    detector = YoloDetector(args.weights)
    items = load_eval_set(args.data)
    raw = predict_all(lambda image, media: detector.raw(image), items, workers=1)
    raw_samples = [{"gt": gt, "pred": pred} for (_, gt), pred in zip(items, raw)]

    thresholds, metrics = sweep(raw_samples, class_names(), kcal_per_piece())
    kept = [{"gt": s["gt"], "pred": postprocess(s["pred"], thresholds["conf"], thresholds["iou"])} for s in raw_samples]
    thresholds["review_conf"] = review_threshold(kept, thresholds["conf"], args.review_precision)

    out = args.weights.with_suffix(".json")
    out.write_text(json.dumps(thresholds, indent=2))
    o = metrics["overall"]
    print(f"최적 기준값: {thresholds}  (사진 {len(items)}장)")
    print(f"  개수 완전 일치 {o['count_exact_rate']:.1%} · F1 {o['f1'] or 0:.3f}")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
