"""인식 정확도 계산. 앱 리포트, 오프라인 평가(training/evaluate.py), 임계값 튜닝이 함께 쓴다.

박스는 앱 형식 [x1, y1, x2, y2] (0~1000 정규화)
    정답: {"nut_id", "box"}
    예측: {"nut_id", "box", "confidence"}
"""

from collections import Counter

MISSED = "missed"        # 정답이 있는데 못 찾음
FALSE_POS = "false_pos"  # 없는데 찾음


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[dict], iou_thr: float, agnostic: bool = True) -> list[dict]:
    """겹치는 박스 중 신뢰도 높은 것만 남긴다. agnostic=True 면 종류가 달라도 겹치면 하나로 본다."""
    kept: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["confidence"]):
        if all(iou(d["box"], k["box"]) <= iou_thr or (not agnostic and k["nut_id"] != d["nut_id"]) for k in kept):
            kept.append(d)
    return kept


def postprocess(dets: list[dict], conf: float, iou_thr: float) -> list[dict]:
    return nms([d for d in dets if d["confidence"] >= conf], iou_thr)


def greedy_match(preds: list[dict], gts: list[dict], iou_thr: float, same_class: bool):
    """신뢰도 높은 예측부터 가장 많이 겹치는 미매칭 정답에 짝짓는다. [(pred_idx, gt_idx|None)]"""
    used: set[int] = set()
    pairs = []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i].get("confidence", 1.0)):
        best, best_iou = None, iou_thr
        for gi, g in enumerate(gts):
            if gi in used or (same_class and g["nut_id"] != preds[pi]["nut_id"]):
                continue
            v = iou(preds[pi]["box"], g["box"])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            used.add(best)
        pairs.append((pi, best))
    return pairs


def _average_precision(scored: list[tuple[float, bool]], n_gt: int) -> float:
    """VOC 방식(모든 점 보간) AP."""
    if n_gt == 0:
        return float("nan")
    scored = sorted(scored, key=lambda s: -s[0])
    tp = fp = 0
    points = []
    for _, hit in scored:
        tp += hit
        fp += not hit
        points.append((tp / n_gt, tp / (tp + fp)))
    ap, prev_recall = 0.0, 0.0
    for i, (recall, _) in enumerate(points):
        precision = max(p for _, p in points[i:])
        ap += (recall - prev_recall) * precision
        prev_recall = recall
    return ap


def evaluate(samples: list[dict], classes: list[str], kcal_per_piece: dict[str, float],
             iou_thr: float = 0.5) -> dict:
    """samples: [{"gt": [...], "pred": [...]}] → 클래스별·전체 지표."""
    scored = {c: [] for c in classes}
    gt_total, pred_total, tp_total = Counter(), Counter(), Counter()
    count_abs_err = Counter()
    confusion = {g: Counter() for g in [*classes, FALSE_POS]}
    exact_images = 0
    kcal_errors = []

    for s in samples:
        gts = [g for g in s["gt"] if g["nut_id"] in classes]
        preds = [p for p in s["pred"] if p["nut_id"] in classes]

        for pi, gi in greedy_match(preds, gts, iou_thr, same_class=True):
            scored[preds[pi]["nut_id"]].append((preds[pi].get("confidence", 1.0), gi is not None))
            tp_total[preds[pi]["nut_id"]] += gi is not None

        matched_gt = set()
        for pi, gi in greedy_match(preds, gts, iou_thr, same_class=False):
            if gi is None:
                confusion[FALSE_POS][preds[pi]["nut_id"]] += 1
            else:
                matched_gt.add(gi)
                confusion[gts[gi]["nut_id"]][preds[pi]["nut_id"]] += 1
        for gi, g in enumerate(gts):
            if gi not in matched_gt:
                confusion[g["nut_id"]][MISSED] += 1

        gc = Counter(g["nut_id"] for g in gts)
        pc = Counter(p["nut_id"] for p in preds)
        gt_total.update(gc)
        pred_total.update(pc)
        for c in classes:
            count_abs_err[c] += abs(gc[c] - pc[c])
        exact_images += gc == pc
        gt_kcal = sum(kcal_per_piece[c] * n for c, n in gc.items())
        pred_kcal = sum(kcal_per_piece[c] * n for c, n in pc.items())
        if gt_kcal > 0:
            kcal_errors.append(abs(pred_kcal - gt_kcal) / gt_kcal)

    n = len(samples)
    per_class = {}
    for c in classes:
        p = tp_total[c] / pred_total[c] if pred_total[c] else None
        r = tp_total[c] / gt_total[c] if gt_total[c] else None
        ap = _average_precision(scored[c], gt_total[c])
        per_class[c] = {
            "gt": gt_total[c],
            "pred": pred_total[c],
            "tp": tp_total[c],
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None),
            "ap50": None if ap != ap else ap,
            "count_mae": count_abs_err[c] / n if n else None,
        }

    present = [c for c in classes if gt_total[c]]
    tp, gt_n, pred_n = sum(tp_total.values()), sum(gt_total.values()), sum(pred_total.values())
    precision = tp / pred_n if pred_n else None
    recall = tp / gt_n if gt_n else None
    return {
        "n_images": n,
        "iou_threshold": iou_thr,
        "overall": {
            "map50": sum(per_class[c]["ap50"] for c in present) / len(present) if present else None,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else None,
            "count_exact_rate": exact_images / n if n else None,
            "kcal_error_pct": 100 * sum(kcal_errors) / len(kcal_errors) if kcal_errors else None,
        },
        "per_class": per_class,
        "confusion": {g: dict(row) for g, row in confusion.items()},
    }
