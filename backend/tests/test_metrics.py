from app.metrics import FALSE_POS, MISSED, evaluate, iou, nms, postprocess

CLASSES = ["almond", "walnut"]
KCAL = {"almond": 7.0, "walnut": 26.0}


def box(x, y, s=100):
    return [x, y, x + s, y + s]


def test_iou():
    assert iou(box(0, 0), box(0, 0)) == 1.0
    assert iou(box(0, 0), box(200, 200)) == 0.0
    assert round(iou(box(0, 0), box(50, 0)), 3) == round(5000 / 15000, 3)


def test_nms_keeps_highest_and_separate_boxes():
    dets = [
        {"nut_id": "almond", "box": box(0, 0), "confidence": 0.6},
        {"nut_id": "walnut", "box": box(5, 5), "confidence": 0.9},
        {"nut_id": "almond", "box": box(300, 300), "confidence": 0.5},
    ]
    kept = nms(dets, 0.5)
    assert [d["confidence"] for d in kept] == [0.9, 0.5]
    assert len(nms(dets, 0.5, agnostic=False)) == 3
    assert len(postprocess(dets, conf=0.55, iou_thr=0.5)) == 1


def test_perfect_prediction():
    gt = [{"nut_id": "almond", "box": box(0, 0)}, {"nut_id": "walnut", "box": box(300, 300)}]
    pred = [{**g, "confidence": 0.9} for g in gt]
    r = evaluate([{"gt": gt, "pred": pred}], CLASSES, KCAL)
    assert r["overall"]["map50"] == 1.0
    assert r["overall"]["count_exact_rate"] == 1.0
    assert r["overall"]["kcal_error_pct"] == 0.0


def test_wrong_class_missed_and_false_positive():
    gt = [{"nut_id": "almond", "box": box(0, 0)}, {"nut_id": "almond", "box": box(300, 0)}]
    pred = [
        {"nut_id": "walnut", "box": box(0, 0), "confidence": 0.8},   # 종류 틀림
        {"nut_id": "almond", "box": box(600, 600), "confidence": 0.7},  # 엉뚱한 곳
    ]
    r = evaluate([{"gt": gt, "pred": pred}], CLASSES, KCAL)
    assert r["per_class"]["almond"]["recall"] == 0.0
    assert r["per_class"]["almond"]["count_mae"] == 1.0
    assert r["confusion"]["almond"] == {"walnut": 1, MISSED: 1}
    assert r["confusion"][FALSE_POS] == {"almond": 1}
    assert r["overall"]["count_exact_rate"] == 0.0
    # 정답 14kcal, 예측 26+7=33kcal
    assert round(r["overall"]["kcal_error_pct"], 1) == round(100 * 19 / 14, 1)


def test_ap_penalises_low_ranked_hits():
    gt = [{"nut_id": "almond", "box": box(0, 0)}, {"nut_id": "almond", "box": box(300, 0)}]
    pred = [
        {"nut_id": "almond", "box": box(700, 700), "confidence": 0.9},  # 높은 신뢰도 오검출
        {"nut_id": "almond", "box": box(0, 0), "confidence": 0.5},
        {"nut_id": "almond", "box": box(300, 0), "confidence": 0.4},
    ]
    r = evaluate([{"gt": gt, "pred": pred}], CLASSES, KCAL)
    assert r["per_class"]["almond"]["recall"] == 1.0
    assert 0.6 < r["per_class"]["almond"]["ap50"] < 0.7  # 2/3 precision 에서 recall 1
