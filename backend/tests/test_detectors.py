from app.detector import DetectionError, HybridDetector, YoloDetector


class Fixed:
    def __init__(self, name, dets=None, error=False):
        self.name, self.dets, self.error = name, dets or [], error

    def detect(self, image, media_type):
        if self.error:
            raise DetectionError("down")
        return self.dets


SURE = {"nut_id": "almond", "box": [0, 0, 100, 100], "confidence": 0.9, "needs_review": False}
UNSURE = {"nut_id": "walnut", "box": [200, 200, 300, 300], "confidence": 0.3, "needs_review": True}
CLAUDE = [{"nut_id": "walnut", "box": [200, 200, 300, 300], "confidence": 0.95}]


def test_hybrid_keeps_confident_yolo_result():
    out = HybridDetector(Fixed("yolo", [SURE, SURE, SURE, UNSURE]), Fixed("claude", CLAUDE)).detect(b"", "image/jpeg")
    assert len(out) == 4 and {d["source"] for d in out} == {"yolo"}


def test_hybrid_falls_back_when_unsure_or_empty():
    assert HybridDetector(Fixed("yolo", [SURE, UNSURE]), Fixed("claude", CLAUDE)).detect(b"", "")[0]["source"] == "claude"
    assert HybridDetector(Fixed("yolo", []), Fixed("claude", CLAUDE)).detect(b"", "")[0]["source"] == "claude"


def test_hybrid_survives_claude_failure():
    out = HybridDetector(Fixed("yolo", [UNSURE]), Fixed("claude", error=True)).detect(b"", "")
    assert out[0]["source"] == "yolo" and out[0]["needs_review"]


def test_yolo_postprocess_uses_thresholds():
    det = object.__new__(YoloDetector)
    det.thresholds = {"conf": 0.3, "iou": 0.5, "review_conf": 0.6}
    det.raw = lambda image: [
        {"nut_id": "almond", "box": [0, 0, 100, 100], "confidence": 0.9},
        {"nut_id": "almond", "box": [5, 5, 100, 100], "confidence": 0.8},   # 중복 → NMS 로 제거
        {"nut_id": "walnut", "box": [300, 300, 400, 400], "confidence": 0.45},  # 확인 필요
        {"nut_id": "walnut", "box": [600, 600, 700, 700], "confidence": 0.1},   # 기준 미달 → 제거
    ]
    out = det.detect(b"", "image/jpeg")
    assert [(d["confidence"], d["needs_review"]) for d in out] == [(0.9, False), (0.45, True)]


def test_resolve_class_accepts_common_variants():
    from app.detector import UNKNOWN, resolve_class

    assert resolve_class("Almond") == "almond"
    assert resolve_class("아몬드") == "almond"
    assert resolve_class("Pistachios") == "pistachio"
    assert resolve_class("walnut_kernel") == "walnut"
    assert resolve_class("Brazil nut") == UNKNOWN
    assert resolve_class("건포도") == "raisin"
