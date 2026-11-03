"""견과류 객체 인식기.

- ClaudeDetector: Claude 비전 모델로 사진 속 견과류를 하나하나 찾아 종류와 위치(박스)를 반환한다.
- YoloDetector: 직접 학습한 YOLO 가중치 (training/ 파이프라인으로 학습, 기준값은 가중치 옆 .json).
- HybridDetector: YOLO 우선, 애매하면 Claude 로 다시 확인.
- DemoDetector: API 키가 없을 때 화면 흐름을 확인하기 위한 가짜 결과.

모든 인식기는 같은 형식의 결과를 돌려준다:
    [{"nut_id": "almond", "box": [x1, y1, x2, y2], "confidence": 0.93, "needs_review": False}, ...]
box 좌표는 이미지 크기와 무관하게 0~1000 으로 정규화한 값이다.
"""

import base64
import json
import os
from pathlib import Path
from typing import Protocol

from .metrics import postprocess
from .nutrition import load_nuts

UNKNOWN = "unknown"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "nuts.pt"


class DetectionError(Exception):
    pass


class Detector(Protocol):
    name: str

    def detect(self, image: bytes, media_type: str) -> list[dict]: ...


def _detection_schema() -> dict:
    labels = [*load_nuts().keys(), UNKNOWN]
    return {
        "type": "object",
        "properties": {
            "detections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "nut_id": {"type": "string", "enum": labels},
                        "box": {"type": "array", "items": {"type": "integer"}},
                        "confidence": {"type": "number"},
                    },
                    "required": ["nut_id", "box", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["detections"],
        "additionalProperties": False,
    }


def _system_prompt() -> str:
    catalog = "\n".join(f"- {nut['id']}: {nut['name_ko']} ({nut['name_en']})" for nut in load_nuts().values())
    return f"""You are a food-vision system that finds every individual nut and raisin in a photo.

Report one detection per visible nut (kernel or in-shell) or raisin, even when many items of several kinds are mixed together or touching. Do not merge neighbouring items into one box, and do not skip partially hidden items at the edges of a pile.

For each nut give:
- nut_id: one of the ids below. Use "{UNKNOWN}" for a nut, seed or dried fruit that is not in the list. Ignore everything else (plates, hands, packaging).
- box: [x_min, y_min, x_max, y_max] tightly around that nut, in coordinates normalised to 0-1000 where (0, 0) is the top-left corner of the image.
- confidence: 0.0-1.0, how sure you are about the nut type.

Known ids:
{catalog}

Return an empty detections list if the photo contains none of them."""


class ClaudeDetector:
    name = "claude"

    def __init__(self, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("NUT_CLAUDE_MODEL", "claude-opus-5-5")

    def detect(self, image: bytes, media_type: str, instruction: str = "Detect every nut and raisin in this photo.") -> list[dict]:
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                # 안전 분류기가 요청을 거절하면 서버에서 대체 모델로 자동 재시도한다.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": "medium",
                    "format": {"type": "json_schema", "schema": _detection_schema()},
                },
                system=_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64.standard_b64encode(image).decode("utf-8"),
                                },
                            },
                            {"type": "text", "text": instruction},
                        ],
                    }
                ],
            )
        except anthropic.RateLimitError as e:
            raise DetectionError("요청이 많아 잠시 후 다시 시도해 주세요.") from e
        except anthropic.APIStatusError as e:
            raise DetectionError(f"인식 서버 오류 ({e.status_code})") from e
        except anthropic.APIConnectionError as e:
            raise DetectionError("인식 서버에 연결할 수 없어요.") from e

        if response.stop_reason == "refusal":
            raise DetectionError("이 사진은 분석할 수 없어요. 다른 사진으로 시도해 주세요.")
        if response.stop_reason == "max_tokens":
            raise DetectionError("견과류가 너무 많아요. 조금 나눠서 찍어 주세요.")

        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            detections = json.loads(text)["detections"]
        except (json.JSONDecodeError, KeyError) as e:
            raise DetectionError("인식 결과를 읽지 못했어요.") from e
        return [_clean(d) for d in detections if len(d.get("box", [])) == 4]


DEFAULT_THRESHOLDS = {"conf": 0.25, "iou": 0.5, "review_conf": 0.5}


def load_thresholds(weights: str | Path) -> dict:
    """가중치 옆의 같은 이름 .json (training/tune.py 가 만든 값). 없으면 기본값."""
    path = Path(weights).with_suffix(".json")
    return {**DEFAULT_THRESHOLDS, **json.loads(path.read_text())} if path.exists() else dict(DEFAULT_THRESHOLDS)


_EXTRA_ALIASES = {"peanuts": "peanut", "groundnut": "peanut", "pistachios": "pistachio", "cashewnut": "cashew",
                  "cashews": "cashew", "hazelnuts": "hazelnut", "filbert": "hazelnut", "walnuts": "walnut",
                  "almonds": "almond", "raisins": "raisin", "driedgrape": "raisin", "캐슈": "cashew", "헤이즐럿": "hazelnut"}


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_class(name: str) -> str:
    """모델 클래스 이름 → 앱 id. 'Almond', 'almonds', '아몬드', 'walnut_kernel' 처럼 달라도 맞춰 준다."""
    key = _norm(name)
    table = {}
    for nut_id, nut in load_nuts().items():
        for alias in (nut_id, nut["name_en"], nut["name_ko"]):
            table[_norm(alias)] = nut_id
    table.update({_norm(k): v for k, v in _EXTRA_ALIASES.items()})
    if key in table:
        return table[key]
    # 'walnut_kernel', 'roasted almond' 처럼 이름 일부에 들어 있는 경우
    hits = {v for k, v in table.items() if len(k) >= 4 and k in key}
    return hits.pop() if len(hits) == 1 else UNKNOWN


class YoloDetector:
    """직접 학습한 YOLO 모델(ultralytics). 클래스 이름은 resolve_class 로 앱 id 에 맞춘다.

    YOLO 자체 NMS 는 거의 끄고(raw) 앱 공통 후처리(metrics.postprocess)로 걸러서,
    tune.py 가 고른 기준값과 실제 앱 동작이 정확히 같게 한다.
    """

    name = "yolo"

    def __init__(self, weights: str | Path, thresholds: dict | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.thresholds = thresholds or load_thresholds(weights)
        # 모델 클래스 번호 → 앱 id (휴대폰용 ONNX 도 같은 순서를 쓴다)
        self.class_ids = [resolve_class(self.model.names[i]) for i in sorted(self.model.names)]

    def raw(self, image: bytes) -> list[dict]:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image)).convert("RGB")
        result = self.model.predict(img, conf=0.01, iou=0.95, max_det=600, verbose=False)[0]
        return [
            {
                "nut_id": self.class_ids[int(cls)],
                "box": [round(v * 1000) for v in box],
                "confidence": round(float(conf), 3),
            }
            for box, cls, conf in zip(result.boxes.xyxyn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
        ]

    def detect(self, image: bytes, media_type: str) -> list[dict]:
        t = self.thresholds
        kept = postprocess(self.raw(image), t["conf"], t["iou"])
        return [{**d, "needs_review": d["confidence"] < t["review_conf"]} for d in kept]


class HybridDetector:
    """평소엔 빠르고 무료인 YOLO, 애매한 사진만 Claude 에 다시 묻는다.

    YOLO 결과가 비었거나 '확인 필요' 박스 비율이 max_review_share 를 넘으면 Claude 결과를 쓴다.
    Claude 가 실패하면 YOLO 결과를 그대로 돌려준다 (점선 박스로 사용자 확인 유도).
    """

    name = "hybrid"

    def __init__(self, primary: "Detector", fallback: "Detector", max_review_share: float = 0.3):
        self.primary = primary
        self.fallback = fallback
        self.max_review_share = max_review_share

    def detect(self, image: bytes, media_type: str) -> list[dict]:
        dets = self.primary.detect(image, media_type)
        review = sum(1 for d in dets if d.get("needs_review"))
        if dets and review / len(dets) <= self.max_review_share:
            return [{**d, "source": self.primary.name} for d in dets]
        try:
            return [{**d, "source": self.fallback.name} for d in self.fallback.detect(image, media_type)]
        except DetectionError:
            return [{**d, "source": self.primary.name} for d in dets]


class DemoDetector:
    """API 키 없이 UI 를 확인하기 위한 고정 결과."""

    name = "demo"

    def detect(self, image: bytes, media_type: str) -> list[dict]:
        return [
            {"nut_id": "almond", "box": [80, 120, 230, 260], "confidence": 0.95},
            {"nut_id": "almond", "box": [260, 110, 400, 250], "confidence": 0.93},
            {"nut_id": "almond", "box": [430, 130, 570, 270], "confidence": 0.9},
            {"nut_id": "walnut", "box": [120, 420, 380, 680], "confidence": 0.92},
            {"nut_id": "walnut", "box": [420, 400, 680, 660], "confidence": 0.88},
            {"nut_id": "cashew", "box": [640, 140, 800, 300], "confidence": 0.86},
            {"nut_id": "pistachio", "box": [720, 480, 840, 600], "confidence": 0.81},
            {"nut_id": "raisin", "box": [860, 300, 940, 380], "confidence": 0.84},
        ]


def _clean(detection: dict) -> dict:
    x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in detection["box"])
    return {
        "nut_id": detection["nut_id"],
        "box": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        "confidence": round(max(0.0, min(1.0, float(detection["confidence"]))), 2),
    }


def create_detector() -> Detector:
    """환경변수로 인식기를 고른다. NUT_DETECTOR=claude|yolo|hybrid|demo
    (기본: models/nuts.pt 가 있고 키도 있으면 hybrid, 모델만 있으면 yolo, 키만 있으면 claude, 둘 다 없으면 demo)."""
    choice = os.getenv("NUT_DETECTOR")
    weights = Path(os.getenv("NUT_YOLO_WEIGHTS", DEFAULT_WEIGHTS))
    has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if choice is None:
        choice = ("hybrid" if has_key else "yolo") if weights.exists() else ("claude" if has_key else "demo")
    if choice == "yolo":
        return YoloDetector(weights)
    if choice == "hybrid":
        return HybridDetector(YoloDetector(weights), ClaudeDetector(),
                              float(os.getenv("NUT_HYBRID_MAX_REVIEW", "0.3")))
    if choice == "claude":
        return ClaudeDetector()
    return DemoDetector()
