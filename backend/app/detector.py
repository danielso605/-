"""견과류 객체 인식기.

- ClaudeDetector: Claude 비전 모델로 사진 속 견과류를 하나하나 찾아 종류와 위치(박스)를 반환한다.
- YoloDetector: 직접 학습한 YOLO 가중치가 있을 때 사용하는 온디바이스/서버 추론기 (선택).
- DemoDetector: API 키가 없을 때 화면 흐름을 확인하기 위한 가짜 결과.

모든 인식기는 같은 형식의 결과를 돌려준다:
    [{"nut_id": "almond", "box": [x1, y1, x2, y2], "confidence": 0.93}, ...]
box 좌표는 이미지 크기와 무관하게 0~1000 으로 정규화한 값이다.
"""

import base64
import json
import os
from typing import Protocol

from .nutrition import load_nuts

UNKNOWN = "unknown"


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
    return f"""You are a food-vision system that finds every individual nut in a photo.

Report one detection per visible nut kernel or shelled nut, even when many nuts of several kinds are mixed together or touching. Do not merge neighbouring nuts into one box, and do not skip partially hidden nuts at the edges of a pile.

For each nut give:
- nut_id: one of the ids below. Use "{UNKNOWN}" for something that is clearly a nut or seed but not in the list. Ignore objects that are not nuts (plates, hands, packaging, raisins, dried fruit).
- box: [x_min, y_min, x_max, y_max] tightly around that nut, in coordinates normalised to 0-1000 where (0, 0) is the top-left corner of the image.
- confidence: 0.0-1.0, how sure you are about the nut type.

Known nut ids:
{catalog}

Return an empty detections list if the photo contains no nuts."""


class ClaudeDetector:
    name = "claude"

    def __init__(self, model: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or os.getenv("NUT_CLAUDE_MODEL", "claude-opus-5-5")

    def detect(self, image: bytes, media_type: str) -> list[dict]:
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
                            {"type": "text", "text": "Detect every nut in this photo."},
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


class YoloDetector:
    """직접 학습한 YOLO 모델(ultralytics) 사용. 클래스 이름은 nuts.json 의 id 와 같아야 한다."""

    name = "yolo"

    def __init__(self, weights: str):
        from ultralytics import YOLO

        self.model = YOLO(weights)

    def detect(self, image: bytes, media_type: str) -> list[dict]:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image)).convert("RGB")
        result = self.model.predict(img, verbose=False)[0]
        known = load_nuts()
        detections = []
        for box, cls, conf in zip(result.boxes.xyxyn.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = result.names[int(cls)]
            detections.append(
                {
                    "nut_id": label if label in known else UNKNOWN,
                    "box": [round(v * 1000) for v in box],
                    "confidence": float(conf),
                }
            )
        return detections


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
        ]


def _clean(detection: dict) -> dict:
    x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in detection["box"])
    return {
        "nut_id": detection["nut_id"],
        "box": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        "confidence": round(max(0.0, min(1.0, float(detection["confidence"]))), 2),
    }


def create_detector() -> Detector:
    """환경변수로 인식기를 고른다. NUT_DETECTOR=claude|yolo|demo (기본: 키가 있으면 claude)."""
    choice = os.getenv("NUT_DETECTOR")
    if choice == "yolo":
        return YoloDetector(os.environ["NUT_YOLO_WEIGHTS"])
    if choice == "demo":
        return DemoDetector()
    if choice == "claude" or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return ClaudeDetector()
    return DemoDetector()
