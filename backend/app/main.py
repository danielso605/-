"""견과류 인식 & 칼로리 안내 API."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import samples
from .detector import UNKNOWN, DetectionError, create_detector
from .metrics import evaluate
from .nutrition import DAILY_RECOMMENDED_G, load_nuts, nutrition_for, summarize

BACKEND_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"
REPORTS_DIR = BACKEND_DIR / "data" / "reports"
MODEL_DIR = BACKEND_DIR / "models"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="견과류 인식 & 칼로리 안내")
detector = create_detector()


async def read_image(image: UploadFile) -> bytes:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "JPG, PNG, WEBP 사진만 올릴 수 있어요.")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "사진 용량이 너무 커요 (최대 10MB).")
    return data


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "detector": detector.name}


@app.get("/api/nuts")
def list_nuts() -> dict:
    return {"daily_recommended_g": DAILY_RECOMMENDED_G, "nuts": list(load_nuts().values())}


def model_classes() -> list[str] | None:
    for d in (detector, getattr(detector, "primary", None)):
        if d is not None and hasattr(d, "class_ids"):
            return d.class_ids
    return None


@app.get("/api/config")
def config() -> dict:
    """휴대폰 내 인식(ONNX)용 모델이 배포돼 있는지 알려 준다."""
    onnx = MODEL_DIR / "nuts.onnx"
    thresholds = MODEL_DIR / "nuts.json"
    return {
        "detector": detector.name,
        "ondevice_model": "/models/nuts.onnx" if onnx.exists() else None,
        # 휴대폰용 ONNX 출력 순서 = 학습한 모델의 클래스 순서
        "classes": model_classes() or list(load_nuts()),
        "thresholds": json.loads(thresholds.read_text()) if thresholds.exists() else None,
    }


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    data = await read_image(image)
    try:
        detections = detector.detect(data, image.content_type)
    except DetectionError as e:
        raise HTTPException(502, str(e)) from e

    nuts = load_nuts()
    counts = Counter(d["nut_id"] for d in detections if d["nut_id"] != UNKNOWN)
    items = [
        {
            "nut_id": nut_id,
            "name_ko": nuts[nut_id]["name_ko"],
            "count": count,
            "nutrition": nutrition_for(nut_id, count),
        }
        for nut_id, count in counts.most_common()
    ]
    return {
        "detector": detector.name,
        "detections": detections,
        "items": items,
        "unknown_count": sum(1 for d in detections if d["nut_id"] == UNKNOWN),
        "needs_review": sum(1 for d in detections if d.get("needs_review")),
        "total": summarize(items),
        "daily_recommended_g": DAILY_RECOMMENDED_G,
    }


class Box(BaseModel):
    nut_id: str
    box: list[float] = Field(min_length=4, max_length=4)
    confidence: float | None = None


class SamplePayload(BaseModel):
    split: str
    consent: bool
    detector: str
    gt: list[Box]
    pred: list[Box]


@app.post("/api/samples")
async def save_sample(image: UploadFile = File(...), payload: str = Form(...)) -> dict:
    """사진 + 사용자가 고친 정답 + 당시 예측을 저장한다 (정확도 측정·재학습용)."""
    try:
        body = SamplePayload.model_validate_json(payload)
    except ValidationError as e:
        raise HTTPException(422, "저장 형식이 올바르지 않아요.") from e
    if not body.consent:
        raise HTTPException(400, "사진 저장에 동의해야 저장할 수 있어요.")
    if body.split not in samples.SPLITS:
        raise HTTPException(400, "split 은 test 또는 feedback 이어야 해요.")
    known = set(load_nuts())
    if any(b.nut_id not in known for b in body.gt):
        raise HTTPException(400, "알 수 없는 견과류 종류가 있어요.")

    data = await read_image(image)
    sample_id = samples.save_sample(
        data,
        gt=[b.model_dump(exclude={"confidence"}) for b in body.gt],
        pred=[b.model_dump() for b in body.pred],
        detector=body.detector,
        split=body.split,
    )
    return {"id": sample_id, "count": len(samples.load_samples(body.split))}


@app.get("/api/samples")
def list_samples() -> dict:
    return {split: len(samples.load_samples(split)) for split in samples.SPLITS}


@app.delete("/api/samples/{split}/{sample_id}")
def delete_sample(split: str, sample_id: str) -> dict:
    if split not in samples.SPLITS or not samples.delete_sample(split, sample_id):
        raise HTTPException(404, "해당 사진이 없어요.")
    return {"deleted": sample_id}


@app.get("/api/report")
def report() -> dict:
    """정확도 리포트: 앱에서 찍은 테스트 사진(인식기별) + 오프라인 평가 결과 + 모델 교체 이력."""
    nuts = load_nuts()
    classes = list(nuts)
    kcal = {c: nutrition_for(c, 1)["kcal"] for c in classes}

    by_detector = defaultdict(list)
    for s in samples.load_samples("test"):
        by_detector[s["detector"]].append(s)
    live = [
        {"detector": name, "metrics": evaluate(items, classes, kcal),
         "first": items[0]["created_at"], "last": items[-1]["created_at"]}
        for name, items in sorted(by_detector.items())
    ]

    offline = []
    if REPORTS_DIR.exists():
        offline = [json.loads(p.read_text()) for p in sorted(REPORTS_DIR.glob("*.json"))]
    registry = MODEL_DIR / "registry.json"
    history = json.loads(registry.read_text())["versions"] if registry.exists() else []

    return {
        "classes": classes,
        "names_ko": {c: nuts[c]["name_ko"] for c in classes},
        "live": live,
        "offline": offline,
        "history": history,
    }


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/label")
    def label_page() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "label.html")

    @app.get("/report")
    def report_page() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "report.html")

    @app.get("/sw.js")
    def service_worker() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "sw.js", media_type="text/javascript")

    @app.get("/models/nuts.onnx")
    def onnx_model() -> FileResponse:
        path = MODEL_DIR / "nuts.onnx"
        if not path.exists():
            raise HTTPException(404, "휴대폰용 모델이 아직 없어요.")
        return FileResponse(path, media_type="application/octet-stream")
