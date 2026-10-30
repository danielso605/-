"""견과류 인식 & 칼로리 안내 API."""

from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .detector import UNKNOWN, DetectionError, create_detector
from .nutrition import DAILY_RECOMMENDED_G, load_nuts, nutrition_for, summarize

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(title="견과류 인식 & 칼로리 안내")
detector = create_detector()


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "detector": detector.name}


@app.get("/api/nuts")
def list_nuts() -> dict:
    return {"daily_recommended_g": DAILY_RECOMMENDED_G, "nuts": list(load_nuts().values())}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)) -> dict:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "JPG, PNG, WEBP 사진만 올릴 수 있어요.")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "사진 용량이 너무 커요 (최대 10MB).")

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
        "total": summarize(items),
        "daily_recommended_g": DAILY_RECOMMENDED_G,
    }


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")
