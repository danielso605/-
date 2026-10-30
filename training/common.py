"""학습 스크립트 공통 유틸. 클래스 목록은 앱과 같은 backend/app/data/nuts.json 을 기준으로 한다."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUTS_JSON = ROOT.parent / "backend" / "app" / "data" / "nuts.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def class_names() -> list[str]:
    """YOLO 클래스 번호 순서대로의 id 목록 (nuts.json 배열 순서)."""
    with NUTS_JSON.open(encoding="utf-8") as f:
        return [nut["id"] for nut in json.load(f)["nuts"]]


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """YOLO 라벨 파일: 줄마다 `class cx cy w h` (0~1 정규화)."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def write_labels(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in rows))
