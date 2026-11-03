"""사용자가 찍고 정답을 고친 사진 저장소.

- test     : 정확도 측정 전용 (학습·증강에 절대 쓰지 않음)
- feedback : 재학습용 (training/sources.yaml 에 format: yolo 로 추가해 사용)

폴더 구조 (YOLO 형식이라 학습 파이프라인이 그대로 읽는다):
    <split>/images/<id>.jpg
    <split>/labels/<id>.txt      정답 박스 (YOLO)
    <split>/meta/<id>.json       정답 + 당시 인식기 예측 + 인식기 이름 + 저장 시각
    <split>/data.yaml
"""

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .nutrition import load_nuts

SAMPLES_DIR = Path(os.getenv("NUT_SAMPLES_DIR", Path(__file__).resolve().parents[1] / "data" / "samples"))
SPLITS = ("test", "feedback")


def _split_dir(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"split 은 {SPLITS} 중 하나여야 합니다.")
    return SAMPLES_DIR / split


def _yolo_rows(gt: list[dict]) -> str:
    names = list(load_nuts())
    lines = []
    for g in gt:
        x1, y1, x2, y2 = (v / 1000 for v in g["box"])
        lines.append(f"{names.index(g['nut_id'])} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
    return "".join(lines)


def save_sample(image: bytes, gt: list[dict], pred: list[dict], detector: str, split: str) -> str:
    base = _split_dir(split)
    now = datetime.now(timezone.utc)
    sample_id = f"{now:%Y%m%d_%H%M%S}_{secrets.token_hex(4)}"
    for sub in ("images", "labels", "meta"):
        (base / sub).mkdir(parents=True, exist_ok=True)

    (base / "images" / f"{sample_id}.jpg").write_bytes(image)
    (base / "labels" / f"{sample_id}.txt").write_text(_yolo_rows(gt))
    (base / "meta" / f"{sample_id}.json").write_text(json.dumps(
        {"id": sample_id, "split": split, "detector": detector, "created_at": now.isoformat(), "gt": gt, "pred": pred},
        ensure_ascii=False))
    data_yaml = base / "data.yaml"
    if not data_yaml.exists():
        data_yaml.write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(load_nuts())))
    return sample_id


def load_samples(split: str) -> list[dict]:
    meta_dir = _split_dir(split) / "meta"
    if not meta_dir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(meta_dir.glob("*.json"))]


def delete_sample(split: str, sample_id: str) -> bool:
    base = _split_dir(split)
    if not sample_id.replace("_", "").isalnum():
        return False
    found = False
    for sub, ext in (("images", "jpg"), ("labels", "txt"), ("meta", "json")):
        path = base / sub / f"{sample_id}.{ext}"
        if path.exists():
            path.unlink()
            found = True
    return found

