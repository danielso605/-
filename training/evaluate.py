"""인식기 정확도 평가 → backend/data/reports/*.json (앱의 /report 화면에 그래프로 표시됨).

평가 데이터 (--data):
    app      앱 인식 테스트 화면에서 저장한 실사진 (backend/data/samples/test)  ← 가장 중요
    test     datasets/nuts 의 test 분할
    val      datasets/nuts 의 val 분할
    <폴더>    images/ + labels/ 가 있는 YOLO 형식 폴더

인식기 (--detector, 여러 개 가능):
    yolo:<가중치.pt>   (기준값은 가중치 옆 .json)
    claude
    demo

    python evaluate.py --data app --detector yolo:../backend/models/nuts.pt claude
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from build_dataset import OUT
from common import IMAGE_EXTS, ROOT, class_names, read_labels

sys.path.insert(0, str(ROOT.parent / "backend"))
from app.detector import ClaudeDetector, DemoDetector, YoloDetector  # noqa: E402
from app.metrics import evaluate  # noqa: E402
from app.nutrition import nutrition_for  # noqa: E402
from app.samples import load_samples, SAMPLES_DIR  # noqa: E402

REPORTS_DIR = ROOT.parent / "backend" / "data" / "reports"


def yolo_rows_to_boxes(rows, names) -> list[dict]:
    return [{"nut_id": names[c], "box": [round((cx - w / 2) * 1000), round((cy - h / 2) * 1000),
                                         round((cx + w / 2) * 1000), round((cy + h / 2) * 1000)]}
            for c, cx, cy, w, h in rows]


def load_eval_set(data: str) -> list[tuple[Path, list[dict]]]:
    """[(사진 경로, 정답 박스 목록)]"""
    names = class_names()
    if data == "app":
        return [(SAMPLES_DIR / "test" / "images" / f"{s['id']}.jpg", s["gt"]) for s in load_samples("test")]
    base = OUT if data in ("test", "val") else Path(data)
    img_dir = base / "images" / data if data in ("test", "val") else base / "images"
    lbl_dir = base / "labels" / data if data in ("test", "val") else base / "labels"
    return [(p, yolo_rows_to_boxes(read_labels(lbl_dir / f"{p.stem}.txt"), names))
            for p in sorted(img_dir.glob("*")) if p.suffix.lower() in IMAGE_EXTS]


def make_detector(spec: str):
    if spec.startswith("yolo:"):
        return YoloDetector(spec.split(":", 1)[1])
    return {"claude": ClaudeDetector, "demo": DemoDetector}[spec]()


def predict_all(detect, items, workers: int) -> list[list[dict]]:
    def one(item):
        path = item[0]
        media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return detect(path.read_bytes(), media)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, items))


def kcal_per_piece() -> dict[str, float]:
    return {c: nutrition_for(c, 1)["kcal"] for c in class_names()}


def run_evaluation(detector, items, workers: int = 4) -> dict:
    preds = predict_all(detector.detect, items, workers)
    samples = [{"gt": gt, "pred": pred} for (_, gt), pred in zip(items, preds)]
    return evaluate(samples, class_names(), kcal_per_piece())


def save_report(name: str, detector: str, data: str, metrics: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    path = REPORTS_DIR / f"{now:%Y%m%d_%H%M%S}_{name}.json"
    path.write_text(json.dumps({"name": name, "detector": detector, "dataset": data,
                                "created_at": now.isoformat(), "metrics": metrics}, ensure_ascii=False, indent=2))
    return path


def print_summary(label: str, m: dict) -> None:
    o = m["overall"]
    fmt = lambda v, pct=True: "–" if v is None else (f"{v * 100:.1f}%" if pct else f"{v:.1f}%")  # noqa: E731
    print(f"\n[{label}] 사진 {m['n_images']}장")
    print(f"  개수 완전 일치 {fmt(o['count_exact_rate'])} · 칼로리 오차 {fmt(o['kcal_error_pct'], False)}"
          f" · mAP50 {fmt(o['map50'])} · 정밀도 {fmt(o['precision'])} · 재현율 {fmt(o['recall'])}")
    for c, r in m["per_class"].items():
        if r["gt"]:
            print(f"    {c:<10} 정답 {r['gt']:>4} · 재현율 {fmt(r['recall'])} · 정밀도 {fmt(r['precision'])}"
                  f" · 사진당 개수 오차 {r['count_mae']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="app")
    parser.add_argument("--detector", nargs="+", default=["yolo:" + str(ROOT.parent / "backend" / "models" / "nuts.pt")])
    parser.add_argument("--name", help="리포트 이름 (기본: 데이터_인식기)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    items = load_eval_set(args.data)
    if not items:
        raise SystemExit(f"'{args.data}' 평가 데이터가 없습니다. 앱의 /label 화면에서 테스트 사진을 먼저 저장하세요.")
    for spec in args.detector:
        detector = make_detector(spec)
        metrics = run_evaluation(detector, items, args.workers)
        label = Path(spec.split(":", 1)[1]).stem if spec.startswith("yolo:") else spec
        print_summary(label, metrics)
        path = save_report(args.name or f"{args.data}_{label}", spec, args.data, metrics)
        print(f"  → {path}")


if __name__ == "__main__":
    main()
