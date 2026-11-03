"""박스 라벨이 없는 사진(format: classified)에 Claude 로 YOLO 박스 라벨을 만든다.

클래스별 폴더 사진은 그 폴더의 클래스만 있다고 알려 주고 박스만 찾게 한다.
folder_map 값이 "mixed" 인 폴더는 7종 전체 중에서 종류까지 판별하게 한다.
결과는 autolabels/<출처>/<원본 상대경로>.txt 에 저장되고, 이미 만든 라벨은 건너뛴다.

    export ANTHROPIC_API_KEY=...
    python autolabel.py --sources sources.yaml --dry-run   # 장수만 세기
    python autolabel.py --sources sources.yaml --workers 4
"""

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from PIL import Image

from common import ROOT, class_names, list_images, write_labels

sys.path.insert(0, str(ROOT.parent / "backend"))
from app.detector import UNKNOWN, ClaudeDetector, DetectionError  # noqa: E402
from app.nutrition import load_nuts  # noqa: E402

LABEL_DIR = ROOT / "autolabels"
MIXED = "mixed"
MAX_SIDE = 1568


def source_dir(src: dict) -> Path:
    return (ROOT / src["path"]).resolve() if src["type"] == "local" else ROOT / "raw" / src["name"]


def find_folder(base: Path, folder_name: str) -> Path | None:
    """zip 구조가 제각각이라 이름이 일치하는 하위 폴더를 깊이 상관없이 찾는다."""
    if (base / folder_name).is_dir():
        return base / folder_name
    matches = [p for p in base.rglob(folder_name) if p.is_dir()]
    return min(matches, key=lambda p: len(p.parts)) if matches else None


def jobs_for(src: dict) -> list[tuple[Path, Path, str]]:
    base = source_dir(src)
    jobs = []
    for folder_name, target in src["folder_map"].items():
        folder = find_folder(base, folder_name)
        if folder is None:
            print(f"  ! {src['name']}: '{folder_name}' 폴더를 찾지 못함")
            continue
        for image in list_images(folder):
            label = LABEL_DIR / src["name"] / image.relative_to(base).with_suffix(".txt")
            if not label.exists():
                jobs.append((image, label, target))
    return jobs


def encode(image_path: Path) -> bytes:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def label_one(detector: ClaudeDetector, image: Path, label: Path, target: str) -> int:
    names = class_names()
    nuts = load_nuts()
    if target == MIXED:
        instruction = "Detect every nut and raisin in this photo."
    else:
        instruction = (
            f"This photo contains only {nuts[target]['name_en']} ({target}). "
            f"Put a separate box around every single one and label each as {target}."
        )
    detections = detector.detect(encode(image), "image/jpeg", instruction=instruction)

    rows = []
    for d in detections:
        nut_id = target if target != MIXED else d["nut_id"]
        if nut_id == UNKNOWN or nut_id not in names:
            continue
        x1, y1, x2, y2 = (v / 1000 for v in d["box"])
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue
        rows.append((names.index(nut_id), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    write_labels(label, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=ROOT / "sources.yaml", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="출처별 최대 장수 (시험용)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))["sources"]
    jobs = []
    for src in sources:
        if src.get("format") != "classified":
            continue
        src_jobs = jobs_for(src)[: args.limit]
        print(f"[{src['name']}] 라벨 필요: {len(src_jobs)}장")
        jobs += src_jobs
    if args.dry_run or not jobs:
        return

    detector = ClaudeDetector()
    done = failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(label_one, detector, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except (DetectionError, OSError) as e:
                failed += 1
                print(f"  ! {futures[future]}: {e}")
            if (done + failed) % 50 == 0:
                print(f"  진행: {done + failed}/{len(jobs)} (실패 {failed})")
    print(f"완료: {done}장, 실패: {failed}장")


if __name__ == "__main__":
    main()
