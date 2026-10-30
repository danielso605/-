"""모든 출처를 하나의 YOLO 데이터셋(datasets/nuts)으로 합치고 train/val/test 로 나눈다.

- 7개 대상 클래스만 남기고, 다른 클래스(마카다미아, 무화과 등)가 섞인 사진은 통째로 뺀다
  (박스를 지우기만 하면 그 물체가 '배경'으로 잘못 학습되기 때문).
- 같은 사진(바이트 동일)은 한 번만 넣는다.
- 분할은 파일 해시로 정해서 다시 돌려도 같은 사진은 같은 분할에 들어간다.
- 데이터 증강은 반드시 이 다음 단계에서 train 에만 한다 (val/test 오염 방지).

    python build_dataset.py --sources sources.yaml --min-images 2000
"""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

from autolabel import LABEL_DIR, find_folder, source_dir
from common import ROOT, class_names, list_images, read_labels, write_labels

OUT = ROOT / "datasets" / "nuts"
SPLITS = ("train", "val", "test")


def split_for(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def source_names(base: Path) -> list[str]:
    """Roboflow/Ultralytics 형식 data.yaml 의 names (list 또는 dict)."""
    for cfg in base.rglob("data.yaml"):
        names = yaml.safe_load(cfg.read_text())["names"]
        return list(names.values()) if isinstance(names, dict) else list(names)
    raise SystemExit(f"{base} 에서 data.yaml 을 찾지 못했습니다.")


def yolo_label_path(image: Path) -> Path:
    parts = list(image.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def collect_yolo(src: dict, names: list[str]):
    base = source_dir(src)
    src_names = source_names(base)
    mapping = {i: names.index(src["class_map"][n]) for i, n in enumerate(src_names) if n in src["class_map"]}
    for image in list_images(base):
        if "images" not in image.parts:
            continue
        rows = read_labels(yolo_label_path(image))
        if not rows or any(c not in mapping for c, *_ in rows):
            yield image, None
            continue
        yield image, [(mapping[c], *box) for c, *box in rows]


def collect_classified(src: dict):
    base = source_dir(src)
    for folder_name in src["folder_map"]:
        folder = find_folder(base, folder_name)
        if folder is None:
            continue
        for image in list_images(folder):
            label = LABEL_DIR / src["name"] / image.relative_to(base).with_suffix(".txt")
            yield image, read_labels(label) or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=ROOT / "sources.yaml", type=Path)
    parser.add_argument("--min-images", type=int, default=2000, help="클래스별 목표 원본 사진 수 (부족하면 경고)")
    args = parser.parse_args()

    names = class_names()
    sources = yaml.safe_load(args.sources.read_text(encoding="utf-8"))["sources"]
    if OUT.exists():
        shutil.rmtree(OUT)

    seen: set[str] = set()
    skipped = Counter()
    images_per_class = {s: Counter() for s in SPLITS}
    boxes_per_class = {s: Counter() for s in SPLITS}

    for src in sources:
        items = collect_yolo(src, names) if src["format"] == "yolo" else collect_classified(src)
        for image, rows in items:
            if rows is None:
                skipped[src["name"]] += 1
                continue
            digest = hashlib.md5(image.read_bytes()).hexdigest()
            if digest in seen:
                skipped["중복"] += 1
                continue
            seen.add(digest)

            split = split_for(digest)
            stem = f"{src['name']}__{digest[:12]}"
            dest = OUT / "images" / split / f"{stem}{image.suffix.lower()}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, dest)
            write_labels(OUT / "labels" / split / f"{stem}.txt", rows)

            for c in {r[0] for r in rows}:
                images_per_class[split][names[c]] += 1
            for r in rows:
                boxes_per_class[split][names[r[0]]] += 1

    (OUT / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": str(OUT), "train": "images/train", "val": "images/val", "test": "images/test",
             "names": dict(enumerate(names))},
            allow_unicode=True, sort_keys=False,
        )
    )

    print(f"\n{'클래스':<10}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'합계':>8}{'박스수':>9}")
    shortage = {}
    for n in names:
        counts = [images_per_class[s][n] for s in SPLITS]
        total = sum(counts)
        print(f"{n:<10}" + "".join(f"{c:>8}" for c in counts) + f"{total:>8}{sum(boxes_per_class[s][n] for s in SPLITS):>9}")
        if total < args.min_images:
            shortage[n] = args.min_images - total
    if skipped:
        print("\n제외된 사진:", dict(skipped), "(라벨 없음 / 대상 외 클래스 포함 / 중복)")
    if shortage:
        print(f"\n⚠️  목표 {args.min_images}장에 못 미치는 클래스 (부족분):", shortage)

    (OUT / "stats.json").write_text(json.dumps(
        {"images_per_class": {s: dict(c) for s, c in images_per_class.items()},
         "boxes_per_class": {s: dict(c) for s, c in boxes_per_class.items()},
         "shortage": shortage}, ensure_ascii=False, indent=2))
    print(f"\n완료 → {OUT}")


if __name__ == "__main__":
    main()
