"""모든 출처를 하나의 YOLO 데이터셋(datasets/nuts)으로 합치고 train/val/test 로 나눈다.

- 7개 대상 클래스만 남기고, 다른 클래스(마카다미아, 무화과 등)가 섞인 사진은 통째로 뺀다
  (박스를 지우기만 하면 그 물체가 '배경'으로 잘못 학습되기 때문).
- 같은 사진(바이트 동일)은 한 번만 넣는다.
- 분할은 파일 해시로 정해서 다시 돌려도 같은 사진은 같은 분할에 들어간다.
- 데이터 증강은 반드시 이 다음 단계에서 train 에만 한다 (val/test 오염 방지).
- 라벨 형식: yolo / classified(+autolabel) / coco(JSON) / voc(XML)
- rejects.txt (review.py 로 검수하며 적은 사진 id) 에 있는 사진은 뺀다.
- 앱의 정확도 측정용 사진(backend/data/samples/test)은 학습에 절대 들어가지 않게 막는다.

    python build_dataset.py --sources sources.yaml --min-images 2000
"""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml

from autolabel import LABEL_DIR, find_folder, source_dir
from common import IMAGE_EXTS, ROOT, class_names, list_images, read_labels, write_labels

OUT = ROOT / "datasets" / "nuts"
SPLITS = ("train", "val", "test")
REJECTS = ROOT / "rejects.txt"
APP_TEST_DIR = ROOT.parent / "backend" / "data" / "samples" / "test"


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


def _box_row(cls: int, x1: float, y1: float, x2: float, y2: float, w: float, h: float):
    x1, x2 = max(0.0, x1), min(w, x2)
    y1, y2 = max(0.0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (cls, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def collect_coco(src: dict, names: list[str]):
    """COCO JSON: images[{id,file_name,width,height}], annotations[{image_id,category_id,bbox:[x,y,w,h]}]."""
    base = source_dir(src)
    for ann_file in sorted(base.rglob(src.get("annotations", "*.json"))):
        coco = json.loads(ann_file.read_text(encoding="utf-8"))
        if "images" not in coco or "annotations" not in coco:
            continue
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for img in coco["images"]:
            path = ann_file.parent / img["file_name"]
            if not path.exists():
                matches = list(base.rglob(Path(img["file_name"]).name))
                if not matches:
                    continue
                path = matches[0]
            rows, ok = [], True
            for a in by_image.get(img["id"], []):
                target = src["class_map"].get(cats[a["category_id"]])
                if target is None:
                    ok = False
                    break
                x, y, w, h = a["bbox"]
                row = _box_row(names.index(target), x, y, x + w, y + h, img["width"], img["height"])
                if row:
                    rows.append(row)
            yield path, rows if ok and rows else None


def collect_voc(src: dict, names: list[str]):
    """Pascal VOC XML: 사진마다 <filename>, <size>, <object><name><bndbox> 를 가진 xml."""
    base = source_dir(src)
    for xml_file in sorted(base.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        if root.tag != "annotation":
            continue
        filename = root.findtext("filename") or ""
        path = xml_file.with_name(filename)
        if not path.exists():
            candidates = [p for p in xml_file.parent.glob(f"{xml_file.stem}.*") if p.suffix.lower() in IMAGE_EXTS]
            if not candidates and filename:
                candidates = list(base.rglob(filename))
            if not candidates:
                continue
            path = candidates[0]
        w = float(root.findtext("size/width") or 0)
        h = float(root.findtext("size/height") or 0)
        if not w or not h:
            from PIL import Image
            with Image.open(path) as im:
                w, h = im.size
        rows, ok = [], True
        for obj in root.iter("object"):
            target = src["class_map"].get(obj.findtext("name", "").strip())
            if target is None:
                ok = False
                break
            bb = obj.find("bndbox")
            coords = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            row = _box_row(names.index(target), *coords, w, h)
            if row:
                rows.append(row)
        yield path, rows if ok and rows else None


COLLECTORS = {"yolo": collect_yolo, "coco": collect_coco, "voc": collect_voc}


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

    for src in sources:
        if src["type"] == "local" and APP_TEST_DIR.resolve() in [*(ROOT / src["path"]).resolve().parents, (ROOT / src["path"]).resolve()]:
            raise SystemExit(f"[{src['name']}] 앱 정확도 측정용 사진(samples/test)은 학습에 쓸 수 없습니다. samples/feedback 을 쓰세요.")

    rejects = {line.split()[0] for line in REJECTS.read_text().splitlines() if line.strip() and not line.startswith("#")} \
        if REJECTS.exists() else set()
    # 앱 테스트 사진과 같은 사진이 학습 데이터에 섞여 들어가지 않게 해시로 막는다
    seen: set[str] = {hashlib.md5(p.read_bytes()).hexdigest() for p in list_images(APP_TEST_DIR)} \
        if APP_TEST_DIR.exists() else set()
    skipped = Counter()
    images_per_class = {s: Counter() for s in SPLITS}
    boxes_per_class = {s: Counter() for s in SPLITS}

    for src in sources:
        fmt = src["format"]
        items = collect_classified(src) if fmt == "classified" else COLLECTORS[fmt](src, names)
        for image, rows in items:
            if rows is None:
                skipped[src["name"]] += 1
                continue
            digest = hashlib.md5(image.read_bytes()).hexdigest()
            if digest in seen:
                skipped["중복"] += 1
                continue
            if digest[:12] in rejects:
                skipped["검수 제외"] += 1
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
