"""train 분할만 데이터 증강해서 클래스마다 사진 수를 목표 범위(기본 5,000~10,000장)로 맞춘다.

- 증강: 밝기·대비·감마, 회전(90° 단위 + ±30°), 좌우/상하 반전, 약간의 크기·색조 변화.
  박스는 이미지와 함께 변환되며, 회전으로 잘려 40% 미만만 보이는 박스는 버린다.
- 클래스별 '사진 수' = 그 견과류가 한 개 이상 들어 있는 사진 수 (원본 + 증강본).
  가장 부족한 클래스부터 그 클래스가 들어 있는 사진을 골라 증강본을 만든다.
  이미 상한(max)에 도달한 클래스가 함께 들어 있는 사진은 가능하면 피한다.
- val/test 는 건드리지 않는다 (실제 성능을 정직하게 재기 위해).

    python augment.py --min 5000 --max 10000 --workers 8
"""

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import albumentations as A
import cv2

from common import class_names, list_images, read_labels, write_labels
from build_dataset import OUT

TRAIN_IMAGES = OUT / "images" / "train"
TRAIN_LABELS = OUT / "labels" / "train"
PREFIX = "aug_"
MAX_SIDE = 1024
RETRIES = 5


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=MAX_SIDE),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.25, p=0.9),
            A.RandomGamma(gamma_limit=(70, 140), p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-30, 30), scale=(0.85, 1.15), rotate_method="ellipse",
                     border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["classes"], min_visibility=0.4, clip=True),
    )


def label_path(image: Path) -> Path:
    return TRAIN_LABELS / f"{image.stem}.txt"


def plan(min_count: int, max_count: int, seed: int) -> tuple[list[tuple[Path, int]], Counter]:
    """어떤 원본을 몇 번째 증강본으로 만들지 미리 정한다 (박스가 잘려 나가는 경우는 근사)."""
    rng = random.Random(seed)
    names = class_names()
    originals = [p for p in list_images(TRAIN_IMAGES) if not p.name.startswith(PREFIX)]
    classes_of = {p: {r[0] for r in read_labels(label_path(p))} for p in originals}
    counts = Counter(c for cs in classes_of.values() for c in cs)

    by_class = {c: [p for p, cs in classes_of.items() if c in cs] for c in range(len(names))}
    jobs: list[tuple[Path, int]] = []
    copies = Counter()
    while True:
        needy = [c for c in range(len(names)) if by_class[c] and counts[c] < min_count]
        if not needy:
            break
        c = min(needy, key=lambda k: counts[k])
        pool = by_class[c]
        preferred = [p for p in pool if all(counts[o] < max_count for o in classes_of[p] if o != c)]
        image = rng.choice(preferred or pool)
        copies[image] += 1
        jobs.append((image, copies[image]))
        for o in classes_of[image]:
            counts[o] += 1
    return jobs, counts


def augment_one(job: tuple[Path, int, int]) -> bool:
    image_path, index, seed = job
    random.seed(seed)
    transform = build_transform()
    transform.set_random_seed(seed)
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    rows = read_labels(label_path(image_path))
    boxes = [r[1:] for r in rows]
    classes = [r[0] for r in rows]
    for _ in range(RETRIES):
        out = transform(image=image, bboxes=boxes, classes=classes)
        if len(out["bboxes"]):
            stem = f"{PREFIX}{index:03d}_{image_path.stem}"
            cv2.imwrite(str(TRAIN_IMAGES / f"{stem}.jpg"), out["image"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            write_labels(TRAIN_LABELS / f"{stem}.txt",
                         [(int(c), *map(float, b)) for c, b in zip(out["classes"], out["bboxes"])])
            return True
    return False


def recount() -> Counter:
    counts = Counter()
    for label in TRAIN_LABELS.glob("*.txt"):
        for c in {r[0] for r in read_labels(label)}:
            counts[c] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min", type=int, default=5000, help="클래스별 최소 사진 수")
    parser.add_argument("--max", type=int, default=10000, help="클래스별 상한 (넘치지 않도록 노력)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for old in [*TRAIN_IMAGES.glob(f"{PREFIX}*"), *TRAIN_LABELS.glob(f"{PREFIX}*")]:
        old.unlink()

    names = class_names()
    before = recount()
    jobs, _ = plan(args.min, args.max, args.seed)
    print(f"증강본 {len(jobs)}장 생성 시작…")
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(augment_one, [(p, i, args.seed * 1_000_003 + n) for n, (p, i) in enumerate(jobs)],
                                chunksize=16))
    print(f"생성 성공 {sum(results)}장, 실패 {len(results) - sum(results)}장")

    after = recount()
    print(f"\n{'클래스':<10}{'원본':>8}{'증강 후':>10}")
    report = {}
    for c, n in enumerate(names):
        flag = ""
        if after[c] < args.min:
            flag = "  ⚠️ 하한 미달"
        elif after[c] > args.max:
            flag = "  ⚠️ 상한 초과 (부족한 클래스와 같은 사진에 섞여 있어 함께 늘어남)"
        if before[c] == 0:
            flag = "  ⚠️ 원본 사진 없음 — 증강 불가"
        print(f"{n:<10}{before[c]:>8}{after[c]:>10}{flag}")
        report[n] = {"original": before[c], "after_augment": after[c]}
    (OUT / "augment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
