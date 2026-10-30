"""데이터셋 품질 검사. build_dataset.py 다음, augment.py 전에 돌린다.

검사 항목: 열리지 않는 사진 · 라벨 없는 사진 / 사진 없는 라벨 · 잘못된 줄·클래스 번호 ·
이미지 밖으로 나간 박스 · 너무 작은/큰 박스 · 극단적인 가로세로 비 · 같은 물체에 겹쳐 그린 박스

    python check_dataset.py                 # 보고서만
    python check_dataset.py --write-rejects # 심각한 문제(사진 깨짐·잘못된 라벨) 사진을 rejects.txt 에 추가
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from build_dataset import OUT, REJECTS, SPLITS
from common import IMAGE_EXTS, class_names

MIN_AREA = 0.0004   # 이미지의 0.04% (≈ 한 변 2%) 보다 작으면 의심
MAX_AREA = 0.9
MAX_ASPECT = 6.0
DUP_IOU = 0.9

SEVERE = {"broken_image", "bad_line", "bad_class", "out_of_bounds"}


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def check_label_file(path: Path, n_classes: int) -> list[str]:
    issues = []
    boxes = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            if len(parts) != 5:
                raise ValueError
        except ValueError:
            issues.append("bad_line")
            continue
        if not 0 <= c < n_classes:
            issues.append("bad_class")
        if w <= 0 or h <= 0 or cx - w / 2 < -0.01 or cy - h / 2 < -0.01 or cx + w / 2 > 1.01 or cy + h / 2 > 1.01:
            issues.append("out_of_bounds")
        area = w * h
        if 0 < area < MIN_AREA:
            issues.append("tiny_box")
        if area > MAX_AREA:
            issues.append("huge_box")
        if w > 0 and h > 0 and max(w / h, h / w) > MAX_ASPECT:
            issues.append("extreme_aspect")
        for c2, b2 in boxes:
            if c2 == c and _iou(b2, (cx, cy, w, h)) > DUP_IOU:
                issues.append("duplicate_box")
        boxes.append((c, (cx, cy, w, h)))
    if not boxes:
        issues.append("empty_label")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-rejects", action="store_true")
    parser.add_argument("--dataset", type=Path, default=OUT)
    args = parser.parse_args()

    n_classes = len(class_names())
    counts = Counter()
    examples = defaultdict(list)
    severe_ids = set()

    for split in SPLITS:
        img_dir, lbl_dir = args.dataset / "images" / split, args.dataset / "labels" / split
        images = {p.stem: p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS} if img_dir.exists() else {}
        labels = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
        for stem in images.keys() - labels.keys():
            counts["no_label"] += 1
            examples["no_label"].append(f"{split}/{stem}")
        for stem in labels.keys() - images.keys():
            counts["no_image"] += 1
            examples["no_image"].append(f"{split}/{stem}")
        for stem in images.keys() & labels.keys():
            issues = set(check_label_file(labels[stem], n_classes))
            try:
                with Image.open(images[stem]) as im:
                    im.verify()
            except Exception:
                issues.add("broken_image")
            for issue in issues:
                counts[issue] += 1
                examples[issue].append(f"{split}/{stem}")
            if issues & SEVERE:
                severe_ids.add(stem.split("__")[-1][:12])

    print("검사 결과" + ("" if counts else ": 문제 없음 ✅"))
    labels_ko = {
        "broken_image": "열리지 않는 사진", "no_label": "라벨 없는 사진", "no_image": "사진 없는 라벨",
        "bad_line": "잘못된 라벨 줄", "bad_class": "없는 클래스 번호", "out_of_bounds": "이미지 밖 박스",
        "tiny_box": "너무 작은 박스", "huge_box": "너무 큰 박스", "extreme_aspect": "극단적 가로세로 비",
        "duplicate_box": "겹쳐 그린 중복 박스", "empty_label": "박스 없는 라벨",
    }
    for issue, n in counts.most_common():
        mark = "❗" if issue in SEVERE else "⚠️ "
        print(f"  {mark} {labels_ko.get(issue, issue)}: {n}건  예) {', '.join(examples[issue][:3])}")

    (args.dataset / "check_report.json").parent.mkdir(parents=True, exist_ok=True)
    (args.dataset / "check_report.json").write_text(json.dumps(
        {"counts": counts, "examples": {k: v[:50] for k, v in examples.items()}}, ensure_ascii=False, indent=2))

    if args.write_rejects and severe_ids:
        existing = REJECTS.read_text() if REJECTS.exists() else ""
        new = sorted(i for i in severe_ids if i not in existing)
        with REJECTS.open("a") as f:
            f.writelines(f"{i} check_dataset 자동 제외\n" for i in new)
        print(f"rejects.txt 에 {len(new)}개 추가 → build_dataset.py 를 다시 실행하세요.")


if __name__ == "__main__":
    main()
