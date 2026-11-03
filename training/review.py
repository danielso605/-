"""라벨 검수용 모아보기(contact sheet). 박스를 그린 사진을 격자로 모아 한눈에 훑어본다.

    python review.py                       # 종류별로 최대 200장씩, review/ 에 jpg + index.html
    python review.py --source allernuts    # 자동 라벨링한 출처만
    python review.py --split val --per-class 50

틀린 사진을 발견하면 칸 위에 적힌 id(12자리)를 rejects.txt 에 한 줄씩 적고
build_dataset.py 를 다시 돌리면 그 사진이 빠진다. (이유는 id 뒤에 공백 후 자유롭게 메모)
"""

import argparse
import html
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from build_dataset import OUT, SPLITS
from common import ROOT, class_names, read_labels

REVIEW_DIR = ROOT / "review"
COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#f4511e"]
TILE = 256
COLS, ROWS = 6, 4


def font(size: int):
    for name in ("NotoSansCJK-Regular.ttc", "NanumGothic.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tile(image_path: Path, label_path: Path, names: list[str]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((TILE, TILE - 18))
    canvas = Image.new("RGB", (TILE, TILE), "white")
    ox, oy = (TILE - img.width) // 2, 18 + (TILE - 18 - img.height) // 2
    canvas.paste(img, (ox, oy))
    draw = ImageDraw.Draw(canvas)
    for c, cx, cy, w, h in read_labels(label_path):
        x1, y1 = ox + (cx - w / 2) * img.width, oy + (cy - h / 2) * img.height
        x2, y2 = ox + (cx + w / 2) * img.width, oy + (cy + h / 2) * img.height
        draw.rectangle([x1, y1, x2, y2], outline=COLORS[c % len(COLORS)], width=2)
    sample_id = image_path.stem.split("__")[-1][:12]
    n = len(read_labels(label_path))
    draw.text((4, 2), f"{sample_id}  ({n})", fill="black", font=font(12))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, help="한 분할만 (기본: 전체)")
    parser.add_argument("--source", help="이 출처 이름으로 시작하는 사진만")
    parser.add_argument("--per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = class_names()
    by_class: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    for split in [args.split] if args.split else SPLITS:
        for label in sorted((OUT / "labels" / split).glob("*.txt")):
            if label.stem.startswith(("aug_", "syn_")):
                continue  # 증강·합성본은 원본에서 자동으로 만들어지므로 원본만 본다
            if args.source and not label.stem.startswith(args.source):
                continue
            images = list((OUT / "images" / split).glob(f"{label.stem}.*"))
            if images:
                for c in {r[0] for r in read_labels(label)}:
                    by_class[c].append((images[0], label))

    if REVIEW_DIR.exists():
        shutil.rmtree(REVIEW_DIR)
    REVIEW_DIR.mkdir()
    rng = random.Random(args.seed)
    links = []
    for c in sorted(by_class):
        items = by_class[c]
        rng.shuffle(items)
        items = items[: args.per_class]
        per_sheet = COLS * ROWS
        for page in range(0, len(items), per_sheet):
            sheet = Image.new("RGB", (COLS * TILE, ROWS * TILE), "#ddd")
            for i, (img, lbl) in enumerate(items[page: page + per_sheet]):
                sheet.paste(tile(img, lbl, names), ((i % COLS) * TILE, (i // COLS) * TILE))
            name = f"{names[c]}_{page // per_sheet + 1:02d}.jpg"
            sheet.save(REVIEW_DIR / name, quality=88)
            links.append((names[c], name))
        print(f"{names[c]}: {len(items)}장")

    body = "".join(f'<h2>{html.escape(n)}</h2><img src="{html.escape(f)}" style="max-width:100%">' for n, f in links)
    (REVIEW_DIR / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><title>라벨 검수</title>"
        f"<p>틀린 사진의 id 를 training/rejects.txt 에 적으세요. 괄호 안은 박스 수.</p>{body}", encoding="utf-8")
    print(f"→ {REVIEW_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
