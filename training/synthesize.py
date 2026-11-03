"""섞인 견과류 사진 합성기 (copy-paste).

공개 데이터는 대부분 '한 종류만, 깨끗한 배경' 사진이라, 실제 앱에서 찍는 '여러 종류가 섞이고
붙어 있는 사진'과 차이가 크다. train 분할의 낱알을 오려내 배경 위에 무작위로 섞어 붙여
박스 라벨이 자동으로 붙은 합성 사진을 만든다.

- 오려내기: train 분할 박스만 사용 (val/test 오염 없음). 박스 가장자리가 균일한 배경이면
  배경색과 다른 부분만 마스크로, 아니면 부드러운 타원 마스크로 오려낸다.
- 배경: --backgrounds 폴더(접시·손바닥·식탁 등 견과류 없는 사진 권장)가 없으면 단색+질감 배경을 만든다.
- 배치: 실제 크기 비율을 대략 맞추고, 앞에 붙인 낱알이 40% 넘게 가려지지 않게 한다.
- 부족한 클래스가 더 자주 뽑히도록 가중치를 준다.

    python synthesize.py --count 3000 --backgrounds ./raw/backgrounds
결과: datasets/nuts/images/train/syn_*.jpg (+ labels). augment.py 는 이것도 원본처럼 센다.
"""

import argparse
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from build_dataset import OUT
from common import class_names, list_images, read_labels, write_labels

TRAIN_IMAGES = OUT / "images" / "train"
TRAIN_LABELS = OUT / "labels" / "train"
PREFIX = "syn_"
MAX_OCCLUSION = 0.4
# 대략적인 실제 길이(cm). 한 사진 안에서 크기 비율을 현실적으로 맞추는 데 쓴다.
REAL_SIZE_CM = {"almond": 2.2, "peanut": 1.4, "pistachio": 2.0, "cashew": 2.3, "hazelnut": 1.6, "walnut": 3.0, "raisin": 1.1}


def cut_out(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image | None:
    """박스 영역을 RGBA 로 오려낸다."""
    W, H = image.size
    cx, cy, w, h = box
    x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
    x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None
    crop = image.crop((x1, y1, x2, y2)).convert("RGB")
    arr = np.asarray(crop).astype(np.float32)

    ellipse = Image.new("L", crop.size, 0)
    ImageDraw.Draw(ellipse).ellipse([1, 1, crop.width - 2, crop.height - 2], fill=255)
    mask = np.asarray(ellipse.filter(ImageFilter.GaussianBlur(max(1, min(crop.size) // 30)))).astype(np.float32)

    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.std(axis=0).mean() < 18:  # 테두리가 균일 = 깨끗한 배경 → 배경과 다른 부분만
        dist = np.linalg.norm(arr - np.median(border, axis=0), axis=2)
        fg = np.clip((dist - 20) * 8, 0, 255)
        if (fg > 128).mean() > 0.15:
            mask = np.minimum(mask, fg)
    rgba = crop.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)))
    return rgba


def make_background(rng: random.Random, size: int, backgrounds: list[Path]) -> Image.Image:
    if backgrounds:
        bg = Image.open(rng.choice(backgrounds)).convert("RGB")
        scale = size / min(bg.size)
        bg = bg.resize((int(bg.width * scale) + 1, int(bg.height * scale) + 1))
        x, y = rng.randint(0, bg.width - size), rng.randint(0, bg.height - size)
        return bg.crop((x, y, x + size, y + size))
    base = np.array([rng.randint(150, 245), rng.randint(140, 235), rng.randint(120, 225)], dtype=np.float32)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(3, 14), (size // 8, size // 8, 1))
    noise = np.asarray(Image.fromarray(np.clip(noise + 128, 0, 255).astype(np.uint8)[:, :, 0])
                       .resize((size, size), Image.BICUBIC)).astype(np.float32)[:, :, None] - 128
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def visible_fraction(box, later_boxes) -> float:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    covered = 0.0
    for a1, b1, a2, b2 in later_boxes:
        covered += max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    return max(0.0, 1 - covered / area)


def compose(job: tuple[int, int, list[tuple[str, int, tuple]], list[str], int, list[str]]) -> int:
    index, seed, pool, backgrounds, size, names = job
    rng = random.Random(seed)
    canvas = make_background(rng, size, [Path(b) for b in backgrounds])
    px_per_cm = size / rng.uniform(12, 22)  # 사진 한 변이 12~22cm 정도인 접사
    placed: list[tuple[int, list[float]]] = []

    for path, cls, box in pool:
        image = Image.open(path)
        piece = cut_out(image, box)
        if piece is None:
            continue
        piece = piece.rotate(rng.uniform(0, 360), expand=True, resample=Image.BICUBIC)
        bbox = piece.getchannel("A").point(lambda a: 255 if a > 64 else 0).getbbox()
        if not bbox:
            continue
        piece = piece.crop(bbox)
        target = REAL_SIZE_CM[names[cls]] * px_per_cm * rng.uniform(0.85, 1.15)
        scale = target / max(piece.size)
        piece = piece.resize((max(4, int(piece.width * scale)), max(4, int(piece.height * scale))), Image.BICUBIC)
        rgb = ImageEnhance.Brightness(piece.convert("RGB")).enhance(rng.uniform(0.85, 1.15))
        piece = Image.merge("RGBA", (*rgb.split(), piece.getchannel("A")))

        for _ in range(15):
            x = rng.randint(0, size - piece.width)
            y = rng.randint(0, size - piece.height)
            new = [x, y, x + piece.width, y + piece.height]
            # 새 낱알은 맨 위에 붙으므로, 기존 낱알마다 '이후에 붙은 것 + 새 것'에 가려지는 비율을 확인
            if all(visible_fraction(b, [new] + [p for _, p in placed[i + 1:]]) >= 1 - MAX_OCCLUSION
                   for i, (_, b) in enumerate(placed)):
                shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
                shadow.putalpha(piece.getchannel("A").point(lambda a: a * 0.35).filter(ImageFilter.GaussianBlur(3)))
                canvas.paste(shadow, (x + 3, y + 3), shadow)
                canvas.paste(piece, (x, y), piece)
                placed.append((cls, new))
                break

    if len(placed) < 2:
        return 0
    stem = f"{PREFIX}{index:06d}"
    canvas.save(TRAIN_IMAGES / f"{stem}.jpg", quality=90)
    write_labels(TRAIN_LABELS / f"{stem}.txt", [
        (c, (b[0] + b[2]) / 2 / size, (b[1] + b[3]) / 2 / size, (b[2] - b[0]) / size, (b[3] - b[1]) / size)
        for c, b in placed])
    return len(placed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000, help="만들 합성 사진 수")
    parser.add_argument("--backgrounds", type=Path, help="배경 사진 폴더 (견과류 없는 접시·손·식탁 사진)")
    parser.add_argument("--min-items", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=25)
    parser.add_argument("--size", type=int, default=960)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for old in [*TRAIN_IMAGES.glob(f"{PREFIX}*"), *TRAIN_LABELS.glob(f"{PREFIX}*")]:
        old.unlink()

    names = class_names()
    pieces: dict[int, list[tuple[str, tuple]]] = {c: [] for c in range(len(names))}
    for image in list_images(TRAIN_IMAGES):
        if image.name.startswith(("aug_", PREFIX)):
            continue
        for c, *box in read_labels(TRAIN_LABELS / f"{image.stem}.txt"):
            pieces[c].append((str(image), tuple(box)))
    available = [c for c in pieces if pieces[c]]
    if not available:
        raise SystemExit("train 분할에 박스가 없습니다. build_dataset.py 를 먼저 실행하세요.")
    print("오려낼 수 있는 낱알:", {names[c]: len(pieces[c]) for c in pieces})

    # 낱알이 적은 클래스일수록 자주 뽑는다 (1/sqrt 가중치)
    weights = [1 / np.sqrt(len(pieces[c])) for c in available]
    backgrounds = [str(p) for p in list_images(args.backgrounds)] if args.backgrounds else []
    rng = random.Random(args.seed)
    jobs = []
    for i in range(args.count):
        kinds = rng.choices(available, weights=weights, k=rng.randint(1, min(4, len(available))))
        n = rng.randint(args.min_items, args.max_items)
        pool = []
        for _ in range(n):
            c = rng.choice(kinds)
            path, box = rng.choice(pieces[c])
            pool.append((path, c, box))
        jobs.append((i, args.seed * 1_000_003 + i, pool, backgrounds, args.size, names))

    with ProcessPoolExecutor(args.workers) as ex:
        placed = list(ex.map(compose, jobs, chunksize=8))
    made = sum(1 for p in placed if p)
    counts = Counter()
    for label in TRAIN_LABELS.glob(f"{PREFIX}*.txt"):
        counts.update(names[r[0]] for r in read_labels(label))
    print(f"합성 사진 {made}장 생성 (낱알 {sum(placed)}개)", dict(counts))


if __name__ == "__main__":
    main()
