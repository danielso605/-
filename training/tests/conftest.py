import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TRAINING.parent / "backend"))


def blob_image(path: Path, boxes_px, size=(400, 300), bg=(235, 235, 235)):
    """단색 배경에 갈색 타원(견과류 대용)을 그린 테스트용 사진."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in boxes_px:
        draw.ellipse((x1, y1, x2, y2), fill=(150, 95, 50))
    im.save(path)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """build_dataset/augment/synthesize/check 가 쓰는 출력 위치를 임시 폴더로 바꾼다."""
    import augment
    import autolabel
    import build_dataset
    import check_dataset
    import synthesize

    out = tmp_path / "datasets" / "nuts"
    monkeypatch.setattr(build_dataset, "OUT", out)
    monkeypatch.setattr(build_dataset, "REJECTS", tmp_path / "rejects.txt")
    monkeypatch.setattr(build_dataset, "APP_TEST_DIR", tmp_path / "app_test")
    monkeypatch.setattr(check_dataset, "REJECTS", tmp_path / "rejects.txt")
    monkeypatch.setattr(check_dataset, "OUT", out)
    monkeypatch.setattr(autolabel, "LABEL_DIR", tmp_path / "autolabels")
    monkeypatch.setattr(build_dataset, "LABEL_DIR", tmp_path / "autolabels")
    for mod in (augment, synthesize):
        monkeypatch.setattr(mod, "TRAIN_IMAGES", out / "images" / "train")
        monkeypatch.setattr(mod, "TRAIN_LABELS", out / "labels" / "train")
    return out
