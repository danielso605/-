import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

from conftest import blob_image


def run_main(module, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", [module.__name__, *argv])
    module.main()


def sources_file(tmp_path, sources) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sources": sources}))
    return path


# ---------- 1. COCO / VOC 형식 변환 ----------

def make_coco(base: Path):
    images, anns = [], []
    for i in range(6):
        boxes = [(20 + 60 * j, 40, 70 + 60 * j, 100) for j in range(i % 3 + 1)]
        blob_image(base / "imgs" / f"c{i}.jpg", boxes, bg=(230, 230, 200 + i))
        images.append({"id": i, "file_name": f"imgs/c{i}.jpg", "width": 400, "height": 300})
        cat = 3 if i == 5 else 1  # 마지막 사진엔 대상 외 클래스(fig)
        anns += [{"image_id": i, "category_id": cat, "bbox": [x1, y1, x2 - x1, y2 - y1]} for x1, y1, x2, y2 in boxes]
    (base / "ann.json").write_text(json.dumps({
        "images": images, "annotations": anns,
        "categories": [{"id": 1, "name": "Almond"}, {"id": 3, "name": "Fig"}]}))


def make_voc(base: Path):
    for i in range(4):
        blob_image(base / f"v{i}.jpg", [(100, 100, 160, 150)], bg=(200 + i * 10, 220, 230))
        (base / f"v{i}.xml").write_text(
            f"<annotation><filename>v{i}.jpg</filename><size><width>400</width><height>300</height></size>"
            f"<object><name>Walnut</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>160</xmax><ymax>150</ymax>"
            f"</bndbox></object></annotation>")


def test_build_dataset_reads_coco_and_voc_and_rejects(tmp_path, dataset, monkeypatch):
    import build_dataset

    make_coco(tmp_path / "coco")
    make_voc(tmp_path / "voc")
    src = sources_file(tmp_path, [
        {"name": "coco", "type": "local", "path": str(tmp_path / "coco"), "format": "coco",
         "class_map": {"Almond": "almond"}},
        {"name": "voc", "type": "local", "path": str(tmp_path / "voc"), "format": "voc",
         "class_map": {"Walnut": "walnut"}},
    ])
    run_main(build_dataset, monkeypatch, "--sources", str(src), "--min-images", "1")
    stats = json.loads((dataset / "stats.json").read_text())
    total = lambda c: sum(stats["images_per_class"][s].get(c, 0) for s in build_dataset.SPLITS)  # noqa: E731
    assert total("almond") == 5        # fig 섞인 사진 1장 제외
    assert total("walnut") == 4

    label = next((dataset / "labels").rglob("coco__*.txt"))
    c, cx, cy, w, h = map(float, label.read_text().split("\n")[0].split())
    assert c == 0 and abs(cx - 45 / 400) < 1e-3 and abs(w - 50 / 400) < 1e-3

    # 검수에서 뺀 사진은 다시 빌드하면 빠진다
    reject_id = label.stem.split("__")[1]
    build_dataset.REJECTS.write_text(f"{reject_id} 박스 틀림\n")
    run_main(build_dataset, monkeypatch, "--sources", str(src), "--min-images", "1")
    stats = json.loads((dataset / "stats.json").read_text())
    assert total("almond") == 4


def test_build_dataset_refuses_app_test_set(tmp_path, dataset, monkeypatch):
    import build_dataset

    src = sources_file(tmp_path, [{"name": "oops", "type": "local", "path": str(tmp_path / "app_test"),
                                   "format": "yolo", "class_map": {}}])
    with pytest.raises(SystemExit, match="samples/test"):
        run_main(build_dataset, monkeypatch, "--sources", str(src))


# ---------- 2·3. 검수 모아보기, 품질 검사 ----------

def write_split(dataset: Path, split: str, stem: str, labels: str, boxes_px=((40, 40, 120, 110),)):
    blob_image(dataset / "images" / split / f"{stem}.jpg", boxes_px)
    (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
    (dataset / "labels" / split / f"{stem}.txt").write_text(labels)


def test_check_dataset_finds_problems(tmp_path, dataset, monkeypatch):
    import check_dataset

    write_split(dataset, "train", "ok__aaaaaaaaaaaa", "0 0.2 0.25 0.2 0.23\n")
    write_split(dataset, "train", "bad__bbbbbbbbbbbb", "9 0.5 0.5 0.1 0.1\n0 0.95 0.5 0.3 0.1\n")
    write_split(dataset, "val", "dup__cccccccccccc", "1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.21\n0 0.1 0.1 0.001 0.001\n")
    (dataset / "images" / "val" / "broken__dddddddddddd.jpg").write_bytes(b"not an image")
    (dataset / "labels" / "val" / "broken__dddddddddddd.txt").write_text("0 0.5 0.5 0.1 0.1\n")

    run_main(check_dataset, monkeypatch, "--write-rejects")
    counts = json.loads((dataset / "check_report.json").read_text())["counts"]
    assert counts["bad_class"] == 1 and counts["out_of_bounds"] == 1
    assert counts["duplicate_box"] == 1 and counts["tiny_box"] == 1 and counts["broken_image"] == 1
    rejects = check_dataset.REJECTS.read_text()
    assert "bbbbbbbbbbbb" in rejects and "dddddddddddd" in rejects and "aaaaaaaaaaaa" not in rejects


def test_review_contact_sheets(tmp_path, dataset, monkeypatch):
    import review

    monkeypatch.setattr(review, "OUT", dataset)
    monkeypatch.setattr(review, "REVIEW_DIR", tmp_path / "review")
    for i in range(30):
        write_split(dataset, "train", f"src__{i:012d}", "0 0.2 0.25 0.2 0.23\n")
    write_split(dataset, "train", "aug_001_src__x", "0 0.2 0.25 0.2 0.23\n")
    run_main(review, monkeypatch)
    sheets = sorted((tmp_path / "review").glob("almond_*.jpg"))
    assert len(sheets) == 2  # 24장 + 6장, 증강본 제외
    assert Image.open(sheets[0]).size == (6 * review.TILE, 4 * review.TILE)


# ---------- 4. 섞인 사진 합성 ----------

def test_synthesize_mixes_classes_with_valid_boxes(tmp_path, dataset, monkeypatch):
    import synthesize
    from common import read_labels

    for i in range(4):
        write_split(dataset, "train", f"a__{i}", "0 0.2 0.25 0.2 0.23\n")                       # almond
        write_split(dataset, "train", f"r__{i}", "6 0.5 0.5 0.1 0.1\n", ((180, 135, 220, 165),))  # raisin
    run_main(synthesize, monkeypatch, "--count", "6", "--workers", "1", "--size", "320",
             "--min-items", "4", "--max-items", "8")
    labels = sorted((dataset / "labels" / "train").glob("syn_*.txt"))
    assert len(labels) >= 4
    classes = set()
    for label in labels:
        rows = read_labels(label)
        assert len(rows) >= 2
        for c, cx, cy, w, h in rows:
            classes.add(c)
            assert 0 <= cx - w / 2 and cx + w / 2 <= 1 and 0 <= cy - h / 2 and cy + h / 2 <= 1
    assert classes == {0, 6}


# ---------- 5. 평가 ----------

def test_evaluate_folder_with_demo_detector(tmp_path, dataset):
    from evaluate import load_eval_set, run_evaluation
    from app.detector import DemoDetector

    folder = tmp_path / "eval"
    blob_image(folder / "images" / "x.jpg", [])
    (folder / "labels").mkdir()
    # 데모 결과와 같은 아몬드 3개 + 데모에 없는 땅콩 1개
    (folder / "labels" / "x.txt").write_text(
        "0 0.155 0.19 0.15 0.14\n0 0.33 0.18 0.14 0.14\n0 0.5 0.2 0.14 0.14\n1 0.5 0.9 0.05 0.05\n")
    items = load_eval_set(str(folder))
    m = run_evaluation(DemoDetector(), items, workers=1)
    assert m["per_class"]["almond"]["recall"] == 1.0
    assert m["per_class"]["peanut"]["recall"] == 0.0
    assert m["overall"]["count_exact_rate"] == 0.0


# ---------- 6. 기준값 튜닝 ----------

def test_tune_picks_thresholds_that_fix_counts():
    from tune import review_threshold, sweep
    from app.metrics import postprocess

    gt = [{"nut_id": "almond", "box": [0, 0, 100, 100]}, {"nut_id": "almond", "box": [200, 0, 300, 100]}]
    raw = [
        {"nut_id": "almond", "box": [0, 0, 100, 100], "confidence": 0.9},
        {"nut_id": "almond", "box": [10, 0, 110, 100], "confidence": 0.6},   # 중복 (IoU≈0.82)
        {"nut_id": "almond", "box": [200, 0, 300, 100], "confidence": 0.35},
        {"nut_id": "almond", "box": [600, 600, 700, 700], "confidence": 0.2},  # 오검출
    ]
    th, m = sweep([{"gt": gt, "pred": raw}] * 3, ["almond"], {"almond": 7.0})
    assert m["overall"]["count_exact_rate"] == 1.0
    assert 0.2 < th["conf"] <= 0.35 and th["iou"] < 0.82
    kept = [{"gt": gt, "pred": postprocess(raw, th["conf"], th["iou"])}]
    assert review_threshold(kept, th["conf"]) == th["conf"]  # 남은 박스가 모두 정답이면 확인 필요 없음
    noisy = [{"gt": gt[:1], "pred": raw[:1] + [{"nut_id": "almond", "box": [500, 0, 600, 100], "confidence": 0.45}]}]
    assert review_threshold(noisy, 0.3) == 0.5  # 0.4~0.5 구간은 틀렸으므로 0.5 미만은 확인 필요


# ---------- 10. 모델 버전 관리 ----------

def test_promote_decision_rules():
    from promote import decide

    m = lambda cnt, mp: {"count_exact_rate": cnt, "map50": mp}  # noqa: E731
    assert decide(m(0.5, 0.5), None)[0]
    assert decide(m(0.62, 0.5), m(0.60, 0.7))[0]           # 개수 정확도 상승
    assert not decide(m(0.55, 0.9), m(0.60, 0.5))[0]       # 개수 하락은 mAP 가 올라도 거부
    assert decide(m(0.605, 0.72), m(0.60, 0.70))[0]        # 비슷 + mAP 상승
    assert not decide(m(0.605, 0.705), m(0.60, 0.70))[0]   # 의미 없는 차이


def test_promote_register_and_rollback(tmp_path, monkeypatch):
    import promote

    models = tmp_path / "models"
    monkeypatch.setattr(promote, "MODEL_DIR", models)
    monkeypatch.setattr(promote, "VERSIONS_DIR", models / "versions")
    monkeypatch.setattr(promote, "REGISTRY", models / "registry.json")
    monkeypatch.setattr(promote, "ACTIVE", models / "nuts.pt")
    metrics = {"n_images": 10, "overall": {"count_exact_rate": 0.5, "map50": 0.6}}

    c1 = tmp_path / "run1" / "best.pt"
    c1.parent.mkdir()
    c1.write_bytes(b"weights-1")
    c1.with_suffix(".json").write_text('{"conf": 0.3}')
    c1.with_suffix(".onnx").write_bytes(b"onnx-1")
    c2 = tmp_path / "run2" / "best.pt"
    c2.parent.mkdir()
    c2.write_bytes(b"weights-2")  # 기준값/onnx 없음

    assert promote.register(c1, metrics, "app", "first") == "v1"
    assert promote.register(c2, metrics, "app", "better") == "v2"
    assert (models / "nuts.pt").read_bytes() == b"weights-2"
    assert not (models / "nuts.onnx").exists()  # v2 엔 휴대폰 모델이 없으니 예전 것도 지움

    promote.rollback("v1")
    assert (models / "nuts.pt").read_bytes() == b"weights-1"
    assert (models / "nuts.onnx").read_bytes() == b"onnx-1"
    reg = json.loads((models / "registry.json").read_text())
    assert [v["active"] for v in reg["versions"]] == [True, False]
