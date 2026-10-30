import os

os.environ["NUT_DETECTOR"] = "demo"

from fastapi.testclient import TestClient  # noqa: E402

from app.detector import _clean, _detection_schema  # noqa: E402
from app.main import app  # noqa: E402
from app.nutrition import load_nuts, nutrition_for  # noqa: E402

client = TestClient(app)


def test_nut_db_is_consistent():
    for nut in load_nuts().values():
        assert nut["piece_weight_g"] > 0
        assert set(nut["highlight"]) == {"senior", "athlete", "general"}
        assert nut["benefits"]


def test_nutrition_scales_with_count():
    n = nutrition_for("almond", 10)  # 1.2g * 10 = 12g
    assert n["weight_g"] == 12.0
    assert n["kcal"] == round(579 * 0.12, 1)


def test_analyze_groups_detections_by_type():
    res = client.post("/api/analyze", files={"image": ("a.jpg", b"fake", "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    counts = {item["nut_id"]: item["count"] for item in body["items"]}
    assert counts == {"almond": 3, "walnut": 2, "cashew": 1, "pistachio": 1, "raisin": 1}
    assert body["total"]["kcal"] == round(sum(i["nutrition"]["kcal"] for i in body["items"]), 1)


def test_analyze_rejects_non_image():
    res = client.post("/api/analyze", files={"image": ("a.txt", b"hi", "text/plain")})
    assert res.status_code == 415


def test_clean_clamps_and_orders_box():
    d = _clean({"nut_id": "almond", "box": [1200, 50, -10, 30], "confidence": 1.4})
    assert d["box"] == [0, 30, 1000, 50]
    assert d["confidence"] == 1.0


def test_supported_classes():
    assert list(load_nuts()) == ["almond", "peanut", "pistachio", "cashew", "hazelnut", "walnut", "raisin"]


def test_schema_lists_all_nuts():
    labels = _detection_schema()["properties"]["detections"]["items"]["properties"]["nut_id"]["enum"]
    assert set(load_nuts()) <= set(labels)


def test_frontend_served():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
