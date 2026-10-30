"""견과류 영양 정보 DB 로딩 및 칼로리 계산."""

import json
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "nuts.json"

# 하루 권장 견과류 섭취량(한 줌). 대한영양사협회·AHA 권고 기준 약 25~30g.
DAILY_RECOMMENDED_G = 30


@lru_cache
def load_nuts() -> dict[str, dict]:
    with DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return {nut["id"]: nut for nut in data["nuts"]}


def nutrition_for(nut_id: str, count: int) -> dict:
    """개수로부터 추정 무게와 영양소를 계산한다."""
    nut = load_nuts()[nut_id]
    weight_g = round(nut["piece_weight_g"] * count, 1)
    ratio = weight_g / 100
    return {
        "weight_g": weight_g,
        **{key: round(value * ratio, 1) for key, value in nut["per_100g"].items()},
    }


def summarize(items: list[dict]) -> dict:
    """종류별 결과 목록을 합산한다."""
    total = {"weight_g": 0.0, "kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carbs_g": 0.0, "fiber_g": 0.0}
    for item in items:
        for key in total:
            total[key] += item["nutrition"][key]
    total = {key: round(value, 1) for key, value in total.items()}
    total["daily_ratio"] = round(total["weight_g"] / DAILY_RECOMMENDED_G, 2)
    return total
