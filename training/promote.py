"""새로 학습한 모델을 '지금 쓰는 모델'보다 나을 때만 교체한다 (모델 버전 관리).

1. 같은 평가 데이터(기본: 앱 실사진 test, 부족하면 datasets test 분할)로 후보와 현재 모델을 평가
2. 판정: '개수 완전 일치 비율'이 min_gain 이상 오르면 교체.
         거의 같으면(차이 < min_gain) mAP50 이 min_gain 이상 오를 때만 교체. 떨어지면 거부.
3. 교체 시 backend/models/versions/vN.* 로 보관하고 nuts.pt / nuts.json / nuts.onnx 를 바꾼 뒤
   registry.json 에 기록 → 앱 /report 화면의 '모델 버전별 성능 변화' 그래프에 표시

    python promote.py --candidate runs/nuts/weights/best.pt
    python promote.py --rollback v2          # 예전 버전으로 되돌리기
    python promote.py --list
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT
from evaluate import load_eval_set, print_summary, run_evaluation, save_report

MODEL_DIR = ROOT.parent / "backend" / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
REGISTRY = MODEL_DIR / "registry.json"
ACTIVE = MODEL_DIR / "nuts.pt"
MIN_APP_IMAGES = 50
SIDE_FILES = (".json", ".onnx")  # 가중치 옆에 함께 다니는 파일: 기준값, 휴대폰용 모델


def decide(candidate: dict, current: dict | None, min_gain: float = 0.01) -> tuple[bool, str]:
    """overall 지표 두 개를 비교해 (교체 여부, 이유)."""
    if current is None:
        return True, "사용 중인 모델이 없어 첫 버전으로 등록"
    c_cnt, o_cnt = candidate["count_exact_rate"] or 0, current["count_exact_rate"] or 0
    c_map, o_map = candidate["map50"] or 0, current["map50"] or 0
    if c_cnt >= o_cnt + min_gain:
        return True, f"개수 완전 일치 {o_cnt:.1%} → {c_cnt:.1%}"
    if c_cnt <= o_cnt - min_gain:
        return False, f"개수 완전 일치가 떨어짐 {o_cnt:.1%} → {c_cnt:.1%}"
    if c_map >= o_map + min_gain:
        return True, f"개수 정확도는 비슷하고 mAP50 상승 {o_map:.1%} → {c_map:.1%}"
    return False, f"의미 있는 개선 없음 (개수 {o_cnt:.1%}→{c_cnt:.1%}, mAP50 {o_map:.1%}→{c_map:.1%})"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"versions": []}


def save_registry(reg: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def activate(version_weights: Path) -> None:
    shutil.copy2(version_weights, ACTIVE)
    for ext in SIDE_FILES:
        src, dst = version_weights.with_suffix(ext), ACTIVE.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, dst)
        elif dst.exists():
            dst.unlink()


def register(candidate: Path, metrics: dict, data: str, reason: str) -> str:
    reg = load_registry()
    version = f"v{len(reg['versions']) + 1}"
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    stored = VERSIONS_DIR / f"{version}.pt"
    shutil.copy2(candidate, stored)
    for ext in SIDE_FILES:
        if candidate.with_suffix(ext).exists():
            shutil.copy2(candidate.with_suffix(ext), stored.with_suffix(ext))
    activate(stored)
    for v in reg["versions"]:
        v["active"] = False
    reg["versions"].append({
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(candidate),
        "sha256": hashlib.sha256(stored.read_bytes()).hexdigest()[:16],
        "test_set": data,
        "n_images": metrics["n_images"],
        "metrics": metrics["overall"],
        "reason": reason,
        "active": True,
    })
    save_registry(reg)
    return version


def rollback(version: str) -> None:
    reg = load_registry()
    target = VERSIONS_DIR / f"{version}.pt"
    if not target.exists():
        raise SystemExit(f"{version} 이 없습니다. --list 로 확인하세요.")
    activate(target)
    for v in reg["versions"]:
        v["active"] = v["version"] == version
    save_registry(reg)
    print(f"{version} 으로 되돌렸습니다. 서버를 다시 시작하세요.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, help="후보 가중치 (.pt). 옆에 .json(기준값), .onnx 가 있으면 함께 등록")
    parser.add_argument("--data", help="평가 데이터 (기본: 앱 test 사진이 50장 이상이면 app, 아니면 test 분할)")
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--force", action="store_true", help="성능과 관계없이 교체")
    parser.add_argument("--rollback")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for v in load_registry()["versions"]:
            m = v["metrics"]
            print(f"{'*' if v['active'] else ' '} {v['version']}  {v['created_at'][:10]}  "
                  f"개수 일치 {m['count_exact_rate'] or 0:.1%}  mAP50 {m['map50'] or 0:.1%}  ({v['test_set']}, {v['n_images']}장)")
        return
    if args.rollback:
        rollback(args.rollback)
        return
    if not args.candidate:
        parser.error("--candidate 가 필요합니다")

    from app.detector import YoloDetector

    data = args.data or ("app" if len(load_eval_set("app")) >= MIN_APP_IMAGES else "test")
    items = load_eval_set(data)
    if not items:
        raise SystemExit(f"평가 데이터 '{data}' 가 비어 있습니다.")

    cand = run_evaluation(YoloDetector(args.candidate), items)
    print_summary(f"후보 {args.candidate.name}", cand)
    save_report(f"{data}_candidate", f"yolo:{args.candidate}", data, cand)

    cur = None
    if ACTIVE.exists():
        cur = run_evaluation(YoloDetector(ACTIVE), items)
        print_summary("현재 모델", cur)

    ok, reason = decide(cand["overall"], cur["overall"] if cur else None, args.min_gain)
    if args.force and not ok:
        ok, reason = True, f"--force ({reason})"
    if not ok:
        print(f"\n교체하지 않음: {reason}")
        return
    version = register(args.candidate, cand, data, reason)
    print(f"\n✅ {version} 으로 교체: {reason}\n서버를 다시 시작하면 새 모델이 적용됩니다.")


if __name__ == "__main__":
    main()
