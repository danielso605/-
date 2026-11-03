"""YOLO 학습 → 평가(test) → 배포용 내보내기.

GPU 에서 실행하세요. (CPU 로는 수만 장 학습에 며칠이 걸립니다.)

    python train.py --model yolo11s.pt --epochs 100 --imgsz 640
    # 끝나면 tune.py → promote.py 순서로 앱에 적용
"""

import argparse

from ultralytics import YOLO

from build_dataset import OUT
from common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s.pt", help="사전학습 가중치 (yolo11n/s/m.pt) 또는 .yaml(처음부터)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 이면 GPU 메모리에 맞게 자동")
    parser.add_argument("--device", default=None, help="예: 0, cpu")
    parser.add_argument("--export", nargs="*", default=["onnx"], help="내보낼 형식 (onnx, tflite, coreml …)")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(OUT / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs"),
        name="nuts",
        exist_ok=True,
        patience=20,
        # 오프라인 증강을 이미 했으므로 학습 중 색·회전 증강은 약하게 유지
        hsv_v=0.2,
        degrees=0.0,
        fliplr=0.5,
        mosaic=1.0,
    )

    best = ROOT / "runs" / "nuts" / "weights" / "best.pt"
    best_model = YOLO(str(best))
    metrics = best_model.val(data=str(OUT / "data.yaml"), split="test", imgsz=args.imgsz, device=args.device)
    print(f"\n[test] mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}")
    for i, name in best_model.names.items():
        print(f"  {name:<10} mAP50-95={metrics.box.maps[i]:.3f}")

    for fmt in args.export:
        best_model.export(format=fmt, imgsz=args.imgsz)  # runs/nuts/weights/best.onnx 등
    print(f"""
다음 단계:
  python tune.py --weights {best} --data val      # 판정 기준값 고르기 → best.json
  python promote.py --candidate {best}            # 지금 모델보다 나을 때만 앱에 적용""")


if __name__ == "__main__":
    main()
