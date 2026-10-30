# YOLO 학습 파이프라인

대상 클래스 7종 (번호 순서): `0 almond 아몬드`, `1 peanut 땅콩`, `2 pistachio 피스타치오`, `3 cashew 캐슈넛`,
`4 hazelnut 헤이즐넛`, `5 walnut 호두`, `6 raisin 건포도`
— 클래스 목록은 앱과 같은 `backend/app/data/nuts.json` 에서 읽습니다.

## 순서

```bash
cd training
pip install -r requirements.txt
cp sources.example.yaml sources.yaml      # 데이터 출처·링크 채우기

python download.py                         # 1. 데이터 내려받기 → raw/
python autolabel.py --dry-run              # 2. 박스 없는 사진 수 확인 (= Claude 호출 수)
python autolabel.py --workers 4            #    Claude 로 박스 라벨 생성 → autolabels/
python build_dataset.py --min-images 2000  # 3. 7종만 남겨 합치기, train/val/test 8:1:1 분할
python augment.py --min 5000 --max 10000   # 4. train 만 증강 → 클래스당 5,000~10,000장
python train.py --model yolo11s.pt --epochs 100   # 5. 학습(GPU) → test 평가 → backend/models/nuts.pt
```

학습된 모델로 앱 실행:

```bash
cd ../backend
NUT_DETECTOR=yolo NUT_YOLO_WEIGHTS=models/nuts.pt uvicorn app.main:app
```

## 단계별 설명

| 단계 | 하는 일 | 주의 |
| --- | --- | --- |
| download | Roboflow(API 키) / zip 링크 / 로컬 폴더 | Kaggle·Mendeley 는 로그인 후 받은 링크 또는 수동 다운로드 후 `local` 로 |
| autolabel | 분류용 사진(박스 없음)에 Claude 로 박스 생성. 폴더 클래스를 알려 주고 박스만 찾게 함 | 사진 1장 = API 1회 호출. 먼저 `--limit 50` 으로 품질 확인 후 전체 실행, 일부는 사람이 검수 권장 |
| build_dataset | 7종 외 클래스가 섞인 사진은 통째로 제외, 중복 제거, 해시 기반 고정 분할 | 클래스별 원본 2,000장 미달 시 경고 출력 |
| augment | 밝기·대비·감마, 90°회전·±30°회전, 반전, 약간의 크기/색조 변화. 부족한 클래스부터 채움 | **train 에만** 적용 (val/test 는 원본만 → 정직한 성능 측정) |
| train | Ultralytics YOLO11 학습, test 분할 mAP 출력, ONNX 등 내보내기 | GPU 필수 수준. Colab(T4)·로컬 GPU 권장 |

### "클래스당 N장"의 기준

한 사진에 여러 종류가 섞여 있을 수 있으므로, **그 견과류가 1개 이상 들어 있는 사진 수**로 셉니다.
원본 목표 2,000장 → 증강 후 5,000~10,000장. 원본이 적을수록 증강본이 같은 사진의 변형이 되어
과적합 위험이 커지므로, 증강 배수보다 **원본 다양성**(배경·조명·껍질 유무·섞인 사진)이 더 중요합니다.

## 공개 데이터 후보 (2026-10 조사)

박스 라벨이 있는 7종 통합 데이터셋은 없어서 조합 + 자동 라벨링이 필요합니다.

| 데이터 | 포함 클래스 | 형식 |
| --- | --- | --- |
| [Roboflow 프로젝트](https://universe.roboflow.com/baraka-qofeh/barakat_model_objec2t_detection-697np) | 호두·아몬드·캐슈넛, 1,096장 | 박스 있음 |
| [AllerNuts (Mendeley)](https://data.mendeley.com/datasets/3gtbxvgm5f) | 아몬드·캐슈넛·땅콩·피스타치오, 4,390장 | 분류용 |
| [Dry Fruit Image Dataset (Mendeley)](https://data.mendeley.com/datasets/yfhgn8py5f/1) | 아몬드·캐슈넛·건포도(+무화과) | 분류용 |
| [11종 견과 데이터셋 (논문)](https://journals.sagepub.com/doi/10.1177/02601060221113928) | 헤이즐넛·호두 등 종류별 200장 | 분류용 |

**헤이즐넛·호두·건포도**는 공개 데이터가 특히 적어 2,000장을 채우려면 직접 촬영(또는 Roboflow Universe·Kaggle 추가 검색)이 필요합니다.
각 데이터셋의 **라이선스(상업적 이용 가능 여부)를 반드시 확인**하세요.
