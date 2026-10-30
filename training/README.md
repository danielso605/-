# YOLO 학습 파이프라인

대상 클래스 7종 (번호 순서): `0 almond 아몬드`, `1 peanut 땅콩`, `2 pistachio 피스타치오`, `3 cashew 캐슈넛`,
`4 hazelnut 헤이즐넛`, `5 walnut 호두`, `6 raisin 건포도`
— 클래스 목록은 앱과 같은 `backend/app/data/nuts.json` 에서 읽습니다.

## 순서

```bash
cd training
pip install -r requirements.txt
cp sources.example.yaml sources.yaml        # 데이터 출처·링크 채우기 (format: yolo / coco / voc / classified)

python download.py                           # 1. 데이터 내려받기 → raw/
python autolabel.py --dry-run                # 2. 박스 없는 사진 수 확인 (= Claude 호출 수)
python autolabel.py --workers 4              #    Claude 로 박스 라벨 생성 → autolabels/
python build_dataset.py --min-images 2000    # 3. 7종만 남겨 합치기, train/val/test 8:1:1 분할
python check_dataset.py --write-rejects      # 4. 품질 검사 (깨진 사진·잘못된 박스는 rejects.txt 로)
python review.py --source allernuts          # 5. 모아보기로 자동 라벨 검수 → 틀린 id 를 rejects.txt 에
python build_dataset.py                      #    (rejects 반영해 다시 빌드)
python synthesize.py --count 3000 --backgrounds raw/backgrounds   # 6. 섞인 사진 합성 (train)
python augment.py --min 5000 --max 10000     # 7. train 만 증강 → 클래스당 5,000~10,000장
python train.py --model yolo11s.pt           # 8. 학습(GPU) → best.pt + best.onnx(휴대폰용)
python tune.py --weights runs/nuts/weights/best.pt --data val      # 9. 판정 기준값 → best.json
python promote.py --candidate runs/nuts/weights/best.pt            # 10. 지금 모델보다 나을 때만 앱에 적용
python evaluate.py --data app --detector yolo:../backend/models/nuts.pt claude   # 실사진 정확도 비교
```

`promote.py` 가 `backend/models/nuts.pt`(+`nuts.json` 기준값, `nuts.onnx` 휴대폰용)를 바꾸면, 서버를 다시 시작하는 것만으로
앱이 YOLO를 씁니다. API 키도 있으면 자동으로 **하이브리드**(YOLO 우선, 애매하면 Claude)로 동작합니다.

## 단계별 설명

| 단계 | 하는 일 | 주의 |
| --- | --- | --- |
| download | Roboflow(API 키) / zip 링크 / 로컬 폴더 | Kaggle·Mendeley 는 로그인 후 받은 링크 또는 수동 다운로드 후 `local` 로 |
| autolabel | 분류용 사진(박스 없음)에 Claude 로 박스 생성. 폴더 클래스를 알려 주고 박스만 찾게 함 | 사진 1장 = API 1회 호출. 먼저 `--limit 50` 으로 품질 확인 후 전체 실행 |
| build_dataset | yolo·coco·voc·classified 형식을 합침. 7종 외 클래스가 섞인 사진은 통째로 제외, 중복 제거, 해시 기반 고정 분할 | 앱의 정확도 측정용 사진(samples/test)은 학습에 못 들어가게 막음 |
| check_dataset | 깨진 사진, 라벨 없는 사진, 이미지 밖 박스, 너무 작은/큰 박스, 중복 박스 | `--write-rejects` 는 심각한 문제만 자동 제외 |
| review | 박스 그린 사진을 6×4 격자로 모아 `review/index.html` 로 | 틀린 칸 위의 id 12자리를 `rejects.txt` 에 |
| synthesize | train 의 낱알을 오려 배경 위에 섞어 붙임 (실제 크기 비율, 가림 40% 이하, 부족한 종류 우선) | 견과류 없는 배경 사진(접시·손·식탁) 폴더를 주면 훨씬 현실적 |
| augment | 밝기·대비·감마, 90°회전·±30°회전, 반전, 약간의 크기/색조 변화. 부족한 클래스부터 채움 | **train 에만** 적용 (val/test 는 원본만 → 정직한 성능 측정) |
| train | Ultralytics YOLO11 학습, test 분할 mAP 출력, ONNX 내보내기 | GPU 필수 수준. Colab(T4)·로컬 GPU 권장. 휴대폰 인식은 입력 640 기준 |
| tune | val 에서 conf·iou 조합을 바꿔 **개수 완전 일치 비율**이 가장 높은 값 선택, 정밀도가 낮은 신뢰도 구간은 '확인 필요'로 | test 로는 튜닝 금지 (스크립트가 막음) |
| promote | 같은 평가 데이터로 후보·현재 모델 비교 → 개수 정확도(동률이면 mAP50)가 1%p 이상 좋을 때만 교체, `versions/` 보관 | `--list`, `--rollback v2` |
| evaluate | 여러 인식기를 같은 사진으로 평가해 `backend/data/reports/` 에 저장 → 앱 `/report` 그래프 | `--data app` 이 실제 사용 환경과 가장 가까움 |

## 앱 실사진 테스트셋 · 재학습 데이터

앱의 **`/label` (인식 테스트)** 화면에서 직접 찍은 사진의 인식 결과를 고쳐 저장하면:

- **테스트용** → `backend/data/samples/test/` : `/report` 정확도 그래프와 `evaluate.py --data app`, `promote.py` 판정에 사용. 학습에는 절대 안 들어감.
- **재학습용** → `backend/data/samples/feedback/` : YOLO 형식이라 `sources.yaml` 에 아래처럼 추가하면 다음 학습에 들어감.

```yaml
  - name: app_feedback
    type: local
    path: ../backend/data/samples/feedback
    format: yolo
    class_map: {almond: almond, peanut: peanut, pistachio: pistachio, cashew: cashew, hazelnut: hazelnut, walnut: walnut, raisin: raisin}
```

정답을 '인식 결과를 고치는' 방식으로 만들기 때문에, 대충 넘기면 인식기에 유리하게 기울 수 있습니다.
테스트용은 박스 하나하나 꼼꼼히 확인하고, **최소 100장 이상**(종류별 고르게, 섞인 사진 위주) 모으는 것을 권장합니다.

## 테스트

```bash
cd training && pytest      # 형식 변환, 검사, 검수, 합성, 평가, 튜닝, 버전 관리 (작은 가짜 데이터로)
```

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
