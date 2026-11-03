# Hugging Face Spaces (Docker) 용. 이 Dockerfile + README.md + nuts.pt (+ 선택: nuts.onnx) 만 Space 에 올리면
# 앱 코드는 GitHub 에서 자동으로 받아 온다.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# GPU 없는 서버라 CPU 용 PyTorch (용량이 훨씬 작음)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

ARG REPO=https://github.com/danielso605/-.git
ARG BRANCH=claude/nut-recognition-calorie-app-y8lu3b
RUN git clone --depth 1 -b ${BRANCH} ${REPO} /app || git clone --depth 1 ${REPO} /app
WORKDIR /app/backend
RUN pip install --no-cache-dir -r requirements.txt ultralytics pillow

# Space 에 올린 모델 파일: .pt 필수(이름 상관없음, 예: best.pt), .onnx·기준값 .json 은 있으면 함께
COPY . /space/
RUN set -e; mkdir -p models data; \
    pt=$(ls /space/*.pt | head -1); cp "$pt" models/nuts.pt; \
    for ext in onnx json; do f=$(ls /space/*.$ext 2>/dev/null | head -1); if [ -n "$f" ]; then cp "$f" models/nuts.$ext; fi; done; \
    chmod -R 777 /app/backend/data /app/backend/models

ENV NUT_DETECTOR=yolo \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    HOME=/tmp
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
