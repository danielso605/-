"use strict";
// 휴대폰 안에서 YOLO(ONNX) 모델로 바로 인식 (서버·인터넷 없이).
// 후처리(신뢰도 기준 → 종류 무관 NMS → '확인 필요' 표시)는 서버 YoloDetector 와 같다.

(function (root) {
  const ORT_VERSION = "1.20.1";
  const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const INPUT = 640;
  const DEFAULTS = { conf: 0.25, iou: 0.5, review_conf: 0.5 };

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return union > 0 ? inter / union : 0;
  }

  function nms(dets, iouThr) {
    const kept = [];
    [...dets].sort((a, b) => b.confidence - a.confidence).forEach((d) => {
      if (kept.every((k) => iou(d.box, k.box) <= iouThr)) kept.push(d);
    });
    return kept;
  }

  /** 사진을 비율 유지로 640×640 에 넣고(남는 곳은 회색) NCHW float32 로 만든다. */
  function letterbox(source, size = INPUT) {
    const scale = Math.min(size / source.width, size / source.height);
    const w = Math.round(source.width * scale), h = Math.round(source.height * scale);
    const padX = Math.floor((size - w) / 2), padY = Math.floor((size - h) / 2);
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, size, size);
    ctx.drawImage(source, padX, padY, w, h);
    const px = ctx.getImageData(0, 0, size, size).data;
    const plane = size * size;
    const data = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      data[i] = px[i * 4] / 255;
      data[plane + i] = px[i * 4 + 1] / 255;
      data[2 * plane + i] = px[i * 4 + 2] / 255;
    }
    return { data, scale, padX, padY, width: source.width, height: source.height };
  }

  /**
   * YOLOv8/11 ONNX 출력 [1, 4+클래스수, 후보수] → 앱 형식 박스 (0~1000).
   * 각 후보: cx, cy, w, h (640 입력 기준 픽셀), 이어서 클래스별 점수.
   */
  function decode(output, dims, classes, lb, conf) {
    const [, channels, n] = dims;
    const nc = channels - 4;
    const dets = [];
    for (let j = 0; j < n; j++) {
      let best = -1, score = 0;
      for (let c = 0; c < nc; c++) {
        const s = output[(4 + c) * n + j];
        if (s > score) { score = s; best = c; }
      }
      if (score < conf) continue;
      const cx = output[j], cy = output[n + j], w = output[2 * n + j], h = output[3 * n + j];
      const toX = (v) => Math.max(0, Math.min(1000, Math.round(((v - lb.padX) / lb.scale / lb.width) * 1000)));
      const toY = (v) => Math.max(0, Math.min(1000, Math.round(((v - lb.padY) / lb.scale / lb.height) * 1000)));
      dets.push({
        nut_id: classes[best] || "unknown",
        box: [toX(cx - w / 2), toY(cy - h / 2), toX(cx + w / 2), toY(cy + h / 2)],
        confidence: Math.round(score * 1000) / 1000,
      });
    }
    return dets;
  }

  function postprocess(dets, t) {
    return nms(dets.filter((d) => d.confidence >= t.conf), t.iou)
      .map((d) => ({ ...d, needs_review: d.confidence < t.review_conf }));
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("인식 엔진을 불러오지 못했어요."));
      document.head.appendChild(s);
    });
  }

  let session = null, config = null;

  const OnDevice = {
    iou, nms, decode, postprocess,

    /** 서버에 휴대폰용 모델이 배포돼 있으면 설정을, 없으면 null. */
    async available() {
      if (config) return config;
      try {
        const res = await fetch("/api/config");
        const c = await res.json();
        if (!c.ondevice_model) return null;
        config = { model: c.ondevice_model, classes: c.classes, thresholds: { ...DEFAULTS, ...(c.thresholds || {}) } };
        return config;
      } catch {
        return null;
      }
    },

    async load() {
      if (session) return session;
      const c = await OnDevice.available();
      if (!c) throw new Error("휴대폰용 모델이 아직 없어요.");
      if (!root.ort) await loadScript(`${ORT_BASE}ort.min.js`);
      root.ort.env.wasm.wasmPaths = ORT_BASE;
      session = await root.ort.InferenceSession.create(c.model, { executionProviders: ["wasm"] });
      return session;
    },

    /** canvas/이미지 → 박스 목록 */
    async detect(source) {
      const s = await OnDevice.load();
      const lb = letterbox(source);
      const tensor = new root.ort.Tensor("float32", lb.data, [1, 3, INPUT, INPUT]);
      const out = await s.run({ [s.inputNames[0]]: tensor });
      const result = out[s.outputNames[0]];
      const raw = decode(result.data, result.dims, config.classes, lb, Math.min(0.01, config.thresholds.conf));
      return postprocess(raw, config.thresholds);
    },
  };

  if (typeof module !== "undefined" && module.exports) module.exports = OnDevice;
  else root.OnDevice = OnDevice;
})(typeof window !== "undefined" ? window : globalThis);
