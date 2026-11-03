// node --test frontend/tests/*.test.js   (설치 필요 없음)
"use strict";
const test = require("node:test");
const assert = require("node:assert");
const OnDevice = require("../ondevice.js");

const CLASSES = ["almond", "peanut", "pistachio", "cashew", "hazelnut", "walnut", "raisin"];

// YOLO11 ONNX 출력과 같은 [1, 4+7, N] 배열을 직접 만든다
function fakeOutput(n, hot) {
  const channels = 4 + CLASSES.length;
  const data = new Float32Array(channels * n);
  hot.forEach(([j, cx, cy, w, h, cls, score]) => {
    [cx, cy, w, h].forEach((v, k) => { data[k * n + j] = v; });
    data[(4 + cls) * n + j] = score;
  });
  return { data, dims: [1, channels, n] };
}

test("decode undoes letterbox and maps to 0-1000", () => {
  const { data, dims } = fakeOutput(100, [[0, 200, 320, 100, 80, 0, 0.9]]);
  const lb = { scale: 0.64, padX: 0, padY: 64, width: 1000, height: 800 }; // 1000×800 사진
  const [d] = OnDevice.decode(data, dims, CLASSES, lb, 0.01);
  assert.strictEqual(d.nut_id, "almond");
  assert.deepStrictEqual(d.box, [234, 422, 391, 578]);
});

test("postprocess filters, removes duplicates across classes and flags low confidence", () => {
  const { data, dims } = fakeOutput(100, [
    [0, 200, 320, 100, 80, 0, 0.9],
    [1, 205, 322, 100, 80, 3, 0.7],   // 같은 물체를 캐슈넛으로도 → 제거
    [2, 450, 320, 40, 40, 6, 0.4],    // 확인 필요
    [3, 500, 400, 50, 50, 5, 0.1],    // 기준 미달
  ]);
  const lb = { scale: 1, padX: 0, padY: 0, width: 640, height: 640 };
  const out = OnDevice.postprocess(OnDevice.decode(data, dims, CLASSES, lb, 0.01), { conf: 0.25, iou: 0.5, review_conf: 0.5 });
  assert.deepStrictEqual(out.map((d) => [d.nut_id, d.needs_review]), [["almond", false], ["raisin", true]]);
});
