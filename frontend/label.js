"use strict";

const UNKNOWN = "unknown";
const HANDLE = 28;   // 모서리 잡는 범위 (0~1000 좌표 기준)
const MIN_SIZE = 8;  // 이보다 작은 박스는 실수로 보고 버림

const st = {
  nuts: {},
  image: null,
  blob: null,
  detector: "",
  pred: [],           // 인식기 원본 결과 (저장 시 함께 보냄)
  gt: [],             // 사용자가 고치는 정답
  selected: -1,
  tool: "select",
  currentClass: null,
  drag: null,
};

const $ = (sel) => document.querySelector(sel);

async function init() {
  const { nuts, list } = await loadNuts();
  st.nuts = nuts;
  st.currentClass = list[0].id;
  $("#palette").innerHTML = list.map((n) =>
    `<button class="chip swatch-btn" data-id="${n.id}" style="--c:${nuts[n.id].color}">${n.name_ko}</button>`).join("");
  $("#palette").addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-id]");
    if (btn) chooseClass(btn.dataset.id);
  });

  $("#camera-input").addEventListener("change", onFile);
  $("#album-input").addEventListener("change", onFile);
  $("#tool-select").addEventListener("click", () => setTool("select"));
  $("#tool-draw").addEventListener("click", () => setTool("draw"));
  $("#delete-btn").addEventListener("click", deleteSelected);
  $("#consent").addEventListener("change", updateSave);
  $("#save-btn").addEventListener("click", save);
  document.addEventListener("keydown", (e) => {
    if ((e.key === "Delete" || e.key === "Backspace") && st.selected >= 0 && e.target === document.body) deleteSelected();
  });

  const canvas = $("#canvas");
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerup", onUp);
  canvas.addEventListener("pointercancel", onUp);

  setTool("select");
  refreshSavedCount();
}

function status(msg, isError = false) {
  const el = $("#status");
  el.hidden = !msg;
  el.textContent = msg || "";
  el.classList.toggle("error", isError);
}

async function refreshSavedCount() {
  try {
    const counts = await (await fetch("/api/samples")).json();
    $("#saved-count").textContent = `저장된 사진: 테스트 ${counts.test}장 · 재학습용 ${counts.feedback}장`;
  } catch { /* 표시만 생략 */ }
}

async function onFile(event) {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  status("🔍 인식 중…");
  $("#editor").hidden = true;
  try {
    const { blob, image } = await resizeImage(file);
    st.image = image;
    st.blob = blob;
    const form = new FormData();
    form.append("image", blob, "photo.jpg");
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "인식에 실패했어요.");

    st.detector = data.detector;
    st.pred = data.detections.map((d) => ({ nut_id: d.nut_id, box: d.box, confidence: d.confidence }));
    st.gt = data.detections.map((d) => ({ nut_id: d.nut_id, box: [...d.box], review: !!d.needs_review }));
    st.selected = -1;
    status(`인식기: ${data.detector} · 박스 ${st.pred.length}개. 틀린 곳을 고친 뒤 저장해 주세요.`);
    $("#editor").hidden = false;
    render();
  } catch (err) {
    status(err.message, true);
  }
}

// ---------- 편집 ----------

function setTool(tool) {
  st.tool = tool;
  $("#tool-select").classList.toggle("active", tool === "select");
  $("#tool-select").setAttribute("aria-pressed", String(tool === "select"));
  $("#tool-draw").classList.toggle("active", tool === "draw");
  $("#tool-draw").setAttribute("aria-pressed", String(tool === "draw"));
  $("#canvas").classList.toggle("drawing", tool === "draw");
  $("#tool-hint").textContent = tool === "draw"
    ? `사진 위를 끌어서 '${st.nuts[st.currentClass].name_ko}' 박스를 그리세요. 종류는 위 버튼으로 바꿀 수 있어요.`
    : "박스를 눌러 선택 → 종류 버튼으로 바꾸기, 끌어서 이동, 모서리를 끌어 크기 조절, 🗑 로 삭제.";
}

function chooseClass(id) {
  st.currentClass = id;
  if (st.selected >= 0) {
    st.gt[st.selected].nut_id = id;
    st.gt[st.selected].review = false;
  }
  setTool(st.tool);
  render();
}

function deleteSelected() {
  if (st.selected < 0) return;
  st.gt.splice(st.selected, 1);
  st.selected = -1;
  render();
}

function toImageCoords(e) {
  const rect = $("#canvas").getBoundingClientRect();
  return [
    Math.max(0, Math.min(1000, ((e.clientX - rect.left) / rect.width) * 1000)),
    Math.max(0, Math.min(1000, ((e.clientY - rect.top) / rect.height) * 1000)),
  ];
}

function hitCorner(box, x, y) {
  const corners = [[0, 1], [2, 1], [0, 3], [2, 3]];
  return corners.findIndex(([ix, iy]) => Math.abs(box[ix] - x) < HANDLE && Math.abs(box[iy] - y) < HANDLE);
}

function hitBox(x, y) {
  // 겹치면 가장 작은 박스를 고른다 (붙어 있는 작은 견과류를 선택하기 쉽게)
  let best = -1, bestArea = Infinity;
  st.gt.forEach((g, i) => {
    const [x1, y1, x2, y2] = g.box;
    const area = (x2 - x1) * (y2 - y1);
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2 && area < bestArea) { best = i; bestArea = area; }
  });
  return best;
}

function onDown(e) {
  const [x, y] = toImageCoords(e);
  e.target.setPointerCapture(e.pointerId);
  if (st.tool === "draw") {
    st.gt.push({ nut_id: st.currentClass, box: [x, y, x, y], review: false });
    st.selected = st.gt.length - 1;
    st.drag = { kind: "draw", ox: x, oy: y };
  } else {
    const sel = st.selected >= 0 ? st.gt[st.selected] : null;
    const corner = sel ? hitCorner(sel.box, x, y) : -1;
    if (corner >= 0) {
      st.drag = { kind: "resize", corner };
    } else {
      st.selected = hitBox(x, y);
      st.drag = st.selected >= 0 ? { kind: "move", lx: x, ly: y } : null;
    }
  }
  render();
}

function onMove(e) {
  if (!st.drag) return;
  const [x, y] = toImageCoords(e);
  const box = st.gt[st.selected].box;
  if (st.drag.kind === "draw") {
    st.gt[st.selected].box = [Math.min(st.drag.ox, x), Math.min(st.drag.oy, y), Math.max(st.drag.ox, x), Math.max(st.drag.oy, y)];
  } else if (st.drag.kind === "move") {
    const dx = Math.max(-box[0], Math.min(1000 - box[2], x - st.drag.lx));
    const dy = Math.max(-box[1], Math.min(1000 - box[3], y - st.drag.ly));
    st.gt[st.selected].box = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy];
    st.drag.lx = x;
    st.drag.ly = y;
  } else {
    const [ix, iy] = [[0, 1], [2, 1], [0, 3], [2, 3]][st.drag.corner];
    box[ix] = x;
    box[iy] = y;
  }
  render();
}

function onUp() {
  if (!st.drag) return;
  if (st.selected >= 0) {
    const g = st.gt[st.selected];
    const [x1, y1, x2, y2] = g.box;
    g.box = [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)].map(Math.round);
    if (g.box[2] - g.box[0] < MIN_SIZE || g.box[3] - g.box[1] < MIN_SIZE) {
      st.gt.splice(st.selected, 1);
      st.selected = -1;
    }
  }
  st.drag = null;
  render();
}

// ---------- 그리기 ----------

function render() {
  const src = st.image;
  const canvas = $("#canvas");
  canvas.width = src.width;
  canvas.height = src.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(src, 0, 0);

  const sx = src.width / 1000, sy = src.height / 1000;
  const line = Math.max(3, src.width / 260);
  const fontPx = Math.max(16, src.width / 36);
  ctx.font = `bold ${fontPx}px sans-serif`;
  ctx.textBaseline = "top";

  st.gt.forEach((g, i) => {
    const nut = st.nuts[g.nut_id];
    const color = nut ? nut.color : "#9e9e9e";
    const [x1, y1, x2, y2] = [g.box[0] * sx, g.box[1] * sy, g.box[2] * sx, g.box[3] * sy];
    const selected = i === st.selected;
    ctx.lineWidth = selected ? line * 1.8 : line;
    ctx.strokeStyle = color;
    ctx.setLineDash(g.review || !nut ? [line * 3, line * 2] : []);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);

    const label = nut ? nut.name_ko : "? 종류 선택";
    const w = ctx.measureText(label).width + 10;
    const ty = Math.max(0, y1 - fontPx - 6);
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty, w, fontPx + 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 5, ty + 3);

    if (selected) {
      const h = HANDLE * 0.6 * sx;
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = color;
      ctx.lineWidth = line / 1.5;
      for (const [cx, cy] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]]) {
        ctx.fillRect(cx - h / 2, cy - h / 2, h, h);
        ctx.strokeRect(cx - h / 2, cy - h / 2, h, h);
      }
    }
  });

  document.querySelectorAll("#palette button").forEach((b) => {
    const active = st.selected >= 0 ? st.gt[st.selected].nut_id === b.dataset.id : st.currentClass === b.dataset.id;
    b.classList.toggle("active", active);
  });
  $("#delete-btn").disabled = st.selected < 0;
  renderCompare();
  updateSave();
}

function renderCompare() {
  const count = (list) => list.reduce((m, d) => m.set(d.nut_id, (m.get(d.nut_id) || 0) + 1), new Map());
  const pc = count(st.pred), gc = count(st.gt);
  const ids = [...new Set([...Object.keys(st.nuts), ...pc.keys(), ...gc.keys()])].filter((id) => pc.get(id) || gc.get(id));
  $("#compare-body").innerHTML = ids.map((id) => {
    const p = pc.get(id) || 0, g = gc.get(id) || 0;
    const name = st.nuts[id] ? st.nuts[id].name_ko : "알 수 없음";
    return `<tr class="${p === g ? "" : "diff"}"><td>${name}</td><td>${p}</td><td>${g}</td></tr>`;
  }).join("") || `<tr><td colspan="3">박스가 없어요</td></tr>`;
}

function updateSave() {
  const unresolved = st.gt.some((g) => !st.nuts[g.nut_id]);
  $("#save-btn").disabled = !$("#consent").checked || unresolved;
  $("#save-btn").textContent = unresolved ? "점선 '?' 박스의 종류를 정하거나 지워 주세요" : "💾 정답 저장";
}

async function save() {
  const split = document.querySelector("input[name=split]:checked").value;
  const payload = {
    split,
    consent: $("#consent").checked,
    detector: st.detector,
    gt: st.gt.map(({ nut_id, box }) => ({ nut_id, box })),
    pred: st.pred,
  };
  const form = new FormData();
  form.append("image", st.blob, "photo.jpg");
  form.append("payload", JSON.stringify(payload));
  $("#save-btn").disabled = true;
  try {
    const res = await fetch("/api/samples", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "저장에 실패했어요.");
    status(`✅ 저장했어요 (${split === "test" ? "테스트" : "재학습용"} ${data.count}장째). 다음 사진을 찍어 주세요.`);
    $("#editor").hidden = true;
    refreshSavedCount();
  } catch (err) {
    status(err.message, true);
    updateSave();
  }
}

init();
