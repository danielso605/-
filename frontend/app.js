"use strict";

const MAX_SIDE = 1568; // 업로드 전 축소: 인식 품질은 유지하면서 전송량·비용을 줄인다
const COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#f4511e", "#3949ab", "#c0ca33", "#6d4c41", "#d81b60", "#546e7a"];

const state = {
  mode: localGet("mode") || "senior",
  nuts: {},           // id -> nut 정보 (/api/nuts)
  dailyG: 30,
  counts: new Map(),  // id -> 개수 (사용자 수정 반영)
  detections: [],
  image: null,
};

const $ = (sel) => document.querySelector(sel);

function localGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function localSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* 저장 불가 환경 무시 */ }
}

async function init() {
  setMode(state.mode);
  if (localGet("large") === "1") document.body.classList.add("large");

  document.querySelectorAll(".modes button").forEach((btn) =>
    btn.addEventListener("click", () => { setMode(btn.dataset.mode); render(); }));
  $("#font-toggle").addEventListener("click", () => {
    const large = document.body.classList.toggle("large");
    localSet("large", large ? "1" : "0");
    drawPhoto();
  });
  $("#camera-input").addEventListener("change", onFile);
  $("#album-input").addEventListener("change", onFile);
  $("#speak").addEventListener("click", speak);
  $("#add-btn").addEventListener("click", () => {
    const id = $("#add-select").value;
    state.counts.set(id, (state.counts.get(id) || 0) + 1);
    render();
  });

  try {
    const res = await fetch("/api/nuts");
    const data = await res.json();
    state.dailyG = data.daily_recommended_g;
    data.nuts.forEach((nut, i) => { state.nuts[nut.id] = { ...nut, color: COLORS[i % COLORS.length] }; });
    $("#add-select").innerHTML = data.nuts.map((n) => `<option value="${n.id}">${n.name_ko}</option>`).join("");
  } catch {
    showStatus("서버에 연결할 수 없어요. 인터넷 연결을 확인해 주세요.", true);
  }
}

function setMode(mode) {
  state.mode = mode;
  document.body.dataset.mode = mode;
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === mode)));
  localSet("mode", mode);
}

function showStatus(message, isError = false) {
  const el = $("#status");
  el.hidden = !message;
  el.textContent = message || "";
  el.classList.toggle("error", isError);
}

async function onFile(event) {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;

  showStatus("🔍 견과류를 찾고 있어요… 잠시만 기다려 주세요.");
  $("#result").hidden = true;

  try {
    const { blob, image } = await resizeImage(file);
    state.image = image;
    const form = new FormData();
    form.append("image", blob, "photo.jpg");
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "분석에 실패했어요.");

    state.detections = data.detections;
    state.counts = new Map(data.items.map((item) => [item.nut_id, item.count]));

    if (data.items.length === 0) {
      showStatus("견과류를 찾지 못했어요. 밝은 곳에서 가까이 다시 찍어 주세요.", true);
      return;
    }
    let msg = "";
    if (data.detector === "demo") msg = "※ 데모 모드: 실제 인식이 아닌 예시 결과입니다.";
    if (data.unknown_count) msg += ` 알 수 없는 견과류 ${data.unknown_count}개는 계산에서 뺐어요.`;
    showStatus(msg.trim());
    $("#result").hidden = false;
    render();
    $("#result").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showStatus(err.message || "분석에 실패했어요. 다시 시도해 주세요.", true);
  }
}

function resizeImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(img.width * scale);
      canvas.height = Math.round(img.height * scale);
      canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        URL.revokeObjectURL(url);
        blob ? resolve({ blob, image: canvas }) : reject(new Error("사진을 처리하지 못했어요."));
      }, "image/jpeg", 0.85);
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error("사진을 열 수 없어요.")); };
    img.src = url;
  });
}

function drawPhoto() {
  const src = state.image;
  if (!src) return;
  const canvas = $("#photo");
  canvas.width = src.width;
  canvas.height = src.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(src, 0, 0);

  const line = Math.max(3, src.width / 250);
  const fontPx = Math.max(16, src.width / 32) * (document.body.classList.contains("large") ? 1.3 : 1);
  ctx.font = `bold ${fontPx}px sans-serif`;
  ctx.textBaseline = "top";

  state.detections.forEach((d) => {
    const nut = state.nuts[d.nut_id];
    const color = nut ? nut.color : "#9e9e9e";
    const [x1, y1, x2, y2] = d.box.map((v, i) => (v / 1000) * (i % 2 === 0 ? src.width : src.height));
    ctx.lineWidth = line;
    ctx.strokeStyle = color;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = nut ? nut.name_ko : "?";
    const w = ctx.measureText(label).width + 10;
    const ty = Math.max(0, y1 - fontPx - 6);
    ctx.fillStyle = color;
    ctx.fillRect(x1, ty, w, fontPx + 6);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 5, ty + 3);
  });
}

function nutritionFor(id, count) {
  const nut = state.nuts[id];
  const weight = nut.piece_weight_g * count;
  const r = weight / 100;
  const p = nut.per_100g;
  return { weight, kcal: p.kcal * r, protein: p.protein_g * r, fat: p.fat_g * r, carbs: p.carbs_g * r, fiber: p.fiber_g * r };
}

function totals() {
  const t = { weight: 0, kcal: 0, protein: 0, fat: 0, carbs: 0, fiber: 0 };
  state.counts.forEach((count, id) => {
    const n = nutritionFor(id, count);
    Object.keys(t).forEach((k) => { t[k] += n[k]; });
  });
  return t;
}

const fmt = (v, digits = 0) => Number(v.toFixed(digits)).toLocaleString("ko-KR");
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function render() {
  drawPhoto();
  const t = totals();
  const ratio = t.weight / state.dailyG;
  const pct = Math.round(ratio * 100);

  $("#total").innerHTML = `
    <div class="sub">사진 속 견과류 총</div>
    <div class="kcal">${fmt(t.kcal)} kcal</div>
    <div class="sub">약 ${fmt(t.weight, 1)}g${state.mode === "athlete" ? ` · 단백질 ${fmt(t.protein, 1)}g` : ""}</div>
    <div class="meter ${ratio > 1 ? "over" : ""}" role="img" aria-label="하루 권장량의 ${pct}%"><span style="width:${Math.min(100, pct)}%"></span></div>
    <div class="sub">하루 권장량(한 줌 ${state.dailyG}g)의 <b>${pct}%</b></div>
    ${ratio > 1 ? `<div class="over-msg">하루 권장량보다 많아요. 나눠서 드세요.</div>` : ""}
  `;

  const entries = [...state.counts.entries()].filter(([, c]) => c > 0);
  $("#items").innerHTML = entries.map(([id, count]) => {
    const nut = state.nuts[id];
    const n = nutritionFor(id, count);
    return `
      <article class="card item">
        <div class="item-head">
          <span class="swatch" style="background:${nut.color}"></span>
          <h3>${esc(nut.name_ko)}</h3>
          <div class="counter">
            <button class="chip" data-id="${id}" data-delta="-1" aria-label="${esc(nut.name_ko)} 한 개 빼기">－</button>
            <output aria-live="polite">${count}개</output>
            <button class="chip" data-id="${id}" data-delta="1" aria-label="${esc(nut.name_ko)} 한 개 더하기">＋</button>
          </div>
        </div>
        <p class="item-kcal">${fmt(n.kcal)} kcal <span class="sub">(약 ${fmt(n.weight, 1)}g)</span></p>
        <div class="highlight">💡 ${esc(nut.highlight[state.mode])}</div>
        <div class="macros athlete-only">
          <div>단백질<b>${fmt(n.protein, 1)}g</b></div>
          <div>지방<b>${fmt(n.fat, 1)}g</b></div>
          <div>탄수화물<b>${fmt(n.carbs, 1)}g</b></div>
        </div>
        <strong>이런 점이 좋아요</strong>
        <ul>${nut.benefits.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>
        ${nut.cautions.length ? `<strong>주의하세요</strong><ul class="cautions">${nut.cautions.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` : ""}
      </article>`;
  }).join("");

  $("#items").querySelectorAll("button[data-delta]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const id = btn.dataset.id;
      state.counts.set(id, Math.max(0, (state.counts.get(id) || 0) + Number(btn.dataset.delta)));
      render();
    }));
}

function speak() {
  if (!("speechSynthesis" in window)) {
    showStatus("이 기기에서는 읽어주기를 지원하지 않아요.", true);
    return;
  }
  const t = totals();
  const parts = [...state.counts.entries()].filter(([, c]) => c > 0).map(([id, count]) => {
    const nut = state.nuts[id];
    return `${nut.name_ko} ${count}개, ${fmt(nutritionFor(id, count).kcal)}칼로리. ${nut.highlight[state.mode]}. ${nut.cautions.join(". ")}`;
  });
  const text = `모두 합쳐 ${fmt(t.kcal)}칼로리입니다. 하루 권장량의 ${Math.round((t.weight / state.dailyG) * 100)}퍼센트예요. ${parts.join(" ")}`;

  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "ko-KR";
  u.rate = state.mode === "senior" ? 0.85 : 1;
  speechSynthesis.speak(u);
}

init();
