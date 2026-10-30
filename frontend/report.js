"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const $ = (sel) => document.querySelector(sel);
let data = null;
let current = null;  // 선택된 결과 { label, desc, metrics }

const pct = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const num = (v, d = 2) => (v == null ? "–" : v.toFixed(d));

function svg(tag, attrs = {}, parent) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (parent) parent.appendChild(el);
  return el;
}

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

// ---------- 툴팁: 값이 먼저, 이름이 뒤 ----------

function showTip(evt, title, rows) {
  const tip = $("#tooltip");
  tip.replaceChildren();
  tip.appendChild(el("div", title));
  rows.forEach(([name, value, color]) => {
    const row = el("div", null, "row");
    if (color) {
      const key = el("span", null, "key");
      key.style.background = color;
      row.appendChild(key);
    }
    row.appendChild(el("strong", value));
    row.appendChild(el("span", name));
    tip.appendChild(row);
  });
  tip.hidden = false;
  const r = evt.target.getBoundingClientRect();
  const x = evt.clientX ?? r.left + r.width / 2;
  const y = evt.clientY ?? r.top;
  tip.style.left = `${Math.min(window.innerWidth - tip.offsetWidth - 8, x + 12)}px`;
  tip.style.top = `${Math.max(8, y - tip.offsetHeight - 8)}px`;
}

function hideTip() { $("#tooltip").hidden = true; }

function bindTip(target, title, rows) {
  target.setAttribute("tabindex", "0");
  target.setAttribute("aria-label", `${title}: ${rows.map((r) => `${r[0]} ${r[1]}`).join(", ")}`);
  target.addEventListener("pointermove", (e) => showTip(e, title, rows));
  target.addEventListener("focus", (e) => showTip(e, title, rows));
  target.addEventListener("pointerleave", hideTip);
  target.addEventListener("blur", hideTip);
}

const cssVar = (name) => getComputedStyle($(".viz-root")).getPropertyValue(name).trim();

// ---------- 데이터 선택 ----------

async function init() {
  data = await (await fetch("/api/report")).json();
  const options = [
    ...data.live.map((r) => ({
      label: `앱 테스트 사진 · ${r.detector} (${r.metrics.n_images}장)`,
      desc: `라벨 화면에서 저장한 사진에 대해, 저장 당시 '${r.detector}' 인식기 결과를 사용자가 고친 정답과 비교했어요.`,
      metrics: r.metrics,
    })),
    ...data.offline.map((r) => ({
      label: `오프라인 평가 · ${r.name} (${r.metrics.n_images}장)`,
      desc: `training/evaluate.py 결과 · 인식기 ${r.detector} · 데이터 ${r.dataset} · ${r.created_at.slice(0, 10)}`,
      metrics: r.metrics,
    })),
  ];
  if (!options.length) {
    $("#empty").hidden = false;
    $("#empty").textContent = "아직 테스트 결과가 없어요. '테스트하러 가기'에서 사진을 찍고 정답을 저장해 주세요.";
    return;
  }
  const select = $("#source");
  options.forEach((o, i) => select.appendChild(new Option(o.label, i)));
  select.addEventListener("change", () => { current = options[select.value]; render(); });
  current = options[0];
  $("#content").hidden = false;
  render();
  new ResizeObserver(() => render()).observe($("#chart-pr"));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
}

function render() {
  if (!current) return;
  const m = current.metrics;
  $("#source-desc").textContent = current.desc;
  renderTiles(m);
  renderPR(m);
  renderCount(m);
  renderConfusion(m);
  renderHistory();
  renderTable(m);
}

function renderTiles(m) {
  const tiles = [
    ["테스트 사진", `${m.n_images}장`, "많을수록 믿을 만해요 (100장 이상 권장)"],
    ["개수 완전 일치", pct(m.overall.count_exact_rate), "모든 종류 개수를 정확히 맞춘 사진 비율"],
    ["칼로리 평균 오차", m.overall.kcal_error_pct == null ? "–" : `${m.overall.kcal_error_pct.toFixed(1)}%`, "실제 칼로리 대비"],
    ["mAP50", pct(m.overall.map50), "박스 위치 + 종류 종합 점수"],
  ];
  $("#tiles").replaceChildren(...tiles.map(([label, value, sub]) => {
    const t = el("div", null, "tile");
    t.append(el("div", label, "label"), el("div", value, "value"), el("div", sub, "sub"));
    return t;
  }));
}

function classRows(m) {
  return data.classes.map((c) => ({ id: c, name: data.names_ko[c], ...m.per_class[c] }));
}

// ---------- 정밀도·재현율: 가로 묶음 막대 ----------

function renderPR(m) {
  const rows = classRows(m);
  const series = [
    { key: "precision", name: "정밀도", color: cssVar("--series-1") },
    { key: "recall", name: "재현율", color: cssVar("--series-2") },
  ];
  $("#legend-pr").replaceChildren(...series.map((s) => {
    const span = el("span", s.name);
    const i = el("i");
    i.style.background = s.color;
    span.prepend(i);
    return span;
  }));

  const width = $("#chart-pr").clientWidth || 600;
  const left = 78, right = 92, top = 20, barH = 12, gap = 2, groupH = barH * 2 + gap + 16;
  const plotW = Math.max(120, width - left - right);
  const height = top + rows.length * groupH + 4;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "종류별 정밀도와 재현율 막대 그래프" });
  const x = (v) => left + v * plotW;

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    svg("line", { x1: x(t), x2: x(t), y1: top - 4, y2: height, class: t === 0 ? "baseline" : "gridline" }, root);
    svg("text", { x: x(t), y: top - 8, "text-anchor": "middle", class: "tick" }, root).textContent = `${t * 100}%`;
  });

  rows.forEach((r, gi) => {
    const gy = top + gi * groupH + 8;
    const label = svg("text", { x: left - 8, y: gy + barH + 1, "text-anchor": "end", "dominant-baseline": "middle" }, root);
    label.textContent = r.name;
    if (!r.gt && !r.pred) {
      svg("text", { x: left + 6, y: gy + barH + 1, "dominant-baseline": "middle", class: "tick" }, root).textContent = "테스트 사진에 없음";
      return;
    }
    const hit = svg("rect", { x: 0, y: gy - 4, width, height: barH * 2 + gap + 8, class: "hit" }, root);
    const g = svg("g", { class: "mark" }, root);
    series.forEach((s, si) => {
      const v = r[s.key];
      const y = gy + si * (barH + gap);
      if (v == null) return;
      const w = Math.max(v * plotW, 2);
      // 기준선 쪽은 각지고 끝만 둥근 막대
      svg("path", { d: barPath(left, y, w, barH, 4), fill: s.color }, g);
    });
    const lbl = svg("text", { x: x(1) + 8, y: gy + barH + 1, "dominant-baseline": "middle", class: "value-label" }, root);
    lbl.textContent = `${pct(r.precision)} / ${pct(r.recall)}`;
    bindTip(hit, r.name, [
      ["정밀도", pct(r.precision), series[0].color],
      ["재현율", pct(r.recall), series[1].color],
      ["정답 / 인식", `${r.gt} / ${r.pred}개`],
    ]);
  });
  $("#chart-pr").replaceChildren(root);
}

function barPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}

// ---------- 개수 오차: 단일 계열 가로 막대 ----------

function renderCount(m) {
  const rows = classRows(m).filter((r) => r.gt || r.pred);
  const width = $("#chart-count").clientWidth || 600;
  const left = 78, right = 56, top = 20, barH = 16, rowH = 30;
  const plotW = Math.max(120, width - left - right);
  const height = top + rows.length * rowH;
  const max = Math.max(1, ...rows.map((r) => r.count_mae ?? 0));
  const niceMax = Math.ceil(max * 2) / 2;
  const x = (v) => left + (v / niceMax) * plotW;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "종류별 평균 개수 오차 막대 그래프" });

  const ticks = [0, niceMax / 2, niceMax];
  ticks.forEach((t) => {
    svg("line", { x1: x(t), x2: x(t), y1: top - 4, y2: height, class: t === 0 ? "baseline" : "gridline" }, root);
    svg("text", { x: x(t), y: top - 8, "text-anchor": "middle", class: "tick" }, root).textContent = `${t}개`;
  });
  const color = cssVar("--series-1");
  rows.forEach((r, i) => {
    const y = top + i * rowH + (rowH - barH) / 2;
    svg("text", { x: left - 8, y: y + barH / 2, "text-anchor": "end", "dominant-baseline": "middle" }, root).textContent = r.name;
    const hit = svg("rect", { x: 0, y: top + i * rowH, width, height: rowH, class: "hit" }, root);
    const g = svg("g", { class: "mark" }, root);
    svg("path", { d: barPath(left, y, Math.max(2, x(r.count_mae) - left), barH, 4), fill: color }, g);
    svg("text", { x: x(r.count_mae) + 6, y: y + barH / 2, "dominant-baseline": "middle", class: "value-label" }, root)
      .textContent = `±${num(r.count_mae)}개`;
    bindTip(hit, r.name, [["사진당 평균 개수 차이", `${num(r.count_mae)}개`, color]]);
  });
  $("#chart-count").replaceChildren(root);
}

// ---------- 혼동 행렬: 단일 색상 농도 ----------

function renderConfusion(m) {
  const present = data.classes.filter((c) => m.per_class[c].gt);
  const cols = [...data.classes, "missed"];
  const colName = (c) => (c === "missed" ? "놓침" : data.names_ko[c]);
  const steps = ["--seq-0", "--seq-1", "--seq-2", "--seq-3", "--seq-4", "--seq-5"].map(cssVar);

  const table = el("table", null, "heat");
  const head = el("tr");
  head.appendChild(el("th", "정답 ＼ 인식", "corner"));
  cols.forEach((c) => head.appendChild(el("th", colName(c))));
  table.appendChild(head);

  const rowsToShow = [...present, ...(m.confusion.false_pos && Object.keys(m.confusion.false_pos).length ? ["false_pos"] : [])];
  rowsToShow.forEach((g) => {
    const tr = el("tr");
    tr.appendChild(el("th", g === "false_pos" ? "없는데 찾음" : data.names_ko[g]));
    const row = m.confusion[g] || {};
    const total = Object.values(row).reduce((a, b) => a + b, 0) || 1;
    cols.forEach((c) => {
      const n = row[c] || 0;
      const share = g === "false_pos" ? null : n / total;
      const td = el("td", n ? (share == null ? `${n}` : `${Math.round(share * 100)}%`) : "");
      const level = n === 0 ? 0 : share == null ? 2 : Math.min(5, 1 + Math.floor(share * 5));
      td.style.background = steps[level];
      if (level > 0) td.style.color = isDark(steps[level]) ? "#ffffff" : "#0b0b0b";
      if (n) bindTip(td, `${g === "false_pos" ? "없는데 찾음" : `정답 ${data.names_ko[g]}`} → ${colName(c)}`,
        [["건수", `${n}개`], ...(share == null ? [] : [["비율", pct(share)]])]);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  $("#chart-confusion").replaceChildren(table);
}

function isDark(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45;
}

// ---------- 모델 버전 이력: 단일 선 ----------

function renderHistory() {
  const versions = (data.history || []).filter((v) => v.metrics && v.metrics.map50 != null);
  $("#history-card").hidden = versions.length === 0;
  if (!versions.length) return;
  const width = $("#chart-history").clientWidth || 600;
  const left = 44, right = 20, top = 16, bottom = 36, height = 200;
  const plotW = width - left - right, plotH = height - top - bottom;
  const x = (i) => left + (versions.length === 1 ? plotW / 2 : (i / (versions.length - 1)) * plotW);
  const y = (v) => top + (1 - v) * plotH;
  const color = cssVar("--series-1");
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "모델 버전별 mAP50 선 그래프" });

  [0, 0.5, 1].forEach((t) => {
    svg("line", { x1: left, x2: width - right, y1: y(t), y2: y(t), class: t === 0 ? "baseline" : "gridline" }, root);
    svg("text", { x: left - 6, y: y(t), "text-anchor": "end", "dominant-baseline": "middle", class: "tick" }, root).textContent = `${t * 100}%`;
  });
  svg("polyline", {
    points: versions.map((v, i) => `${x(i)},${y(v.metrics.map50)}`).join(" "),
    fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round",
  }, root);
  versions.forEach((v, i) => {
    svg("text", { x: x(i), y: height - 14, "text-anchor": "middle", class: "tick" }, root).textContent = v.version;
    const hit = svg("circle", { cx: x(i), cy: y(v.metrics.map50), r: 14, class: "hit" }, root);
    svg("circle", { cx: x(i), cy: y(v.metrics.map50), r: 4.5, fill: color, stroke: cssVar("--surface-1"), "stroke-width": 2, class: "mark" }, root);
    bindTip(hit, `${v.version}${v.active ? " (사용 중)" : ""}`, [
      ["mAP50", pct(v.metrics.map50), color],
      ["개수 완전 일치", pct(v.metrics.count_exact_rate)],
      ["등록일", v.created_at.slice(0, 10)],
    ]);
  });
  const last = versions[versions.length - 1];
  svg("text", { x: x(versions.length - 1), y: y(last.metrics.map50) - 12, "text-anchor": "middle", class: "value-label" }, root)
    .textContent = pct(last.metrics.map50);
  $("#chart-history").replaceChildren(root);
}

function renderTable(m) {
  const table = el("table");
  const head = el("tr");
  ["종류", "정답 수", "인식 수", "맞춘 수", "정밀도", "재현율", "F1", "AP50", "개수 오차"].forEach((h) => head.appendChild(el("th", h)));
  table.appendChild(head);
  classRows(m).forEach((r) => {
    const tr = el("tr");
    [r.name, r.gt, r.pred, r.tp, pct(r.precision), pct(r.recall), pct(r.f1), pct(r.ap50), num(r.count_mae)]
      .forEach((v) => tr.appendChild(el("td", String(v))));
    table.appendChild(tr);
  });
  $("#table").replaceChildren(table);
}

init();
