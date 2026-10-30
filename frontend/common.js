"use strict";
// 메인 화면(app.js)과 테스트·라벨 화면(label.js)이 함께 쓰는 도구

const MAX_SIDE = 1568; // 업로드 전 축소: 인식 품질은 유지하면서 전송량·비용을 줄인다
const COLORS = ["#e53935", "#1e88e5", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#f4511e", "#3949ab", "#c0ca33", "#6d4c41", "#d81b60", "#546e7a"];

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

async function loadNuts() {
  const res = await fetch("/api/nuts");
  const data = await res.json();
  const nuts = {};
  data.nuts.forEach((nut, i) => { nuts[nut.id] = { ...nut, color: COLORS[i % COLORS.length] }; });
  return { nuts, dailyG: data.daily_recommended_g, list: data.nuts };
}
