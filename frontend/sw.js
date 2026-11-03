// 오프라인 지원: 앱 화면·휴대폰용 모델·인식 엔진을 저장해 두고 인터넷 없이도 열리게 한다.
// - 화면 파일, 모델: 저장본을 바로 쓰고 뒤에서 새 버전 받아 두기 (다음 실행에 반영)
// - 인식 엔진(CDN, 버전 고정): 저장본 우선 (처음 한 번은 인터넷 필요)
// - /api/nuts, /api/config: 인터넷 우선, 안 되면 저장본
// - 그 외 API(인식·저장): 저장하지 않음
const CACHE = "nuts-v1";
const SHELL = ["/", "/static/style.css", "/static/common.js", "/static/ondevice.js", "/static/app.js", "/static/manifest.webmanifest"];
const OFFLINE_API = ["/api/nuts", "/api/config"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

async function cacheFirst(request) {
  const hit = await caches.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  if (res.ok) (await caches.open(CACHE)).put(request, res.clone());
  return res;
}

async function staleWhileRevalidate(request, event) {
  const hit = await caches.match(request);
  const update = fetch(request).then(async (res) => {
    if (res.ok) await (await caches.open(CACHE)).put(request, res.clone());
    return res;
  });
  if (hit) {
    event.waitUntil(update.catch(() => {}));
    return hit;
  }
  return update;
}

async function networkFirst(request) {
  try {
    const res = await fetch(request);
    if (res.ok) (await caches.open(CACHE)).put(request, res.clone());
    return res;
  } catch {
    const hit = await caches.match(request);
    if (hit) return hit;
    throw new Error("offline");
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin === location.origin && OFFLINE_API.includes(url.pathname)) {
    event.respondWith(networkFirst(request));
  } else if (url.origin === location.origin && url.pathname.startsWith("/api/")) {
    return;
  } else if (url.origin === location.origin && url.pathname === "/") {
    event.respondWith(networkFirst(request));
  } else if (url.origin === location.origin && (url.pathname.startsWith("/static/") || url.pathname === "/models/nuts.onnx")) {
    event.respondWith(staleWhileRevalidate(request, event));
  } else if (url.hostname === "cdn.jsdelivr.net") {
    event.respondWith(cacheFirst(request));
  }
});
