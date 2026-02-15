/* FK8 LogTool Service Worker (minimal) */
const CACHE_NAME = "fk8-logtool-v1";
const APP_SHELL = [
  "/",
  "/static/manifest.json",
  "/static/icon-192.png",
  "/static/icon-512.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => (k !== CACHE_NAME ? caches.delete(k) : null)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  // Network-first for dynamic content (analysis results, plots)
  if (req.method !== "GET") return;
  const url = new URL(req.url);

  if (url.pathname.startsWith("/runs/") || url.pathname.startsWith("/analyze")) {
    event.respondWith(
      fetch(req).catch(() => caches.match(req))
    );
    return;
  }

  // Cache-first for app shell
  event.respondWith(
    caches.match(req).then((cached) => cached || fetch(req))
  );
});
