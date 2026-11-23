// Shell: navigation, hash routing, theme, range picker, and the live stream (SSE, polling fallback).
import { api } from "./api.js";
import { h, clear, icon, seg, toast, errorBox, loading, closeDrawer, fill } from "./ui.js";
import { hideTip } from "./charts.js";
import * as overview from "./views/overview.js";
import * as documents from "./views/documents.js";
import * as system from "./views/system.js";
import * as ai from "./views/ai.js";
import * as integrations from "./views/integrations.js";
import * as logs from "./views/logs.js";
import * as config from "./views/config.js";
import * as alerts from "./views/alerts.js";

const ROUTES = [
  { id: "overview", title: "Overview", icon: "overview", view: overview },
  { id: "documents", title: "Documents", icon: "documents", view: documents },
  { id: "system", title: "System resources", icon: "system", view: system, nav: "System" },
  { id: "ai", title: "AI monitoring", icon: "ai", view: ai, nav: "AI models" },
  { id: "integrations", title: "Integrations", icon: "integrations", view: integrations },
  { id: "logs", title: "Logs & audit", icon: "logs", view: logs },
  { id: "config", title: "Configuration", icon: "config", view: config },
  { id: "alerts", title: "Alerts", icon: "alerts", view: alerts },
];
const RANGE_LABELS = { "15m": "15m", "1h": "1h", "6h": "6h", "24h": "24h", "7d": "7d", "30d": "30d" };

const store = {
  get(key, fallback) { try { return localStorage.getItem(`ocr-admin-${key}`) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(`ocr-admin-${key}`, value); } catch { /* storage unavailable */ } },
};

const state = { me: null, live: null, listeners: new Set(), cleanup: null, seenAlerts: null, renderId: 0 };

// --- theme -------------------------------------------------------------------------------------------------------
function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  document.getElementById("theme").textContent = { light: "Light", dark: "Dark" }[theme] || "Auto";
}
applyTheme(store.get("theme", "auto"));
document.getElementById("theme").addEventListener("click", () => {
  const next = { auto: "light", light: "dark", dark: "auto" }[store.get("theme", "auto")] || "auto";
  store.set("theme", next);
  applyTheme(next);
  render();  // charts pick up the new colors
});

// --- routing -----------------------------------------------------------------------------------------------------
export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const [id, ...rest] = (path || "overview").split("/");
  return { id: id || "overview", rest: rest.map(decodeURIComponent), params: Object.fromEntries(new URLSearchParams(query || "")) };
}

export function navigate(path, params) {
  const q = new URLSearchParams(Object.entries(params || {}).filter(([, v]) => v !== undefined && v !== null && v !== ""));
  const target = `#/${path}${q.toString() ? `?${q}` : ""}`;
  if (location.hash === target) render(); else location.hash = target;
}

function buildNav() {
  const nav = document.getElementById("nav");
  clear(nav);
  for (const r of ROUTES) {
    nav.append(h("a", { href: `#/${r.id}`, dataset: { id: r.id } }, icon(r.icon), h("span", { class: "label" }, r.nav || r.title),
      r.id === "alerts" ? h("span", { class: "count hidden", id: "alert-count" }, "0") : null));
  }
}

async function render() {
  const { id, rest, params } = parseHash();
  const route = ROUTES.find((r) => r.id === id) || ROUTES[0];
  const renderId = ++state.renderId;
  if (state.cleanup) { try { state.cleanup(); } catch { /* */ } state.cleanup = null; }
  state.listeners.clear();
  closeDrawer();
  hideTip();
  for (const a of document.querySelectorAll("#nav a")) a.classList.toggle("active", a.dataset.id === route.id);
  document.getElementById("title").textContent = route.title;
  document.title = `${route.title} · OCR Admin`;
  const controls = clear(document.getElementById("top-controls"));
  const root = clear(document.getElementById("content"));
  const loader = loading();
  root.append(loader);
  const ctx = {
    me: state.me, params, rest, controls,
    isAdmin: state.me?.user?.role === "admin",
    navigate,
    stale: () => renderId !== state.renderId,
    range: (fallback = "24h") => params.range || store.get(`range-${route.id}`, fallback),
    rangePicker(fallback = "24h", options = Object.keys(RANGE_LABELS)) {
      const current = ctx.range(fallback);
      controls.append(seg(options.map((o) => [o, RANGE_LABELS[o]]), current, (v) => {
        store.set(`range-${route.id}`, v);
        navigate([route.id, ...rest].join("/"), { ...params, range: v });
      }));
      return current;
    },
    onLive(fn) { state.listeners.add(fn); if (state.live) fn(state.live); },
    rerender: render,
  };
  try {
    const cleanup = await route.view.render(root, ctx);
    loader.remove();
    if (!ctx.stale()) state.cleanup = typeof cleanup === "function" ? cleanup : null;
  } catch (e) {
    if (ctx.stale()) return;
    fill(root, errorBox(e));
  }
}

// --- live stream -------------------------------------------------------------------------------------------------
let source = null;
let pollTimer = null;
let lastTick = 0;

function onTick(snap) {
  lastTick = Date.now();
  state.live = snap;
  const dot = document.getElementById("live");
  dot.className = "live-dot on";
  dot.textContent = "Live";
  const count = document.getElementById("alert-count");
  if (count) {
    count.textContent = String(snap.alerts.open);
    count.classList.toggle("hidden", !snap.alerts.open);
  }
  const seen = state.seenAlerts;
  const ids = new Set(snap.alerts.latest.map((a) => a.id));
  if (seen) {
    for (const a of snap.alerts.latest) {
      if (!seen.has(a.id) && a.state === "firing") {
        toast(a.title, a.summary, a.severity === "critical" ? "critical" : "warning", () => navigate("alerts"));
      }
    }
  }
  state.seenAlerts = new Set([...(seen || []), ...ids]);
  for (const fn of state.listeners) {
    try { fn(snap); } catch (e) { console.error(e); }
  }
}

function liveError() {
  const dot = document.getElementById("live");
  dot.className = "live-dot err";
  dot.textContent = "Reconnecting";
}

function connect() {
  if (typeof EventSource === "undefined") return startPolling();
  source = new EventSource("/admin/api/stream");
  source.addEventListener("tick", (e) => onTick(JSON.parse(e.data)));
  source.addEventListener("logout", () => { location.href = "/admin/login"; });
  source.addEventListener("error", () => {
    if (source.readyState === EventSource.CLOSED) liveError();
  });
  // If no tick arrives for a while (proxy buffering SSE, ...), poll instead.
  setInterval(() => {
    if (Date.now() - lastTick > 15000 && !pollTimer) startPolling();
  }, 5000);
}

function startPolling() {
  if (source) { source.close(); source = null; }
  const every = Math.max(2000, (state.me?.sampleIntervalS || 2) * 1000);
  const poll = async () => {
    try { onTick(await api.get("/live")); } catch { liveError(); }
  };
  poll();
  pollTimer = setInterval(poll, every);
}

// --- start -------------------------------------------------------------------------------------------------------
async function start() {
  buildNav();
  try {
    state.me = await api.get("/me");
  } catch (e) {
    fill(document.getElementById("content"), errorBox(e));
    return;
  }
  const { user, version, environment, instance } = state.me;
  document.getElementById("who").textContent = `${user.name} · ${user.role}`;
  document.getElementById("version").textContent = `v${version}`;
  document.getElementById("env").textContent = `${environment}${instance ? ` · ${instance.split("-")[0]}` : ""}`;
  document.getElementById("logout").addEventListener("click", async () => {
    try { await api.post("/logout"); } finally { location.href = "/admin/login"; }
  });
  window.addEventListener("hashchange", render);
  connect();
  render();
}

start();
