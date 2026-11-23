// Helpers shared by the views.
import { h, badge, statusBadge } from "../ui.js";
import { EVENT_LABELS, axisLabel, dateTime, short, time, ms, num } from "../format.js";

/** Continuous buckets from `since` to now: [{x, values: {key: n}}] using `pick(row) -> {key: n}` per series row. */
export function buckets(rows, since, bucketS, pick) {
  const start = Math.floor(since / bucketS) * bucketS;
  const end = Math.floor(Date.now() / 1000 / bucketS) * bucketS;
  const map = new Map();
  for (let t = start; t <= end; t += bucketS) map.set(t, { x: t, values: {} });
  for (const r of rows) {
    const b = map.get(r.t) || map.set(r.t, { x: r.t, values: {} }).get(r.t);
    for (const [k, v] of Object.entries(pick(r) || {})) b.values[k] = (b.values[k] || 0) + v;
  }
  return [...map.values()].sort((a, b) => a.x - b.x);
}

export function timeFormatter(spanS) {
  return (ts, full) => (full ? dateTime(ts) : axisLabel(ts, spanS));
}

export function throughputBuckets(series, since, bucketS) {
  return buckets(series, since, bucketS, (r) => ({
    "job.done": { done: r.n }, "extract.done": { done: r.n }, "job.failed": { failed: r.n },
    "request.error": { rejected: r.n },
  }[r.event]));
}

export const THROUGHPUT_KEYS = [
  { key: "done", name: "Completed", color: "var(--series-1)" },
  { key: "failed", name: "Failed", color: "var(--critical)" },
];

export function eventName(e) {
  return EVENT_LABELS[e] || e;
}

const EVENT_KIND = {
  "job.done": "good", "extract.done": "good", "webhook.delivered": "good", "job.failed": "critical",
  "webhook.failed": "critical", "request.error": "warning", "admin.login_failed": "warning", "job.requeued": "warning",
};

export function eventBadge(e) {
  return badge(eventName(e), EVENT_KIND[e] || (e.startsWith("admin.") ? "info" : "neutral"));
}

/** One line of facts for an audit event (metadata only). */
export function eventSummary(ev) {
  const d = ev.data || {};
  const parts = [];
  if (d.actor) parts.push(`by ${d.actor}`);
  if (d.endpoint) parts.push(d.endpoint);
  if (d.mimeType) parts.push(d.mimeType.replace("application/", "").replace("image/", ""));
  if (d.pages !== undefined) parts.push(`${d.pages} page${d.pages === 1 ? "" : "s"}`);
  if (d.ms !== undefined) parts.push(ms(d.ms));
  if (d.idDetection) parts.push(`${d.idDetection.docType} (${num(d.idDetection.confidence * 100, 0)}%)`);
  if (d.error) parts.push(`error ${d.error}`);
  if (d.code) parts.push(`${d.status} ${d.code}`);
  if (d.route && !d.code) parts.push(d.route);
  if (d.reason) parts.push(d.reason);
  if (d.attempts !== undefined && ev.event.startsWith("webhook")) parts.push(`${d.attempts} attempt(s)`);
  if (d.by) parts.push(`by ${d.by}`);
  if (d.kind) parts.push(d.kind);
  if (d.client) parts.push(`client ${d.client}`);
  if (d.fallbacks) parts.push(`fallback: ${d.fallbacks.join(", ")}`);
  return parts.join(" · ");
}

export function activityList(events, { onClick } = {}) {
  if (!events.length) return h("div", { class: "empty" }, "No activity in this period.");
  return h("div", { class: "table-wrap" }, h("table", { class: "data" }, h("tbody", null, events.map((ev) => h("tr", {
    class: onClick ? "click" : "", onclick: onClick ? () => onClick(ev) : null,
  }, h("td", { class: "nowrap muted num" }, time(ev.at)), h("td", null, eventBadge(ev.event)),
  h("td", { class: "secondary" }, ev.clientId || (ev.data?.actor ? "admin" : "–")),
  h("td", { class: "secondary" }, eventSummary(ev)))))));
}

export function docLink(ev) {
  if (ev.jobId) return `documents/${ev.jobId}`;
  if (["extract.done", "request.error"].includes(ev.event)) return `documents/evt-${ev.id}`;
  return null;
}

export function instanceState(i) {
  return statusBadge(i.state);
}

export function shortId(id) {
  return h("span", { class: "mono", title: id }, short(id, 10));
}

export function healthOk(health) {
  return health ? [health.store, health.ocrModels, health.worker].every(Boolean) : null;
}
