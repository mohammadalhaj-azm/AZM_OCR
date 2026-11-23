// Formatting helpers. Everything renders through textContent (ui.js), so nothing here needs escaping.

const nf = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const compact = new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 });

export const dash = "–";

export function num(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return dash;
  if (digits === 0) return nf0.format(v);
  return Math.abs(v) >= 10000 ? compact.format(v) : nf.format(v);
}

export function int(v) {
  return v === null || v === undefined ? dash : (Math.abs(v) >= 100000 ? compact.format(v) : nf0.format(v));
}

export function pct(v, digits = 1) {
  if (v === null || v === undefined) return dash;
  return `${Number(v).toFixed(digits).replace(/\.0+$/, "")}%`;
}

export function ratio(v) {
  return v === null || v === undefined ? dash : pct(v * 100);
}

export function bytes(v) {
  if (v === null || v === undefined) return dash;
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let x = Math.abs(v);
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i++; }
  return `${x >= 100 || i === 0 ? Math.round(x) : x.toFixed(1)} ${units[i]}`;
}

export function rate(v) {
  return v === null || v === undefined ? dash : `${bytes(v)}/s`;
}

export function ms(v) {
  if (v === null || v === undefined) return dash;
  if (v < 1000) return `${Math.round(v)} ms`;
  if (v < 60000) return `${(v / 1000).toFixed(v < 10000 ? 2 : 1)} s`;
  return duration(v / 1000);
}

export function duration(s) {
  if (s === null || s === undefined) return dash;
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${s % 60 ? `${s % 60} s` : ""}`.trim();
  const h = Math.floor(m / 60);
  if (h < 48) return `${h} h ${m % 60 ? `${m % 60} min` : ""}`.trim();
  return `${Math.floor(h / 24)} d ${h % 24 ? `${h % 24} h` : ""}`.trim();
}

const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const dateTimeFmt = new Intl.DateTimeFormat(undefined, {
  month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
});
const axisDay = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" });
const axisTime = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" });

export function time(ts) {
  return ts ? timeFmt.format(new Date(ts * 1000)) : dash;
}

export function dateTime(ts) {
  return ts ? dateTimeFmt.format(new Date(ts * 1000)) : dash;
}

export function axisLabel(ts, spanS) {
  const d = new Date(ts * 1000);
  return spanS > 2 * 86400 ? axisDay.format(d) : axisTime.format(d);
}

export function ago(ts) {
  if (!ts) return dash;
  const s = Date.now() / 1000 - ts;
  if (s < 5) return "just now";
  if (s < 60) return `${Math.round(s)} s ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function short(id, n = 8) {
  return id ? (id.length > n + 2 ? `${id.slice(0, n)}…` : id) : dash;
}

export function label(key) {
  // snake / camel case -> "Sentence case"
  const s = String(key).replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").toLowerCase();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export const DOC_TYPE_LABELS = {
  national_id: "National ID", passport: "Passport", driver_license: "Driver license",
  residence_permit: "Residence permit", other_id: "Other ID", not_id: "Not an ID",
};

export const EVENT_LABELS = {
  "job.received": "Received", "job.replayed": "Replayed (idempotent)", "job.started": "Processing started",
  "job.done": "Completed", "job.failed": "Failed", "job.requeued": "Re-queued", "job.fetched": "Result fetched",
  "job.deleted": "Deleted", "webhook.delivered": "Webhook delivered", "webhook.failed": "Webhook given up",
  "extract.done": "Text extracted (sync)", "request.error": "Request rejected",
  "admin.login": "Admin login", "admin.login_failed": "Admin login failed", "admin.logout": "Admin logout",
  "admin.config_saved": "Configuration saved", "admin.key_created": "API key created",
  "admin.key_revoked": "API key revoked", "admin.secret_created": "Webhook secret generated",
  "admin.job_deleted": "Job deleted by admin", "admin.alert_acked": "Alert acknowledged",
};
