// DOM helpers. Text always goes through textContent: server data is never parsed as HTML.
import { label as labelOf } from "./format.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  applyAttrs(el, attrs);
  append(el, children);
  return el;
}

export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "style") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

function applyAttrs(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k in el && typeof v !== "string") el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el) {
  while (el.firstChild) el.firstChild.remove();
  return el;
}

// --- icons (inline SVG, stroke = currentColor) -------------------------------------------------------------------
const PATHS = {
  overview: "M3 12h4l3-8 4 16 3-8h4",
  documents: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h6",
  system: "M4 5h16v10H4zM8 19h8M12 15v4",
  ai: "M12 3a4 4 0 0 1 4 4v1a3 3 0 0 1 3 3v2a3 3 0 0 1-3 3v1a4 4 0 0 1-8 0v-1a3 3 0 0 1-3-3v-2a3 3 0 0 1 3-3V7a4 4 0 0 1 4-4zM9 12h6M12 9v6",
  integrations: "M8 7h8a3 3 0 0 1 0 6h-2M16 17H8a3 3 0 0 1 0-6h2",
  logs: "M5 4h14v16H5zM8 8h8M8 12h8M8 16h5",
  config: "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-2.7 1.1V21a2 2 0 1 1-4 0v-.1A1.6 1.6 0 0 0 7.1 19.8l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.6 1.6 0 0 0 3.2 14H3a2 2 0 1 1 0-4h.1A1.6 1.6 0 0 0 4.2 7.1l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.6 1.6 0 0 0 10 3.2V3a2 2 0 1 1 4 0v.1a1.6 1.6 0 0 0 2.7 1.1l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A1.6 1.6 0 0 0 20.8 10H21a2 2 0 1 1 0 4h-.1a1.6 1.6 0 0 0-1.5 1z",
  alerts: "M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.9 1.9 0 0 0 3.4 0",
  check: "M5 12l5 5L20 7",
  warn: "M12 3l10 18H2zM12 10v5M12 18v.5",
  x: "M6 6l12 12M18 6L6 18",
  info: "M12 8v.5M12 11v6M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z",
  minus: "M6 12h12",
};

export function icon(name, cls = "") {
  return s("svg", { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "1.8",
    "stroke-linecap": "round", "stroke-linejoin": "round", class: cls, "aria-hidden": "true" },
  s("path", { d: PATHS[name] || PATHS.info }));
}

// --- building blocks ---------------------------------------------------------------------------------------------
export function card(title, body, { sub, actions, cls = "" } = {}) {
  return h("section", { class: `card ${cls}` },
    title || actions ? h("div", { class: "card-head" },
      h("h2", null, title || "", sub ? h("span", { class: "sub" }, ` · ${sub}`) : null), actions || null) : null,
    body);
}

export function tile(labelText, value, { foot, hero = false, title } = {}) {
  return h("div", { class: `tile${hero ? " hero" : ""}`, title },
    h("span", { class: "label" }, labelText), h("span", { class: "value" }, value),
    foot ? h("span", { class: "foot" }, foot) : null);
}

const SEVERITY_ICON = { good: "check", warning: "warn", critical: "x", info: "info", neutral: "minus" };

export function badge(text, kind = "neutral", { plain = false, title } = {}) {
  return h("span", { class: `badge ${kind}${plain ? " plain" : ""}`, title }, text);
}

export function statusBadge(status) {
  const map = {
    done: ["Done", "good"], ok: ["OK", "good"], ready: ["Ready", "good"], firing: ["Firing", "critical"],
    failed: ["Failed", "critical"], critical: ["Critical", "critical"], rejected: ["Rejected", "warning"],
    queued: ["Queued", "info"], processing: ["Processing", "info"], deleted: ["Deleted", "neutral"],
    warning: ["Warning", "warning"], acknowledged: ["Acknowledged", "warning"], resolved: ["Resolved", "good"],
    disabled: ["Disabled", "neutral"], idle: ["Idle", "neutral"], missing: ["Missing", "critical"],
    stopped: ["Stopped", "neutral"], pending: ["Pending", "info"], delivered: ["Delivered", "good"],
  };
  const [text, kind] = map[status] || [labelOf(status || "unknown"), "neutral"];
  return badge(text, kind);
}

export function healthDot(ok, text, detail) {
  const kind = ok === true ? "good" : ok === false ? "critical" : "neutral";
  return h("div", { class: "dep", title: detail || "" },
    h("span", { class: `badge ${kind}` }, ok === true ? "Up" : ok === false ? "Down" : "n/a"), h("b", null, text));
}

export function kv(pairs) {
  const dl = h("dl", { class: "kv" });
  for (const [k, v] of pairs) {
    if (v === undefined) continue;
    dl.append(h("dt", null, k), h("dd", null, v === null ? "–" : v));
  }
  return dl;
}

export function empty(text = "Nothing here yet.") {
  return h("div", { class: "empty" }, text);
}

export function errorBox(e) {
  return h("div", { class: "banner critical", role: "alert" }, e && e.message ? e.message : String(e));
}

export function meter(labelText, valuePct, detail, { warn = 75, crit = 90 } = {}) {
  const v = valuePct === null || valuePct === undefined ? null : Math.max(0, Math.min(100, valuePct));
  const kind = v === null ? "" : v >= crit ? "critical" : v >= warn ? "warning" : "";
  return h("div", { class: `meter ${kind}` },
    h("div", { class: "row" }, h("span", null, labelText), h("span", { class: "num" }, detail)),
    h("div", { class: "track", role: "meter", "aria-valuenow": v ?? 0, "aria-valuemin": 0, "aria-valuemax": 100,
      "aria-label": labelText },
    h("div", { class: "fill", style: { width: `${v ?? 0}%` } })));
}

export function seg(options, value, onChange) {
  const el = h("div", { class: "seg", role: "group" });
  for (const [val, text] of options) {
    el.append(h("button", { type: "button", class: val === value ? "on" : "", "aria-pressed": String(val === value),
      onclick: () => onChange(val) }, text));
  }
  return el;
}

export function select(options, value, onChange, attrs = {}) {
  const el = h("select", { ...attrs, onchange: (e) => onChange(e.target.value) });
  for (const [val, text] of options) {
    const o = h("option", { value: val }, text);
    if (String(val) === String(value ?? "")) o.selected = true;
    el.append(o);
  }
  return el;
}

/**
 * columns: [{key, label, render(row) -> node|string, cls, sort(row) -> comparable}]
 */
export function table(columns, rows, { onRow, emptyText = "No rows." } = {}) {
  if (!rows || !rows.length) return empty(emptyText);
  let sortKey = null;
  let dir = 1;
  const tbody = h("tbody");
  const draw = () => {
    clear(tbody);
    let list = rows;
    const col = columns.find((c) => c.key === sortKey);
    if (col) {
      const get = col.sort || ((r) => r[col.key]);
      list = [...rows].sort((a, b) => {
        const x = get(a); const y = get(b);
        return (x === y ? 0 : x === null || x === undefined ? 1 : y === null || y === undefined ? -1 : x > y ? 1 : -1) * dir;
      });
    }
    for (const r of list) {
      const tr = h("tr", { class: onRow ? "click" : "", onclick: onRow ? () => onRow(r) : null, tabIndex: onRow ? 0 : null,
        onkeydown: onRow ? (e) => { if (e.key === "Enter") onRow(r); } : null });
      for (const c of columns) {
        const v = c.render ? c.render(r) : r[c.key];
        tr.append(h("td", { class: c.cls || "" }, v === null || v === undefined || v === "" ? "–" : v));
      }
      tbody.append(tr);
    }
  };
  const thead = h("thead", null, h("tr", null, columns.map((c) => h("th", {
    class: c.cls || "", scope: "col",
    onclick: c.nosort ? null : () => { dir = sortKey === c.key ? -dir : 1; sortKey = c.key; draw(); },
    title: c.nosort ? null : "Sort",
  }, c.label))));
  draw();
  return h("div", { class: "table-wrap" }, h("table", { class: "data" }, thead, tbody));
}

// --- overlays ----------------------------------------------------------------------------------------------------
let openDrawer = null;

export function drawer(title, body, { actions } = {}) {
  closeDrawer();
  const overlay = h("div", { class: "overlay", onclick: closeDrawer });
  const panel = h("aside", { class: "drawer", role: "dialog", "aria-modal": "true", "aria-label": title },
    h("div", { class: "drawer-head" }, h("h2", null, title), actions || null,
      h("button", { class: "btn ghost sm", type: "button", onclick: closeDrawer, "aria-label": "Close" }, icon("x", "sev-icon"))),
    h("div", { class: "drawer-body" }, body));
  const onKey = (e) => { if (e.key === "Escape") closeDrawer(); };
  document.addEventListener("keydown", onKey);
  document.body.append(overlay, panel);
  openDrawer = { overlay, panel, onKey };
  panel.querySelector("button")?.focus();
  return panel.querySelector(".drawer-body");
}

export function closeDrawer() {
  if (!openDrawer) return;
  openDrawer.overlay.remove();
  openDrawer.panel.remove();
  document.removeEventListener("keydown", openDrawer.onKey);
  openDrawer = null;
}

export function modal(title, body, actions, { wide = false } = {}) {
  const overlay = h("div", { class: "overlay modal-overlay" });
  const close = () => { overlay.remove(); box.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const box = h("div", { class: `modal${wide ? " wide" : ""}`, role: "dialog", "aria-modal": "true", "aria-label": title },
    h("h2", null, title), body, h("div", { class: "actions" }, (actions || []).map((a) => h("button", {
      type: "button", class: `btn ${a.cls || ""}`, onclick: async () => {
        if (a.onClick) {
          const keep = await a.onClick(close);
          if (keep === true) return;
        }
        close();
      },
    }, a.text))));
  overlay.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  document.body.append(overlay, box);
  (box.querySelector("input, textarea, select") || box.querySelector(".actions button:last-child"))?.focus();
  return { close, box };
}

export function confirm(title, text, { okText = "Confirm", danger = false } = {}) {
  return new Promise((resolve) => {
    modal(title, h("p", { class: "secondary" }, text), [
      { text: "Cancel", onClick: () => resolve(false) },
      { text: okText, cls: danger ? "danger" : "primary", onClick: () => resolve(true) },
    ]);
  });
}

/** Show a secret (API key, webhook secret) exactly once. */
export function showSecret(title, value, note) {
  const box = h("div", { class: "secret-box" }, value);
  modal(title, h("div", null,
    h("p", { class: "secondary" }, note || "Copy it now: it is shown only this once and is never stored in clear."),
    box), [
    { text: "Copy", onClick: async () => { try { await navigator.clipboard.writeText(value); toast("Copied", "", "good"); } catch { /* */ } return true; } },
    { text: "Done", cls: "primary" },
  ]);
}

export function toast(title, text = "", kind = "", onClick) {
  const box = document.getElementById("toasts");
  if (!box) return;
  const el = h("div", { class: `toast ${kind}`, role: "status", onclick: () => { el.remove(); onClick && onClick(); } },
    h("b", null, title), text);
  box.append(el);
  setTimeout(() => el.remove(), kind === "critical" ? 12000 : 6000);
}

export function loading() {
  return h("div", { class: "empty" }, "Loading…");
}

export function link(href, text) {
  return h("a", { href }, text);
}

/** Replace an element's children (null / false children are skipped, unlike Element.append). */
export function fill(el, ...children) {
  clear(el);
  append(el, children);
  return el;
}
