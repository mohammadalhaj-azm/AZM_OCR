// Small SVG charts: line/area (crosshair tooltip), stacked columns (per-column tooltip), sparkline, horizontal bars,
// segmented bar. Marks follow one spec: 2px lines, 10% area wash, <=24px columns with a 4px rounded end, a 2px
// surface gap between stacked segments, hairline grid, one y-axis, a legend for 2+ series, a table view.
import { h, s, clear, table, fill } from "./ui.js";
import { num } from "./format.js";

export const SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
  "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)"];

let tip = null;
function tooltip() {
  if (!tip) {
    tip = h("div", { class: "tooltip hidden", role: "tooltip" });
    document.body.append(tip);
  }
  return tip;
}

function showTip(evt, title, rows) {
  const t = tooltip();
  clear(t);
  t.append(h("div", { class: "t" }, title));
  for (const r of rows) {
    t.append(h("div", { class: "r" }, h("span", null, r.color ? h("i", { style: { background: r.color } }) : null, r.name),
      h("b", null, r.value)));
  }
  t.classList.remove("hidden");
  const pad = 14;
  const { innerWidth: W, innerHeight: H } = window;
  const rect = t.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + rect.width > W - 8) x = evt.clientX - rect.width - pad;
  if (y + rect.height > H - 8) y = evt.clientY - rect.height - pad;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${Math.max(8, y)}px`;
}

export function hideTip() {
  if (tip) tip.classList.add("hidden");
}

export function niceTicks(max, count = 4, integer = false) {
  if (!(max > 0)) return [0, 1];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  let step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => st >= raw) || raw;
  if (integer) step = Math.max(1, Math.ceil(step));
  const ticks = [];
  for (let v = 0; v <= max + step * 0.001; v += step) ticks.push(+v.toFixed(10));
  if (ticks[ticks.length - 1] < max) ticks.push(+(ticks[ticks.length - 1] + step).toFixed(10));
  return ticks;
}

function responsive(draw, height) {
  const el = h("div", { class: "chart", style: { height: `${height}px` } });
  let last = 0;
  const ro = new ResizeObserver((entries) => {
    const w = Math.floor(entries[0].contentRect.width);
    if (w > 0 && w !== last) {
      last = w;
      clear(el);
      el.append(draw(w));
    }
  });
  ro.observe(el);
  el.addEventListener("pointerleave", hideTip);
  return el;
}

export function legend(items) {
  return h("div", { class: "legend" }, items.map((it) => h("span", null,
    h("i", { class: it.line ? "line" : "", style: { background: it.color } }), it.name)));
}

/**
 * series: [{name, color, values: [y|null]}], x: [timestamps]. opts: {height, yFormat, xFormat, yMax, area}
 */
export function lineChart({ x, series, height = 200, yFormat = num, xFormat = String, yMax, area = false, integer = false }) {
  const wrap = h("div");
  if (series.length > 1) wrap.append(legend(series.map((sr) => ({ name: sr.name, color: sr.color, line: true }))));
  const allY = series.flatMap((sr) => sr.values).filter((v) => v !== null && v !== undefined);
  const ticks = niceTicks(yMax ?? Math.max(1e-9, ...allY, 0), 4, integer);
  const top = ticks[ticks.length - 1];
  wrap.append(responsive((W) => {
    const pl = 48, pr = 12, pt = 8, pb = 22;
    const pw = W - pl - pr, ph = height - pt - pb;
    const n = x.length;
    const xs = (i) => pl + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);
    const ys = (v) => pt + ph - (v / top) * ph;
    const svg = s("svg", { width: W, height, viewBox: `0 0 ${W} ${height}`, role: "img",
      "aria-label": series.map((sr) => sr.name).join(", ") });
    for (const t of ticks) {
      svg.append(s("line", { class: t === 0 ? "base-line" : "grid-line", x1: pl, x2: W - pr, y1: ys(t), y2: ys(t) }));
      svg.append(s("text", { x: pl - 8, y: ys(t) + 4, "text-anchor": "end" }, yFormat(t)));
    }
    const labels = Math.min(n, Math.max(2, Math.floor(pw / 110)));
    for (let k = 0; k < labels && n > 1; k++) {
      const i = Math.round((k / (labels - 1)) * (n - 1));
      svg.append(s("text", { x: xs(i), y: height - 6, "text-anchor": k === 0 ? "start" : k === labels - 1 ? "end" : "middle" },
        xFormat(x[i])));
    }
    for (const sr of series) {
      let d = "", fill = "", runStart = null, prev = null;
      sr.values.forEach((v, i) => {
        if (v === null || v === undefined) {
          if (area && runStart !== null) fill += `L${xs(prev)},${ys(0)}Z`;
          runStart = null;
          return;
        }
        if (runStart === null) {
          d += `M${xs(i)},${ys(v)}`;
          if (area) fill += `M${xs(i)},${ys(0)}L${xs(i)},${ys(v)}`;
          runStart = i;
        } else {
          d += `L${xs(i)},${ys(v)}`;
          if (area) fill += `L${xs(i)},${ys(v)}`;
        }
        prev = i;
      });
      if (area && runStart !== null) fill += `L${xs(prev)},${ys(0)}Z`;
      if (area && fill) svg.append(s("path", { d: fill, fill: sr.color, "fill-opacity": 0.1, stroke: "none" }));
      svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": 2, "stroke-linejoin": "round",
        "stroke-linecap": "round" }));
      const lastIdx = sr.values.map((v, i) => (v === null || v === undefined ? -1 : i)).filter((i) => i >= 0).pop();
      if (lastIdx !== undefined && lastIdx >= 0) {
        svg.append(s("circle", { cx: xs(lastIdx), cy: ys(sr.values[lastIdx]), r: 4, fill: sr.color,
          stroke: "var(--surface-1)", "stroke-width": 2 }));
      }
    }
    const cross = s("line", { class: "crosshair hidden", y1: pt, y2: pt + ph });
    const dots = s("g");
    svg.append(cross, dots);
    const hit = s("rect", { x: pl, y: pt, width: pw, height: ph, fill: "transparent" });
    hit.addEventListener("pointermove", (evt) => {
      const r = svg.getBoundingClientRect();
      const px = evt.clientX - r.left;
      const i = Math.max(0, Math.min(n - 1, Math.round(((px - pl) / Math.max(1, pw)) * (n - 1))));
      cross.setAttribute("x1", xs(i)); cross.setAttribute("x2", xs(i));
      cross.classList.remove("hidden");
      clear(dots);
      for (const sr of series) {
        const v = sr.values[i];
        if (v !== null && v !== undefined) {
          dots.append(s("circle", { cx: xs(i), cy: ys(v), r: 4, fill: sr.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
        }
      }
      showTip(evt, xFormat(x[i], true), series.map((sr) => ({ name: sr.name, color: sr.color,
        value: sr.values[i] === null || sr.values[i] === undefined ? "–" : yFormat(sr.values[i]) })));
    });
    hit.addEventListener("pointerleave", () => { cross.classList.add("hidden"); clear(dots); hideTip(); });
    svg.append(hit);
    return svg;
  }, height));
  return wrap;
}

/**
 * Stacked columns. buckets: [{x, values: {key: n}}], keys: [{key, name, color}].
 */
export function columnChart({ buckets, keys, height = 200, yFormat = num, xFormat = String }) {
  const wrap = h("div");
  if (keys.length > 1) wrap.append(legend(keys));
  const totals = buckets.map((b) => keys.reduce((a, k) => a + (b.values[k.key] || 0), 0));
  const ticks = niceTicks(Math.max(1, ...totals), 4, true);  // counts
  const top = ticks[ticks.length - 1];
  wrap.append(responsive((W) => {
    const pl = 44, pr = 8, pt = 8, pb = 22;
    const pw = W - pl - pr, ph = height - pt - pb;
    const n = Math.max(1, buckets.length);
    const band = pw / n;
    const bw = Math.max(2, Math.min(24, band * 0.7));
    const ys = (v) => (v / top) * ph;
    const svg = s("svg", { width: W, height, viewBox: `0 0 ${W} ${height}`, role: "img",
      "aria-label": keys.map((k) => k.name).join(", ") });
    for (const t of ticks) {
      const y = pt + ph - ys(t);
      svg.append(s("line", { class: t === 0 ? "base-line" : "grid-line", x1: pl, x2: W - pr, y1: y, y2: y }));
      svg.append(s("text", { x: pl - 8, y: y + 4, "text-anchor": "end" }, yFormat(t)));
    }
    const labels = Math.min(n, Math.max(2, Math.floor(pw / 110)));
    for (let k = 0; k < labels && n > 1; k++) {
      const i = Math.round((k / (labels - 1)) * (n - 1));
      svg.append(s("text", { x: pl + band * i + band / 2, y: height - 6,
        "text-anchor": k === 0 ? "start" : k === labels - 1 ? "end" : "middle" }, xFormat(buckets[i].x)));
    }
    buckets.forEach((b, i) => {
      const cx = pl + band * i + band / 2;
      const segs = keys.map((k) => ({ k, v: b.values[k.key] || 0 })).filter((sg) => sg.v > 0);
      let base = pt + ph;
      segs.forEach((sg, j) => {
        const full = ys(sg.v);
        const gap = j > 0 ? 2 : 0;
        const hgt = Math.max(1, full - gap);
        const y = base - full;
        const isTop = j === segs.length - 1;
        const r = isTop ? Math.min(4, hgt, bw / 2) : 0;
        const x0 = cx - bw / 2, x1 = cx + bw / 2, yb = base - gap;
        const d = r ? `M${x0},${yb}V${y + r}Q${x0},${y} ${x0 + r},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${yb}Z`
          : `M${x0},${yb}V${y}H${x1}V${yb}Z`;
        svg.append(s("path", { d, fill: sg.k.color }));
        base = y;
      });
      const hit = s("rect", { x: pl + band * i, y: pt, width: band, height: ph, fill: "transparent" });
      hit.addEventListener("pointermove", (evt) => showTip(evt, xFormat(b.x, true), [
        ...keys.map((k) => ({ name: k.name, color: k.color, value: yFormat(b.values[k.key] || 0) })),
        ...(keys.length > 1 ? [{ name: "Total", value: yFormat(totals[i]) }] : []),
      ]));
      hit.addEventListener("pointerleave", hideTip);
      svg.append(hit);
    });
    return svg;
  }, height));
  return wrap;
}

export function sparkline(values, { color = "var(--series-1)", height = 28, width = 120 } = {}) {
  const vals = values.filter((v) => v !== null && v !== undefined);
  const max = Math.max(1e-9, ...vals);
  const n = values.length;
  let d = "";
  values.forEach((v, i) => {
    if (v === null || v === undefined) return;
    const x = n <= 1 ? width / 2 : (i / (n - 1)) * (width - 4) + 2;
    const y = height - 3 - (v / max) * (height - 6);
    d += `${d ? "L" : "M"}${x},${y}`;
  });
  return s("svg", { width, height, viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" },
    s("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
}

/** items: [{name, value, display}] — one series, so no legend; values at the bar ends. */
export function hbars(items, { color = "var(--series-1)", format = num, max } = {}) {
  const top = max ?? Math.max(1e-9, ...items.map((i) => i.value || 0));
  return h("div", { class: "hbars" }, items.map((it) => h("div", { class: "hbar", title: `${it.name}: ${it.display ?? format(it.value)}` },
    h("span", { class: "name" }, it.name),
    h("div", null, h("div", { class: "bar", style: { width: `${Math.max(0.5, ((it.value || 0) / top) * 100)}%`, background: it.color || color } })),
    h("span", { class: "val" }, it.display ?? format(it.value)))));
}

/** One whole split into parts (thin segmented bar) with a legend listing name and share. */
export function segbar(parts) {
  const total = parts.reduce((a, p) => a + (p.value || 0), 0);
  if (!total) return h("div", { class: "muted" }, "No data in this period.");
  const bar = h("div", { class: "segbar", role: "img",
    "aria-label": parts.map((p) => `${p.name} ${Math.round((p.value / total) * 100)}%`).join(", ") },
  parts.filter((p) => p.value > 0).map((p) => h("i", { style: { flex: String(p.value), background: p.color },
    title: `${p.name}: ${p.value} (${Math.round((p.value / total) * 100)}%)` })));
  return h("div", null, bar, h("div", { class: "legend" }, parts.map((p) => h("span", null,
    h("i", { style: { background: p.color } }), `${p.name} · ${num(p.value, 0)} (${Math.round(((p.value || 0) / total) * 100)}%)`))));
}

/** A chart with a "Table" toggle: the same data as rows (accessibility, exact values). */
export function withTable(chartEl, columns, rows) {
  const holder = h("div", null, chartEl);
  let showing = false;
  const button = h("button", { type: "button", class: "btn ghost sm view-toggle", onclick: () => {
    showing = !showing;
    fill(holder, showing ? table(columns, rows) : chartEl);
    button.textContent = showing ? "Chart" : "Table";
  } }, "Table");
  return { body: holder, toggle: button };
}
