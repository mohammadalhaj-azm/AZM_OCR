// AI monitoring: OCR engines, text and vision LLMs (status, memory/VRAM, calls, errors), inference times,
// throughput, decisions, fallbacks and measured accuracy.
import { api } from "../api.js";
import { h, clear, card, kv, badge, table, tile, empty } from "../ui.js";
import { segbar, hbars, columnChart, lineChart, withTable, SERIES } from "../charts.js";
import { ms, int, num, bytes, ago, dateTime, DOC_TYPE_LABELS, pct, dash } from "../format.js";
import { buckets, timeFormatter } from "./common.js";

const share = (v) => (v === null || v === undefined ? dash : pct(v * 100));

function modelStatus(m) {
  if (!m.enabled) return badge("Disabled", "neutral");
  if (!m.reachable) return badge("Server unreachable", "critical");
  if (!m.installed) return badge("Not installed", "critical");
  if (m.loaded) return badge("Loaded", "good");
  return badge(m.loaded === false ? "Ready (not in memory)" : "Ready", "good");
}

function llmCard(m) {
  const c = m.calls;
  return card(`${m.role === "text" ? "Text" : "Vision"} LLM`, h("div", { class: "grid" },
    h("div", { class: "filters" }, modelStatus(m), h("span", { class: "mono" }, m.model)),
    kv([["Endpoint", m.endpoint], ["Backend", m.backend],
      ["In memory", m.memory ? `${bytes(m.memory.sizeBytes)}${m.memory.parameterSize ? ` · ${m.memory.parameterSize}` : ""}${m.memory.quantization ? ` · ${m.memory.quantization}` : ""}` : (m.loaded === null ? "unknown (not Ollama)" : "no")],
      ["On GPU (VRAM)", m.vramPercent !== null && m.vramPercent !== undefined ? `${pct(m.vramPercent)} of the model (${bytes(m.memory?.vramBytes)})` : dash],
      ["Unloads", m.memory?.expiresAt ? dateTime(Date.parse(m.memory.expiresAt) / 1000) : dash],
      ["Calls (this replica)", c ? `${int(c.calls)} · ${int(c.ok)} ok · ${int(c.errors)} errors · ${int(c.retried)} retried` : "none since start"],
      ["In flight", c ? int(c.inFlight) : "0"],
      ["Response time", c && c.latency.n ? `p50 ${ms(c.latency.p50)} · p95 ${ms(c.latency.p95)} (last ${c.latency.n})` : dash],
      ["Last error", c?.lastError ? h("span", { class: "level-ERROR" }, `${c.lastError} · ${ago(c.lastErrorAt)}`) : "none"],
    ])));
}

export async function render(root, ctx) {
  const range = ctx.rangePicker("24h");
  const draw = (d) => {
    clear(root);
    const o = d.ocr;
    const pool = o.pool;
    root.append(h("div", { class: "grid cols-3" },
      card("OCR engine (PaddleOCR PP-OCRv5)", h("div", { class: "grid" },
        h("div", { class: "filters" }, badge(o.modelsPresent ? "Models present" : "Models missing", o.modelsPresent ? "good" : "critical"),
          pool ? badge(`${pool.busy} of ${pool.engines} busy`, pool.busy ? "info" : "neutral") : null),
        kv([["Detection", h("span", { class: "mono" }, o.models.detection)],
          ["Recognition (Arabic)", h("span", { class: "mono" }, o.models.recognitionAr)],
          ["Recognition (English)", o.models.recognitionEn ? h("span", { class: "mono" }, o.models.recognitionEn) : "off"],
          ["Languages", o.languages.join(", ")],
          ["Engines", pool ? `${pool.engines} (${pool.loaded} loaded in RAM)${pool.cpuThreads && !(pool.device || "").startsWith("gpu") ? ` · ${pool.cpuThreads} threads each` : ""}` : dash],
          ["Device", pool ? (pool.device ? h("span", { class: "mono" }, pool.device) : "not loaded yet") : dash],
          ["Parallel AR/EN", pool ? (pool.parallelRec ? "yes" : "no") : dash],
          ["DPI · min score", `${o.dpi} · ${o.minScore}`], ["Preprocessing", o.preprocess ? "deskew + contrast" : "off"]]))),
      ...d.llm.models.map(llmCard)));
    if (d.llm.otherLoaded.length) {
      root.append(h("div", { class: "banner info" }, `Also loaded on the LLM server: ${d.llm.otherLoaded.map((m) => `${m.name} (${bytes(m.sizeBytes)})`).join(", ")}.`));
    }

    const p = d.performance;
    root.append(h("div", { class: "tiles" },
      tile("Documents per hour", num(p.throughput.documentsPerHour), { foot: `average over the last ${range}` }),
      tile("Pages per hour", num(p.throughput.pagesPerHour)),
      tile("OCR per page (p50)", ms(p.stages.ocrPerPage.p50), { foot: `p95 ${ms(p.stages.ocrPerPage.p95)}` }),
      tile("Text LLM (p50)", ms(p.stages.llm.p50), { foot: `p95 ${ms(p.stages.llm.p95)} · ${int(p.stages.llm.n)} calls` }),
      tile("Vision LLM (p50)", ms(p.stages.vision.p50), { foot: `p95 ${ms(p.stages.vision.p95)} · ${int(p.stages.vision.n)} calls` }),
      tile("Fallbacks", int(Object.values(d.decisions.fallbacks).reduce((a, b) => a + b, 0)),
        { foot: Object.entries(d.decisions.fallbacks).map(([k, n]) => `${k} ${n}×`).join(" · ") || "every model answered" }),
    ));

    const stageRows = [
      ["OCR per page", p.stages.ocrPerPage], ["Extraction (per document)", p.stages.extract], ["Wait for LLM thread", p.stages.llmQueue],
      ["Text LLM", p.stages.llm], ["Vision LLM", p.stages.vision], ["Async job, total", p.durations.job],
      ["Sync extract, total", p.durations.extract],
    ].map(([name, st]) => ({ name, ...st }));
    const maxP95 = Math.max(1, ...stageRows.map((r) => r.p95 || 0));
    const timing = table([
      { key: "name", label: "Stage" },
      { key: "p50", label: "p50", cls: "r nowrap", render: (r) => ms(r.p50) },
      { key: "p95", label: "p95", cls: "r nowrap", render: (r) => ms(r.p95) },
      { key: "bar", label: "p95 relative", nosort: true, cls: "barcell", render: (r) => h("div", { class: "inline-bar", title: `p95 ${ms(r.p95)}` },
        h("i", { style: { width: `${((r.p95 || 0) / maxP95) * 100}%` } })) },
      { key: "avg", label: "Average", cls: "r nowrap", render: (r) => ms(r.avg) },
      { key: "max", label: "Max", cls: "r nowrap", render: (r) => ms(r.max) },
      { key: "n", label: "Count", cls: "r", render: (r) => int(r.n) },
    ], stageRows);

    const since = Date.now() / 1000 - ({ "15m": 900, "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000 }[range] || 86400);
    const jobMs = buckets(d.series, since, d.bucketS, (r) => (r.event === "job.done" && r.ms !== null ? { ms: r.ms } : null));
    const x = jobMs.map((b) => b.x);
    const fmt = timeFormatter(Date.now() / 1000 - since);
    const lc = withTable(lineChart({ x, series: [{ name: "Average job time", color: SERIES[0], values: jobMs.map((b) => b.values.ms ?? null) }],
      yFormat: ms, xFormat: fmt, height: 180 }), [{ key: "x", label: "Period", render: (r) => dateTime(r.x) },
      { key: "ms", label: "Average job time", cls: "r", render: (r) => ms(r.values.ms) }], jobMs.filter((b) => b.values.ms));
    root.append(h("div", { class: "grid cols-3" },
      card("Inference and processing times", timing, { cls: "flush span-2", sub: `last ${range}, all replicas` }),
      card("Average job time", lc.body, { actions: lc.toggle })));

    const dec = d.decisions;
    const by = dec.decidedBy;
    const conf = Array.from({ length: 10 }, (_, i) => ({ x: i, values: { n: dec.confidence[String(i + 1)] || 0 } }));
    const types = Object.entries(dec.docType).map(([t, n]) => ({ name: DOC_TYPE_LABELS[t] || t, value: n })).sort((a, b) => b.value - a.value);
    root.append(h("div", { class: "grid cols-3" },
      card("Who decided", h("div", null,
        segbar([{ name: "Rules only", value: by.rules || 0, color: SERIES[0] }, { name: "Text LLM", value: by.llm || 0, color: SERIES[1] },
          { name: "Vision LLM", value: by.vision || 0, color: SERIES[2] }]),
        h("p", { class: "muted" }, "The last stage that decided each ID detection. A valid MRZ, or a long document without ID signals, is decided by rules alone."),
        h("h3", null, "Pages"),
        segbar([{ name: "Text layer", value: dec.pageMethods.text_layer || 0, color: SERIES[0] },
          { name: "OCR", value: dec.pageMethods.ocr || 0, color: SERIES[1] }]))),
      card("Document types", types.length ? hbars(types, { format: int }) : empty("No ID decisions in this period."), { sub: `${int(dec.ids)} identity documents` }),
      card("Confidence of decisions", columnChart({ buckets: conf, keys: [{ key: "n", name: "Decisions", color: SERIES[0] }], height: 170,
        xFormat: (i) => `${i * 10}–${i * 10 + 10}%`, yFormat: (v) => num(v, 0) }))));

    const a = d.accuracy;
    root.append(card("Accuracy (labeled evaluation)", a ? h("div", { class: "grid cols-2" },
      kv([["Run", `${a.run} · ${dateTime(a.at)}`], ["Documents", `${int(a.documents)} (${int(a.labeled)} labeled, ${int(a.errors)} errors)`],
        ["Accuracy (ID vs not ID)", share(a.binary?.accuracy)],
        ["Precision · recall · F1", a.binary ? `${share(a.binary.precision)} · ${share(a.binary.recall)} · ${share(a.binary.f1)}` : dash],
        ["Document type accuracy", share(a.docType?.accuracy)],
        ["OCR per page (eval)", a.ocrMsPerPage ? `median ${ms(a.ocrMsPerPage.median)} · p95 ${ms(a.ocrMsPerPage.p95)}` : dash]]),
      kv(Object.entries(a.settings || {}).map(([k, v]) => [k, String(v)])))
      : h("div", null, empty("No evaluation run found. Accuracy is measured on your own labeled documents:"),
        h("pre", { class: "secret-box" }, "uv run python -m eval.run_eval D:\\docs --labels D:\\docs\\labels.csv")),
    { sub: "from the latest eval.run_eval summary" }));
  };

  draw(await api.get("/ai", { range }));
  const timer = setInterval(async () => {
    try { const d = await api.get("/ai", { range }); if (!ctx.stale()) draw(d); } catch { /* keep */ }
  }, 15000);
  return () => clearInterval(timer);
}
