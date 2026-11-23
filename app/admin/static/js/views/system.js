// System resources: CPU, memory, disks, I/O, GPUs and health per replica, live (this replica every
// SAMPLE_INTERVAL_S) and as history (stored samples).
import { api } from "../api.js";
import { h, clear, card, meter, table, select, kv, healthDot, empty, badge } from "../ui.js";
import { lineChart, withTable, SERIES } from "../charts.js";
import { bytes, rate, pct, duration, int, dateTime, ago, num } from "../format.js";
import { timeFormatter, instanceState } from "./common.js";

export async function render(root, ctx) {
  const range = ctx.rangePicker("15m", ["15m", "1h", "6h", "24h"]);
  let sys = await api.get("/system");
  if (ctx.stale()) return;
  const chosen = ctx.params.instance || sys.self || sys.instances[0]?.instance;
  if (sys.instances.length > 1) {
    ctx.controls.prepend(select(sys.instances.map((i) => [i.instance, `${i.host} · ${i.instance.split("-")[0]}${i.self ? " (this)" : ""}`]),
      chosen, (v) => ctx.navigate("system", { ...ctx.params, instance: v })));
  }
  const live = h("div", { class: "grid cols-2" });
  const charts = h("div", { class: "grid cols-2" });
  const bottom = h("div", { class: "grid cols-2" });
  root.append(live, charts, bottom);

  const drawLive = (i) => {
    clear(live);
    if (!i) { live.append(empty("This replica hasn't reported a sample yet.")); return; }
    const m = i.memory || {};
    const cores = (i.cpu?.perCore || []).map((v) => h("div", { class: "core", title: `${num(v)}%` },
      h("i", { style: { height: `${Math.min(100, v)}%` } })));
    live.append(card(`${i.host || "replica"}`, h("div", { class: "grid" },
      h("div", { class: "filters" }, instanceState(i), badge(`v${i.version}`, "neutral", { plain: true }),
        h("span", { class: "muted" }, `pid ${i.pid} · up ${duration(i.uptimeS)} · sampled ${ago(i.at)}`)),
      meter("CPU", i.cpu?.percent, `${pct(i.cpu?.percent)} of ${i.cpu?.count} cores${i.cpu?.load ? ` · load ${i.cpu.load.join(" / ")}` : ""}`),
      h("div", { class: "cores", "aria-label": "Per-core usage" }, cores),
      meter("Memory", m.percent, `${bytes(m.used)} of ${bytes(m.total)} (${pct(m.percent)})`),
      meter("Swap", m.swapPercent, `${bytes(m.swapUsed)} of ${bytes(m.swapTotal)}`),
      meter("This service (RSS)", m.total ? (m.processRss / m.total) * 100 : null, `${bytes(m.processRss)} · ${int(m.processThreads)} threads · CPU ${pct(i.cpu?.process)}`),
      ...(i.disks || []).map((d) => meter(`Disk: ${d.label}`, d.percent, `${bytes(d.free)} free of ${bytes(d.total)}`)),
    ), { sub: i.instance }));
    const hl = i.health || {};
    const gpus = i.gpus || [];
    live.append(h("div", { class: "grid" },
      card("Health", h("div", { class: "health-strip" }, healthDot(hl.store, "Database"), healthDot(hl.ocrModels, "OCR models"),
        healthDot(hl.worker, "Job worker"), healthDot(hl.llm, "Text LLM"), healthDot(hl.vision, "Vision LLM"))),
      card("I/O", kv([["Network out", rate(i.io?.netSentBps)], ["Network in", rate(i.io?.netRecvBps)],
        ["Disk read", rate(i.io?.diskReadBps)], ["Disk write", rate(i.io?.diskWriteBps)],
        ["Jobs on this replica", int(i.work?.activeJobs)],
        ["OCR engines busy", i.work?.ocr ? `${i.work.ocr.busy} of ${i.work.ocr.engines} (${i.work.ocr.loaded} loaded)` : "–"],
        ["LLM calls in flight", int(i.work?.llmInFlight)]])),
      card("GPUs", gpus.length ? h("div", { class: "grid" }, gpus.map((g) => g.error ? h("div", null, `GPU ${g.index}: ${g.error}`) : h("div", { class: "grid" },
        h("b", null, `GPU ${g.index} · ${g.name}`),
        meter("Utilization", g.util, pct(g.util)),
        meter("VRAM", g.memPercent, `${bytes(g.memUsed)} of ${bytes(g.memTotal)}`),
        h("div", { class: "muted" }, [g.temperature !== null ? `${g.temperature} °C` : null, g.powerW !== null ? `${g.powerW} W${g.powerLimitW ? ` of ${g.powerLimitW} W` : ""}` : null].filter(Boolean).join(" · ") || " ")))) :
        empty(i.gpuStatus === "error" ? `GPU monitoring failed: ${i.gpuDetail}` : `No NVIDIA GPU on this replica (${i.gpuDetail || "CPU only"}). The LLM server's GPU, if any, is shown under AI models.`))));
  };

  const drawCharts = async () => {
    const hist = await api.get("/system/history", { instance: chosen, range });
    if (ctx.stale()) return;
    clear(charts);
    const pts = hist.points;
    if (pts.length < 2) { charts.append(card("History", empty("Not enough samples yet."))); return; }
    const x = pts.map((p) => p.at);
    const fmt = timeFormatter(x[x.length - 1] - x[0]);
    const mk = (title, series, opts = {}) => {
      const cols = [{ key: "at", label: "Time", render: (r) => dateTime(r.at) },
        ...series.map((sr, k) => ({ key: `s${k}`, label: sr.name, cls: "r", render: (r) => (opts.yFormat || num)(r.v[k]) }))];
      const rows = x.map((at, i) => ({ at, v: series.map((sr) => sr.values[i]) })).reverse();
      const c = withTable(lineChart({ x, series, xFormat: fmt, height: 190, ...opts }), cols, rows);
      return card(title, c.body, { actions: c.toggle });
    };
    charts.append(
      mk("CPU and memory", [{ name: "CPU %", color: SERIES[0], values: pts.map((p) => p.cpu) },
        { name: "Memory %", color: SERIES[1], values: pts.map((p) => p.mem) }], { yMax: 100, yFormat: (v) => `${num(v, 0)}%`, area: false }),
      mk("Network", [{ name: "Out", color: SERIES[0], values: pts.map((p) => p.netSent) },
        { name: "In", color: SERIES[1], values: pts.map((p) => p.netRecv) }], { yFormat: rate }),
      mk("Disk I/O", [{ name: "Read", color: SERIES[0], values: pts.map((p) => p.diskRead) },
        { name: "Write", color: SERIES[1], values: pts.map((p) => p.diskWrite) }], { yFormat: rate }),
      mk("Jobs on this replica", [{ name: "Active jobs", color: SERIES[0], values: pts.map((p) => p.active) }], { area: true, integer: true, yFormat: (v) => num(v, 0) }),
    );
    const gpuCount = Math.max(0, ...pts.map((p) => (p.gpu || []).length));
    for (let g = 0; g < gpuCount; g++) {
      charts.append(mk(`GPU ${g}`, [{ name: "Utilization %", color: SERIES[0], values: pts.map((p) => p.gpu?.[g] ?? null) },
        { name: "VRAM %", color: SERIES[1], values: pts.map((p) => p.vram?.[g] ?? null) }], { yMax: 100, yFormat: (v) => `${num(v, 0)}%` }));
    }
  };

  const drawBottom = () => {
    clear(bottom);
    bottom.append(card("Replicas", table([
      { key: "host", label: "Host", render: (r) => h("a", { href: `#/system?instance=${encodeURIComponent(r.instance)}` }, `${r.host}${r.self ? " (this)" : ""}`) },
      { key: "instance", label: "Instance", render: (r) => h("span", { class: "mono muted" }, r.instance) },
      { key: "state", label: "State", render: instanceState },
      { key: "cpu", label: "CPU", cls: "r", render: (r) => pct(r.cpu?.percent) },
      { key: "mem", label: "Memory", cls: "r", render: (r) => pct(r.memory?.percent) },
      { key: "ageS", label: "Last sample", cls: "r", render: (r) => `${num(r.ageS, 0)} s ago` },
      { key: "version", label: "Version", render: (r) => r.version },
    ], sys.instances), { cls: "flush" }));
    const db = sys.database;
    bottom.append(card("Database", db ? kv([["Backend", `PostgreSQL ${db.version}`], ["Size", bytes(db.sizeBytes)],
      ["Connections", `${db.connections} of ${db.maxConnections}`],
      ["This replica's pool", db.pool ? `${db.pool.size} open · ${db.pool.available} idle · ${db.pool.waiting} waiting` : "–"]])
      : empty(sys.jobStore === "memory" ? "Memory job store (development): no database." : "Database statistics unavailable.")));
  };

  const current = () => sys.instances.find((i) => i.instance === chosen);
  drawLive(current());
  drawBottom();
  await drawCharts();
  ctx.onLive((snap) => {
    if (snap.sample && snap.instance === chosen) {
      drawLive({ ...current(), ...snap.sample, state: "ok", self: true, ageS: 0 });
    }
  });
  const timer = setInterval(async () => {
    try {
      sys = await api.get("/system");
      if (ctx.stale()) return;
      if (chosen !== sys.self) drawLive(current());
      drawBottom();
      await drawCharts();
    } catch { /* keep */ }
  }, 10000);
  return () => clearInterval(timer);
}
