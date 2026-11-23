// Overview: KPIs for the period, live queue and health, throughput, replicas, alerts, recent activity.
import { api } from "../api.js";
import { h, clear, card, tile, healthDot, table, badge, empty, fill } from "../ui.js";
import { columnChart, hbars, withTable } from "../charts.js";
import { int, ms, ratio, duration, pct, ago, DOC_TYPE_LABELS, dateTime } from "../format.js";
import { throughputBuckets, THROUGHPUT_KEYS, timeFormatter, activityList, docLink, instanceState } from "./common.js";

export async function render(root, ctx) {
  const range = ctx.rangePicker("24h");
  let data = await api.get("/overview", { range });
  if (ctx.stale()) return;
  const live = {};

  const draw = () => {
    clear(root);
    const k = data.kpis;
    if (!ctx.me.persistent) {
      root.append(h("div", { class: "banner info" }, "Memory job store (development): history, samples, alerts and "
        + "dashboard settings are kept in RAM only and lost on restart."));
    }
    live.queued = h("span", null, int(data.live.jobs.queued));
    live.queuedFoot = h("span", null, `oldest waiting ${duration(data.live.ages.queued)}`);
    live.processing = h("span", null, int(data.live.jobs.processing));
    live.processingFoot = h("span", null, `longest ${duration(data.live.ages.processing)}`);
    root.append(h("div", { class: "tiles" },
      tile("Documents processed", int(k.documents), { hero: true,
        foot: `${int(k.jobsReceived)} async jobs · ${int(k.extracts)} sync extracts · last ${range}` }),
      tile("Success rate", ratio(k.successRate), { foot: `${int(k.done)} completed` }),
      tile("Failed", int(k.failed), { foot: `${int(k.requestErrors)} rejected requests` }),
      tile("Queued now", live.queued, { foot: live.queuedFoot }),
      tile("Processing now", live.processing, { foot: live.processingFoot }),
      tile("Processing time (p50)", ms(k.jobMs.p50), { foot: `p95 ${ms(k.jobMs.p95)} · ${int(k.jobMs.n)} jobs` }),
      tile("Pages", int(k.pages), { foot: `${int(k.ids)} identity documents found` }),
    ));

    live.health = h("div", { class: "health-strip" });
    root.append(card("Service health", live.health, { sub: "this replica, live" }));

    const tb = throughputBuckets(data.series, data.since, data.bucketS);
    const fmt = timeFormatter(Date.now() / 1000 - data.since);
    const chart = withTable(columnChart({ buckets: tb, keys: THROUGHPUT_KEYS, xFormat: fmt, height: 220 }),
      [{ key: "x", label: "Period starting", render: (r) => dateTime(r.x) },
        { key: "done", label: "Completed", cls: "r", render: (r) => int(r.values.done || 0) },
        { key: "failed", label: "Failed", cls: "r", render: (r) => int(r.values.failed || 0) }],
      tb.filter((b) => b.values.done || b.values.failed));
    const alertList = data.alerts.length ? h("div", { class: "table-wrap" }, h("table", { class: "data" }, h("tbody", null,
      data.alerts.map((a) => h("tr", { class: "click", onclick: () => ctx.navigate("alerts") },
        h("td", null, badge(a.severity === "critical" ? "Critical" : "Warning", a.severity === "critical" ? "critical" : "warning")),
        h("td", null, h("div", null, a.title), h("div", { class: "muted" }, a.summary)),
        h("td", { class: "muted nowrap" }, ago(a.startedAt))))))) : empty("No open alerts.");
    root.append(h("div", { class: "grid cols-3" },
      card("Throughput", chart.body, { sub: `per ${duration(data.bucketS)}`, actions: chart.toggle, cls: "span-2" }),
      card("Open alerts", alertList, { actions: h("a", { href: "#/alerts" }, "All alerts") })));
    const types = Object.entries(data.docTypes).map(([t, n]) => ({ name: DOC_TYPE_LABELS[t] || t, value: n }))
      .sort((a, b) => b.value - a.value);

    const replicas = table([
      { key: "host", label: "Replica", render: (r) => h("div", null, h("div", null, r.host || "?", r.self ? " (this)" : ""),
        h("div", { class: "muted mono" }, r.instance)) },
      { key: "state", label: "State", render: instanceState },
      { key: "cpu", label: "CPU", cls: "r", render: (r) => pct(r.cpu) },
      { key: "memory", label: "Memory", cls: "r", render: (r) => pct(r.memory) },
      { key: "active", label: "Active jobs", cls: "r", render: (r) => int(r.work?.activeJobs) },
      { key: "uptimeS", label: "Uptime", cls: "r", render: (r) => duration(r.uptimeS) },
    ], data.instances, { onRow: () => ctx.navigate("system"), emptyText: "No replica has reported yet." });

    const errors = [
      ...Object.entries(data.errors.jobs).map(([code, n]) => ({ name: `job: ${code}`, value: n })),
      ...Object.entries(data.errors.http).map(([code, n]) => ({ name: `http: ${code}`, value: n })),
    ].sort((a, b) => b.value - a.value).slice(0, 8);
    root.append(h("div", { class: "grid cols-3" },
      card("Replicas", replicas, { cls: "flush span-2" }),
      h("div", { class: "grid" },
        card("Errors by code", errors.length ? hbars(errors, { color: "var(--critical)", format: int }) : empty("No errors in this period.")),
        card("Document types", types.length ? hbars(types, { format: int }) : empty("No ID decisions in this period.")))));

    root.append(card("Recent activity", activityList(data.activity, {
      onClick: (ev) => { const to = docLink(ev); if (to) ctx.navigate(to); },
    }), { cls: "flush", actions: h("a", { href: "#/logs?tab=audit" }, "Audit log") }));
  };

  const updateLive = (snap) => {
    if (!live.health) return;
    live.queued.textContent = int(snap.jobs.queued);
    live.queuedFoot.textContent = `oldest waiting ${duration(snap.ages.queued)}`;
    live.processing.textContent = int(snap.jobs.processing);
    live.processingFoot.textContent = `longest ${duration(snap.ages.processing)} · ${snap.active.length} on this replica`;
    const hl = snap.sample?.health || {};
    fill(live.health, 
      healthDot(hl.store, "Database / job store"), healthDot(hl.ocrModels, "OCR models"), healthDot(hl.worker, "Job worker"),
      healthDot(hl.llm, "Text LLM", hl.llmReachable === false ? "LLM server unreachable" : ""),
      healthDot(hl.vision, "Vision LLM"),
      h("div", { class: "dep" }, badge(int(snap.webhooksPending), snap.webhooksPending > 100 ? "warning" : "neutral", { plain: true }),
        h("b", null, "Webhooks pending")));
  };

  let lastSnap = null;
  draw();
  ctx.onLive((snap) => { lastSnap = snap; updateLive(snap); });
  const timer = setInterval(async () => {  // the period's figures; the live parts update on every tick
    try {
      data = await api.get("/overview", { range });
      if (!ctx.stale()) { draw(); if (lastSnap) updateLive(lastSnap); }
    } catch { /* keep the last data */ }
  }, 20000);
  return () => clearInterval(timer);
}
