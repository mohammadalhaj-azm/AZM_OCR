// Document tracking: every submitted document (async jobs, sync extracts, rejected uploads), from submission to
// completion. Metadata only: the dashboard never shows document text, file names or MRZ fields.
import { api } from "../api.js";
import { h, clear, card, table, statusBadge, select, drawer, kv, badge, confirm, toast, errorBox, empty, closeDrawer, fill } from "../ui.js";
import { int, ms, bytes, dateTime, time, num, DOC_TYPE_LABELS, short, dash } from "../format.js";
import { eventBadge, eventSummary } from "./common.js";

const STATUSES = [["", "Any status"], ["done", "Done"], ["failed", "Failed"], ["processing", "Processing"],
  ["queued", "Queued"], ["rejected", "Rejected"], ["deleted", "Deleted before finishing"]];
const KINDS = [["", "Any source"], ["job", "Async job (/v1/documents)"], ["extract", "Sync extract (/v1/extract)"],
  ["rejected", "Rejected upload"]];

export async function render(root, ctx) {
  const range = ctx.rangePicker("7d", ["1h", "24h", "7d", "30d"]);
  const p = ctx.params;
  const filters = { status: p.status || "", kind: p.kind || "", client: p.client || "", docType: p.docType || "",
    error: p.error || "", q: p.q || "" };
  const integrations = await api.get("/integrations", { range: "30d" }).catch(() => ({ clients: [] }));
  if (ctx.stale()) return;

  const apply = (change) => ctx.navigate("documents", { ...filters, ...change, range });
  const search = h("input", { type: "search", placeholder: "Job id, request id, SHA-256 or externalId", value: filters.q,
    onkeydown: (e) => { if (e.key === "Enter") apply({ q: e.target.value.trim() }); } });
  const bar = h("div", { class: "filters" }, search,
    select(STATUSES, filters.status, (v) => apply({ status: v })),
    select(KINDS, filters.kind, (v) => apply({ kind: v })),
    select([["", "Any client"], ...integrations.clients.map((c) => [c.id, c.name])], filters.client, (v) => apply({ client: v })),
    select([["", "Any type"], ...Object.entries(DOC_TYPE_LABELS)], filters.docType, (v) => apply({ docType: v })),
    Object.values(filters).some(Boolean) ? h("button", { class: "btn ghost", type: "button", onclick: () => ctx.navigate("documents", { range }) }, "Clear filters") : null);

  const listBox = h("div");
  const more = h("div", { class: "pager" });
  root.append(bar, card(null, h("div", null, listBox, more), { cls: "flush" }));
  const names = Object.fromEntries(integrations.clients.map((c) => [c.id, c.name]));
  let rows = [];
  let next = null;

  const columns = [
    { key: "receivedAt", label: "Received", render: (r) => h("span", { class: "nowrap num" }, dateTime(r.receivedAt)) },
    { key: "status", label: "Status", render: (r) => statusBadge(r.status) },
    { key: "clientId", label: "Client", render: (r) => names[r.clientId] || r.clientId || "unknown key" },
    { key: "endpoint", label: "Source", render: (r) => (r.kind === "job" ? "async" : r.kind === "extract" ? "sync" : `rejected ${r.httpStatus || ""}`) },
    { key: "pages", label: "Pages", cls: "r", render: (r) => (r.pages ?? dash) },
    { key: "docType", label: "Result", render: (r) => (r.docType ? `${DOC_TYPE_LABELS[r.docType] || r.docType} · ${num(r.confidence * 100, 0)}%`
      : r.errorCode ? h("span", { class: "level-ERROR" }, r.errorCode) : r.kind === "extract" ? "text only" : dash) },
    { key: "ms", label: "Duration", cls: "r", render: (r) => ms(r.ms) },
    { key: "size", label: "Size", cls: "r", render: (r) => bytes(r.size) },
    { key: "id", label: "Id", render: (r) => h("span", { class: "mono muted", title: r.id }, short(r.jobId || r.id, 10)) },
  ];

  const load = async (append = false) => {
    const res = await api.get("/documents", { range, ...filters, before: append ? next : undefined });
    if (ctx.stale()) return;
    rows = append ? [...rows, ...res.items] : res.items;
    next = res.next;
    fill(listBox, table(columns, rows, { onRow: (r) => ctx.navigate(`documents/${r.id}`, { ...filters, range }),
      emptyText: "No documents match." }));
    fill(more, h("span", { class: "muted" }, `${rows.length} shown`),
      next ? h("button", { class: "btn sm", type: "button", onclick: () => load(true) }, "Load more") : null);
  };
  await load();

  if (ctx.rest[0]) openDetail(ctx.rest[0], ctx, () => ctx.navigate("documents", { ...filters, range }));

  // New documents appear on their own while the first page is shown
  let lastEvent = null;
  let lastLoad = Date.now();
  ctx.onLive((snap) => {
    if (lastEvent !== null && snap.latestEventId !== lastEvent && Date.now() - lastLoad > 5000 && !document.querySelector(".drawer")) {
      lastLoad = Date.now();
      load().catch(() => {});
    }
    lastEvent = snap.latestEventId;
  });
}

const STEP_ORDER = [["queue", "Queue wait"], ["extract", "Extraction"], ["ocr", "· OCR"], ["normalize", "· Normalize"],
  ["llmQueue", "Wait for LLM"], ["rules", "Rules"], ["llm", "Text LLM"], ["vision", "Vision LLM"]];

function waterfall(steps, total) {
  const span = Math.max(1, total || Object.values(steps).reduce((a, b) => a + (b || 0), 0));
  const rowsEl = [];
  let offset = 0;
  for (const [key, name] of STEP_ORDER) {
    const v = steps[key];
    if (v === undefined || v === null) continue;
    const nested = name.startsWith("·");
    const start = nested ? steps.__extractStart : offset;
    if (key === "extract") steps.__extractStart = offset;
    rowsEl.push(h("div", { class: "wf" }, h("span", { class: nested ? "muted" : "" }, name),
      h("div", { class: "lane" }, h("i", { style: { left: `${(start / span) * 100}%`, width: `${Math.max(0.4, (v / span) * 100)}%`,
        background: key === "queue" || key === "llmQueue" ? "var(--axis)" : nested ? "var(--series-3)" : "var(--series-1)" } })),
      h("span", { class: "r num" }, ms(v))));
    if (!nested) offset += v;
  }
  return rowsEl.length ? h("div", { class: "waterfall" }, rowsEl) : empty("No step timings recorded.");
}

async function openDetail(id, ctx, onClose) {
  const body = drawer(`Document ${short(id, 12)}`, h("div", { class: "empty" }, "Loading…"));
  const panel = body.closest(".drawer");
  const observer = new MutationObserver(() => {
    if (document.body.contains(panel)) return;
    observer.disconnect();
    // closed by the user (not by navigating elsewhere): back to the list
    if (location.hash.includes(`documents/${encodeURIComponent(id)}`) || location.hash.includes(`documents/${id}`)) onClose();
  });
  observer.observe(document.body, { childList: true });
  let d;
  try {
    d = await api.get(`/documents/${encodeURIComponent(id)}`);
  } catch (e) {
    fill(body, errorBox(e));
    return;
  }
  clear(body);
  const doc = d.document || {};
  const final = [...d.events].reverse().find((e) => ["job.done", "job.failed", "extract.done"].includes(e.event));
  const received = d.events.find((e) => e.event === "job.received") || d.events[0];
  const started = d.events.find((e) => e.event === "job.started");
  const facts = final?.data || {};
  const timings = { ...(facts.timingsMs || {}) };
  if (started && received) timings.queue = Math.round((started.at - received.at) * 1000);

  body.append(h("div", { class: "filters" }, statusBadge(doc.status || d.job?.status),
    d.active ? badge(`now: ${d.active.stage}`, "info") : null,
    doc.requestId ? h("a", { href: `#/logs?tab=trace&requestId=${encodeURIComponent(doc.requestId)}` }, "Trace request") : null,
    ctx.isAdmin && d.job ? h("button", { class: "btn sm danger", type: "button", onclick: async () => {
      if (!await confirm("Delete this job?", "The job and its result are deleted now (the audit trail keeps the record). "
        + "The client can no longer fetch the result.", { okText: "Delete", danger: true })) return;
      try {
        await api.del(`/documents/${encodeURIComponent(id)}`);
        toast("Job deleted", id, "good");
        closeDrawer();
      } catch (e) { toast("Delete failed", e.message, "critical"); }
    } }, "Delete job") : null));

  const det = facts.idDetection;
  body.append(card("Summary", kv([
    ["Client", doc.clientId || "unknown key"], ["Source", doc.endpoint], ["Received", dateTime(doc.receivedAt)],
    ["Finished", doc.finishedAt ? dateTime(doc.finishedAt) : dash], ["Total time", ms(timings.total ?? doc.ms)],
    ["File type", doc.mimeType], ["Size", bytes(doc.size)], ["SHA-256", doc.sha256 ? h("span", { class: "mono" }, doc.sha256) : dash],
    ["Pages", doc.pages !== null && doc.pages !== undefined ? `${doc.pages} (${Object.entries(doc.pageMethods || {}).map(([m, n]) => `${n} ${m.replace("_", " ")}`).join(", ")})` : dash],
    ["Languages", (doc.languages || []).join(", ") || dash], ["OCR confidence", facts.ocrConfidence !== undefined ? `${num(facts.ocrConfidence * 100)}%` : dash],
    ["Profile", doc.profile || facts.processing?.profile || "default"], ["Request id", doc.requestId ? h("span", { class: "mono" }, doc.requestId) : dash],
    ["Error", doc.errorCode ? h("span", { class: "level-ERROR" }, doc.errorCode) : dash],
  ])));
  if (det) {
    body.append(card("ID detection", kv([
      ["Decision", `${DOC_TYPE_LABELS[det.docType] || det.docType}${det.isId ? " (identity document)" : ""}`],
      ["Confidence", `${num(det.confidence * 100)}%`], ["Decided by", (det.decidedBy || []).join(" → ")],
      ["MRZ", det.mrzFound ? (det.mrzValid ? "found, all check digits valid" : "found, some checks failed") : "none"],
      ["Model fallbacks", (facts.fallbacks || []).join(", ") || "none"],
      ["Models", [facts.processing?.ocr?.det, facts.processing?.llm, facts.processing?.vision].filter(Boolean).join(", ") || dash],
      ["Rules version", facts.processing?.rules || dash],
    ])));
  }
  body.append(card("Processing steps", waterfall(timings, timings.total ? timings.total + (timings.queue || 0) : null),
    { sub: "durations from the audit trail" }));
  if (d.job) {
    body.append(card("Job (live)", kv([
      ["Status", d.job.status], ["Attempts", d.job.attempts], ["Webhook", d.job.callback ? `${d.job.webhook?.state || "pending"} (${d.job.webhook?.attempts ?? 0} attempts)` : "none"],
      ["Result kept", d.job.resultTtlS ? `${d.job.resultTtlS} s (profile)` : "RESULT_TTL_S"],
      ["Error", d.job.error ? `${d.job.error.code}: ${d.job.error.message}` : dash],
    ])));
    if (d.job.status === "failed") {
      body.append(h("div", { class: "banner info" }, "Failed jobs can't be re-run from here: the upload is deleted right "
        + "after processing (privacy). The client has to send the document again."));
    }
  }
  body.append(card("Lifecycle", h("ol", { class: "timeline" }, d.timeline.length ? d.timeline.map((t) => h("li", {
    class: ["job.failed", "webhook.failed", "request.error"].includes(t.event) ? "fail" : ["job.done", "extract.done", "webhook.delivered"].includes(t.event) ? "ok" : "",
  }, h("div", null, eventBadge(t.event)), h("div", { class: "meta" }, `${time(t.at)}${t.sincePrevMs !== null ? ` · +${ms(t.sincePrevMs)}` : ""}`),
  h("div", { class: "meta" }, eventSummary({ event: t.event, data: t.data })))) : d.events.map((e) => h("li", null,
    h("div", null, eventBadge(e.event)), h("div", { class: "meta" }, `${time(e.at)} · ${eventSummary(e)}`))))));
  body.append(h("p", { class: "muted" }, `${int(d.events.length)} audit events. Document text, file names and MRZ fields `
    + "are never shown in the dashboard."));
}
