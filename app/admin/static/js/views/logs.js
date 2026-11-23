// Logs & auditing: log records (this replica, or warnings/errors of all replicas), the audit trail, and a trace
// view that puts everything about one request id or job id on one page.
import { api, qs } from "../api.js";
import { h, clear, card, table, select, seg, empty, errorBox, fill } from "../ui.js";
import { dateTime, time, short } from "../format.js";
import { eventBadge, eventSummary, docLink } from "./common.js";

const TABS = [["logs", "Logs"], ["audit", "Audit trail"], ["trace", "Trace a request"]];

export async function render(root, ctx) {
  const tab = ctx.params.tab || "logs";
  root.append(h("nav", { class: "tabs" }, TABS.map(([id, text]) => h("a", { href: `#/logs?tab=${id}`, class: id === tab ? "on" : "" }, text))));
  const body = h("div", { class: "grid" });
  root.append(body);
  if (tab === "audit") return renderAudit(body, ctx);
  if (tab === "trace") return renderTrace(body, ctx);
  return renderLogs(body, ctx);
}

function idLink(kind, value) {
  if (!value) return null;
  return h("a", { href: `#/logs?tab=trace&${kind}=${encodeURIComponent(value)}`, class: "mono", title: value }, short(value, 8));
}

async function renderLogs(body, ctx) {
  const p = ctx.params;
  const f = { source: p.source || "local", level: p.level || "", logger: p.logger || "", text: p.text || "",
    requestId: p.requestId || "", jobId: p.jobId || "" };
  const apply = (change) => ctx.navigate("logs", { tab: "logs", ...f, ...change });
  let follow = p.follow !== "0";
  const text = h("input", { type: "search", placeholder: "Search messages", value: f.text,
    onkeydown: (e) => { if (e.key === "Enter") apply({ text: e.target.value.trim() }); } });
  const logger = h("input", { type: "text", placeholder: "Logger (prefix)", value: f.logger,
    onkeydown: (e) => { if (e.key === "Enter") apply({ logger: e.target.value.trim() }); } });
  const followBtn = h("button", { type: "button", class: "btn sm", onclick: () => { follow = !follow; followBtn.textContent = follow ? "Following" : "Paused"; } },
    follow ? "Following" : "Paused");
  body.append(h("div", { class: "filters" },
    seg([["local", "This replica (all levels)"], ["all", "All replicas (warnings+)"]], f.source, (v) => apply({ source: v })),
    select([["", "Any level"], ["DEBUG", "Debug+"], ["INFO", "Info+"], ["WARNING", "Warning+"], ["ERROR", "Error+"]], f.level, (v) => apply({ level: v })),
    logger, text, followBtn,
    (f.requestId || f.jobId) ? h("button", { class: "btn ghost", type: "button", onclick: () => apply({ requestId: "", jobId: "" }) }, "Clear id filter") : null));
  const list = h("div");
  const more = h("div", { class: "pager" });
  body.append(card(null, h("div", null, list, more), { cls: "flush" }));
  body.append(h("p", { class: "muted" }, "Log messages never contain document text, file names or keys; exceptions are "
    + "recorded by type only."));
  let rows = [];
  let next = null;
  const columns = [
    { key: "at", label: "Time", render: (r) => h("span", { class: "nowrap num" }, dateTime(r.at)) },
    { key: "level", label: "Level", render: (r) => h("span", { class: `level-${r.level}` }, r.level) },
    { key: "logger", label: "Logger", render: (r) => h("span", { class: "mono muted" }, r.logger) },
    { key: "msg", label: "Message", render: (r) => h("span", { class: "mono" }, r.msg, r.exc_type ? ` [${r.exc_type}]` : "") },
    { key: "ids", label: "Ids", nosort: true, render: (r) => h("div", { class: "chips" }, idLink("requestId", r.request_id),
      idLink("jobId", r.job_id), r.client_id ? h("span", { class: "chip" }, r.client_id) : null) },
    ...(f.source === "all" ? [{ key: "instance", label: "Replica", render: (r) => h("span", { class: "mono muted" }, short(r.instance, 10)) }] : []),
  ];
  const load = async (append = false) => {
    const res = await api.get("/logs", { ...f, before: append ? next : undefined, limit: 200 });
    if (ctx.stale()) return;
    rows = append ? [...rows, ...res.items] : res.items;
    next = res.next;
    fill(list, table(columns, rows, { emptyText: f.source === "all" ? "No warnings or errors stored." : "No log records match." }));
    fill(more, res.dropped ? h("span", { class: "muted" }, `${res.dropped} records could not be stored`) : null,
      next ? h("button", { class: "btn sm", type: "button", onclick: () => { follow = false; followBtn.textContent = "Paused"; load(true); } }, "Older") : null);
  };
  await load();
  const timer = setInterval(() => { if (follow) load().catch(() => {}); }, 3000);
  return () => clearInterval(timer);
}

async function renderAudit(body, ctx) {
  const p = ctx.params;
  const range = ctx.rangePicker("7d", ["1h", "24h", "7d", "30d"]);
  const f = { event: p.event || "", client: p.client || "", job: p.job || "", requestId: p.requestId || "", actor: p.actor || "",
    sha256: p.sha256 || "", externalId: p.externalId || "" };
  const apply = (change) => ctx.navigate("logs", { tab: "audit", range, ...f, ...change });
  const input = (key, placeholder) => h("input", { type: "text", placeholder, value: f[key],
    onkeydown: (e) => { if (e.key === "Enter") apply({ [key]: e.target.value.trim() }); } });
  const first = await api.get("/audit", { ...f, range, limit: 100 });
  if (ctx.stale()) return;
  const eventOptions = [["", "All events"], ["job.", "Job lifecycle (job.*)"], ["webhook.", "Webhooks (webhook.*)"],
    ["admin.", "Admin actions (admin.*)"], ...first.events.map((e) => [e, e])];
  body.append(h("div", { class: "filters" }, select(eventOptions, f.event, (v) => apply({ event: v })),
    input("client", "Client id"), input("job", "Job id"), input("requestId", "Request id"), input("actor", "Admin user"),
    input("sha256", "File SHA-256"), input("externalId", "externalId (hashed before search)"),
    h("a", { class: "btn", href: `/admin/api/audit${qs({ ...f, range, limit: 5000, format: "csv" })}`, download: "audit.csv" }, "Export CSV")));
  const list = h("div");
  const more = h("div", { class: "pager" });
  body.append(card(null, h("div", null, list, more), { cls: "flush" }));
  let rows = first.items;
  let next = first.next;
  const draw = () => {
    fill(list, table([
      { key: "at", label: "Time", render: (r) => h("span", { class: "nowrap num" }, dateTime(r.at)) },
      { key: "event", label: "Event", render: (r) => eventBadge(r.event) },
      { key: "clientId", label: "Client", render: (r) => r.clientId || (r.data?.actor ? `admin: ${r.data.actor}` : "–") },
      { key: "details", label: "Details", nosort: true, render: (r) => h("span", { class: "secondary" }, eventSummary(r)) },
      { key: "jobId", label: "Job", render: (r) => idLink("jobId", r.jobId) },
      { key: "requestId", label: "Request", render: (r) => idLink("requestId", r.requestId) },
      { key: "data", label: "Data", nosort: true, render: (r) => h("details", null, h("summary", { class: "muted" }, "json"),
        h("div", { class: "json" }, JSON.stringify(r.data, null, 1))) },
    ], rows, { emptyText: "No audit events match.", onRow: null }));
    fill(more, h("span", { class: "muted" }, `${rows.length} shown`),
      next ? h("button", { class: "btn sm", type: "button", onclick: async () => {
        const r = await api.get("/audit", { ...f, range, before: next, limit: 100 });
        rows = [...rows, ...r.items]; next = r.next; draw();
      } }, "Load more") : null);
  };
  draw();
}

async function renderTrace(body, ctx) {
  const p = ctx.params;
  const input = h("input", { type: "search", placeholder: "X-Request-Id or job id", value: p.requestId || p.jobId || "" });
  const go = (kind) => { const v = input.value.trim(); if (v) ctx.navigate("logs", { tab: "trace", [kind]: v }); };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") go("requestId"); });
  body.append(h("div", { class: "filters" }, input,
    h("button", { class: "btn", type: "button", onclick: () => go("requestId") }, "Find request id"),
    h("button", { class: "btn", type: "button", onclick: () => go("jobId") }, "Find job id")));
  if (!p.requestId && !p.jobId) {
    body.append(empty("Paste the X-Request-Id a client quotes (or a job id) to see its whole story: upload, "
      + "processing, webhook, fetch, and every log line on the way."));
    return;
  }
  let d;
  try {
    d = await api.get("/trace", { requestId: p.requestId, jobId: p.jobId });
  } catch (e) {
    body.append(errorBox(e));
    return;
  }
  if (ctx.stale()) return;
  body.append(h("div", { class: "filters secondary" }, `Request ids: ${d.requestIds.join(", ") || "–"}`, " · ",
    `Jobs: `, ...d.jobIds.map((j) => h("a", { href: `#/documents/${encodeURIComponent(j)}`, class: "mono" }, short(j, 12)))));
  body.append(h("div", { class: "grid cols-2" },
    card("Audit events", d.events.length ? h("ol", { class: "timeline" }, d.events.map((e) => h("li", {
      class: ["job.failed", "webhook.failed", "request.error"].includes(e.event) ? "fail" : ["job.done", "extract.done", "webhook.delivered"].includes(e.event) ? "ok" : "",
    }, h("div", null, eventBadge(e.event), " ", docLink(e) ? h("a", { href: `#/${docLink(e)}` }, "document") : null),
    h("div", { class: "meta" }, `${dateTime(e.at)} · ${e.clientId || "–"}`), h("div", { class: "meta" }, eventSummary(e))))) : empty("No audit events.")),
    card("Log lines", d.logs.length ? h("div", { class: "table-wrap" }, h("table", { class: "data" }, h("tbody", null, d.logs.map((r) => h("tr", null,
      h("td", { class: "nowrap num muted" }, time(r.at)), h("td", null, h("span", { class: `level-${r.level}` }, r.level)),
      h("td", { class: "mono" }, r.msg, r.exc_type ? ` [${r.exc_type}]` : "")))))) : empty("No log lines (this replica's buffer and stored warnings)."), { cls: "flush" })));
}
