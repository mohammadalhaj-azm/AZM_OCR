// Integration management: every client system, its authentication, API usage, errors, webhooks and health.
import { api } from "../api.js";
import { h, clear, card, table, statusBadge, badge, kv, tile, empty, fill } from "../ui.js";
import { columnChart, withTable, SERIES } from "../charts.js";
import { int, ms, ago, dateTime, duration } from "../format.js";
import { activityList, buckets, timeFormatter, docLink } from "./common.js";

const SOURCE = { file: "CLIENTS_FILE", dashboard: "managed in this dashboard", api_keys: "API_KEYS (no registry)", none: "none" };

export async function render(root, ctx) {
  const range = ctx.rangePicker("24h");
  if (ctx.rest[0]) return renderClient(root, ctx, ctx.rest[0], range);
  const d = await api.get("/integrations", { range });
  if (ctx.stale()) return;
  clear(root);
  const ev = (c, e) => c.usage?.events?.[e] || 0;
  root.append(h("div", { class: "filters" }, h("span", { class: "secondary" }, `Client registry: ${SOURCE[d.source] || d.source}`),
    ctx.isAdmin ? h("a", { class: "btn sm", href: "#/config?tab=clients" }, "Manage clients") : null));
  const totals = d.clients.reduce((a, c) => {
    a.requests += ev(c, "job.received") + ev(c, "extract.done");
    a.errors += ev(c, "request.error") + ev(c, "job.failed");
    a.unhealthy += ["critical", "warning"].includes(c.health) ? 1 : 0;
    return a;
  }, { requests: 0, errors: 0, unhealthy: 0 });
  root.append(h("div", { class: "tiles" },
    tile("Connected clients", int(d.clients.filter((c) => c.enabled).length), { foot: `${int(d.clients.length)} registered` }),
    tile("Documents sent", int(totals.requests), { foot: `last ${range}` }),
    tile("Errors", int(totals.errors), { foot: "rejected requests + failed jobs" }),
    tile("Clients needing attention", int(totals.unhealthy)),
    tile("Unknown-key requests", int(d.unauthenticated.requests), { foot: d.unauthenticated.lastSeen ? `last ${ago(d.unauthenticated.lastSeen)}` : "none" }),
  ));
  root.append(card("Clients", table([
    { key: "name", label: "Client", render: (c) => h("div", null, h("div", null, c.name), h("div", { class: "muted mono" }, c.id)) },
    { key: "health", label: "Health", render: (c) => h("span", { title: c.healthReasons.join("\n") }, statusBadge(c.health === "critical" ? "failed" : c.health === "warning" ? "warning" : c.health)) },
    { key: "auth", label: "Authentication", render: (c) => h("div", null, c.enabled ? badge("Enabled", "good") : badge("Disabled", "neutral"),
      h("div", { class: "muted" }, `${c.keys} key${c.keys === 1 ? "" : "s"} · ${c.scopes.join(", ")}`)) },
    { key: "lastSeen", label: "Last request", sort: (c) => c.usage?.lastSeen || 0, render: (c) => (c.usage?.lastSeen ? ago(c.usage.lastSeen) : "–") },
    { key: "uploads", label: "Async", cls: "r", sort: (c) => ev(c, "job.received"), render: (c) => int(ev(c, "job.received")) },
    { key: "extracts", label: "Sync", cls: "r", sort: (c) => ev(c, "extract.done"), render: (c) => int(ev(c, "extract.done")) },
    { key: "errors", label: "Errors", cls: "r", sort: (c) => ev(c, "request.error") + ev(c, "job.failed"), render: (c) => int(ev(c, "request.error") + ev(c, "job.failed")) },
    { key: "avg", label: "Avg job", cls: "r", sort: (c) => c.usage?.avgMs || 0, render: (c) => ms(c.usage?.avgMs) },
    { key: "webhooks", label: "Webhooks", render: (c) => (c.webhook.configured
      ? `${int(ev(c, "webhook.delivered"))} ok · ${int(ev(c, "webhook.failed"))} failed · ${int(c.live.webhooksPending)} pending` : "off") },
    { key: "queued", label: "Queued", cls: "r", sort: (c) => c.live.queued, render: (c) => int(c.live.queued) },
  ], d.clients, { onRow: (c) => ctx.navigate(`integrations/${c.id}`, { range }), emptyText: "No clients registered." }), { cls: "flush" }));
  if (d.orphans.length) {
    root.append(h("div", { class: "banner info" }, `Activity from client ids no longer in the registry: ${d.orphans.join(", ")}.`));
  }
}

async function renderClient(root, ctx, id, range) {
  const d = await api.get(`/integrations/${encodeURIComponent(id)}`, { range });
  if (ctx.stale()) return;
  clear(root);
  const c = d.client;
  const e = (k) => c.usage?.events?.[k] || 0;
  root.append(h("div", { class: "filters" }, h("a", { href: `#/integrations?range=${range}` }, "← All clients"),
    h("h2", null, c.name || c.id), c.removed ? badge("Removed from the registry", "neutral") : statusBadge(c.health === "critical" ? "failed" : c.health),
    ctx.isAdmin && !c.removed ? h("a", { class: "btn sm", href: `#/config?tab=clients&client=${encodeURIComponent(id)}` }, "Edit") : null,
    h("a", { class: "btn sm", href: `#/documents?client=${encodeURIComponent(id)}` }, "Documents")));
  if (c.healthReasons?.length) root.append(h("div", { class: "banner" }, c.healthReasons.join(" · ")));
  root.append(h("div", { class: "tiles" },
    tile("Documents", int(e("job.received") + e("extract.done")), { foot: `${int(e("job.received"))} async · ${int(e("extract.done"))} sync` }),
    tile("Results fetched", int(e("job.fetched")), { foot: `${int(e("job.deleted"))} deleted by the client` }),
    tile("Errors", int(e("request.error") + e("job.failed")), { foot: Object.entries(c.usage?.httpStatus || {}).map(([s, n]) => `${s}: ${n}`).join(" · ") || "none" }),
    tile("Average job", ms(c.usage?.avgMs)),
    tile("Webhooks", c.webhook?.configured ? `${int(e("webhook.delivered"))} / ${int(e("webhook.failed"))}` : "off", { foot: c.webhook?.configured ? `delivered / given up · ${int(c.live?.webhooksPending)} pending` : "no callbacks" }),
  ));
  const since = Date.now() / 1000 - ({ "15m": 900, "1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800, "30d": 2592000 }[range] || 86400);
  const b = buckets(d.series, since, d.bucketS, (r) => ({
    "job.received": { async: r.n }, "extract.done": { sync: r.n }, "request.error": { errors: r.n }, "job.failed": { errors: r.n },
  }[r.event]));
  const keys = [{ key: "async", name: "Async uploads", color: SERIES[0] }, { key: "sync", name: "Sync extracts", color: SERIES[1] },
    { key: "errors", name: "Errors", color: "var(--critical)" }];
  const chart = withTable(columnChart({ buckets: b, keys, xFormat: timeFormatter(Date.now() / 1000 - since), height: 200 }),
    [{ key: "x", label: "Period", render: (r) => dateTime(r.x) }, ...keys.map((k) => ({ key: k.key, label: k.name, cls: "r", render: (r) => int(r.values[k.key] || 0) }))],
    b.filter((x) => Object.keys(x.values).length));
  root.append(h("div", { class: "grid cols-3" },
    card("API usage", chart.body, { sub: `per ${duration(d.bucketS)}`, actions: chart.toggle, cls: "span-2" }),
    card("Configuration", c.removed ? empty("This client is no longer registered.") : kv([
      ["Id", h("span", { class: "mono" }, c.id)], ["Status", c.enabled ? "enabled" : "disabled (keys get 401)"],
      ["Scopes", c.scopes.join(", ")], ["Profiles", c.profiles.join(", ")],
      ["API keys", h("div", { class: "chips" }, c.keyFingerprints.map((f) => h("span", { class: "chip mono" }, `${f}…`)))],
      ["Webhook hosts", c.webhook.hosts.join(", ") || "callbacks off"],
      ["Retry window", c.webhook.retryWindowS ? duration(c.webhook.retryWindowS) : "default"],
      ["Queue limit", c.limits.maxQueuedJobs ?? "global"], ["Sync concurrency", c.limits.extractConcurrent ?? "global"],
      ["Now", `${int(c.live.queued)} queued · ${int(c.live.processing)} processing`],
    ]))));
  const errs = d.stats.dist;
  const errRows = [...Object.entries(errs.httpError).map(([k, n]) => ({ code: k, kind: "request", n })),
    ...Object.entries(errs.error).map(([k, n]) => ({ code: k, kind: "job", n }))].sort((x, y) => y.n - x.n);
  const history = h("div");
  let items = d.requests;
  let next = d.next;
  const more = h("div", { class: "pager" });
  const drawHistory = () => {
    fill(history, activityList(items, { onClick: (ev) => { const to = docLink(ev); if (to) ctx.navigate(to); } }));
    fill(more, next ? h("button", { class: "btn sm", type: "button", onclick: async () => {
      const r = await api.get(`/integrations/${encodeURIComponent(id)}`, { range, before: next });
      items = [...items, ...r.requests]; next = r.next; drawHistory();
    } }, "Load more") : null);
  };
  drawHistory();
  root.append(h("div", { class: "grid cols-3" },
    card("Request history", h("div", null, history, more), { cls: "flush span-2", sub: "audit trail, newest first" }),
    card("Errors by code", errRows.length ? table([{ key: "kind", label: "Kind" }, { key: "code", label: "Code" },
      { key: "n", label: "Count", cls: "r" }], errRows) : empty("No errors in this period."))));
}
