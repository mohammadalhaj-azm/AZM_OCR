// Alerts: what fired, what is still open, who acknowledged it, and the rules behind them.
import { api } from "../api.js";
import { h, clear, card, table, seg, badge, statusBadge, toast, icon, fill } from "../ui.js";
import { dateTime, ago, duration } from "../format.js";

function severity(a) {
  const kind = a.severity === "critical" ? "critical" : a.severity === "warning" ? "warning" : "info";
  return h("span", { class: `badge ${kind} plain` }, icon(kind === "info" ? "info" : kind === "critical" ? "x" : "warn", "sev-icon"),
    a.severity === "critical" ? "Critical" : a.severity === "warning" ? "Warning" : "Info");
}

export async function render(root, ctx) {
  const state = ctx.params.state || "open";
  ctx.controls.append(seg([["open", "Open"], ["resolved", "Resolved"], ["all", "All"]], state, (v) => ctx.navigate("alerts", { state: v })));
  const list = h("div");
  const rulesBox = h("div");
  root.append(card(null, list, { cls: "flush" }), rulesBox);
  let rules = [];

  const load = async () => {
    const d = await api.get("/alerts", { state: state === "all" ? "" : state });
    if (ctx.stale()) return;
    rules = d.rules;
    fill(list, table([
      { key: "severity", label: "Severity", sort: (a) => (a.severity === "critical" ? 0 : 1), render: severity },
      { key: "title", label: "Alert", render: (a) => h("div", null, h("div", null, a.title), h("div", { class: "muted" }, a.summary)) },
      { key: "scope", label: "Where", render: (a) => h("span", { class: "mono muted" }, a.scope ? a.scope.replace("|", " · ") : "platform") },
      { key: "startedAt", label: "Started", render: (a) => h("span", { title: dateTime(a.startedAt) }, ago(a.startedAt)) },
      { key: "dur", label: "Duration", cls: "r", sort: (a) => (a.resolvedAt || Date.now() / 1000) - a.startedAt,
        render: (a) => duration((a.resolvedAt || Date.now() / 1000) - a.startedAt) },
      { key: "state", label: "State", render: (a) => h("div", null, statusBadge(a.state),
        a.ackedBy ? h("div", { class: "muted" }, `by ${a.ackedBy}`) : null) },
      { key: "action", label: "", nosort: true, render: (a) => (ctx.isAdmin && a.state === "firing" ? h("button", {
        class: "btn sm", type: "button", onclick: async (e) => {
          e.stopPropagation();
          try { await api.post(`/alerts/${a.id}/ack`); toast("Acknowledged", a.title, "good"); load(); } catch (err) { toast("Failed", err.message, "critical"); }
        } }, "Acknowledge") : null) },
    ], d.items, { emptyText: state === "open" ? "No open alerts: everything looks fine." : "No alerts." }));
    fill(rulesBox, card("Alert rules", table([
      { key: "title", label: "Rule" },
      { key: "severity", label: "Severity", render: (r) => badge(r.severity, r.severity === "critical" ? "critical" : "warning") },
      { key: "kind", label: "Scope", render: (r) => ({ global: "platform", client: "per client", instance: "per replica", peer: "other replicas" }[r.kind]) },
      { key: "name", label: "Id", render: (r) => h("span", { class: "mono muted" }, r.name) },
    ], rules), { cls: "flush", sub: "evaluated by every replica every ALERT_EVAL_S; thresholds under Configuration → Monitoring & alerts",
      actions: h("a", { href: "#/config?tab=settings&group=Monitoring%20%26%20alerts" }, "Thresholds") }));
  };
  await load();
  let last = null;
  ctx.onLive((snap) => {
    const key = `${snap.alerts.open}:${snap.alerts.latestId}:${snap.alerts.unacked}`;
    if (last !== null && key !== last) load().catch(() => {});
    last = key;
  });
}
