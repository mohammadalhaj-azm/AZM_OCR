// Configuration: settings (live / restart), clients and their keys, processing profiles, ID rules and document
// types, and the version history (diff, restore). Viewers see everything read-only.
import { api } from "../api.js";
import { h, clear, card, table, badge, select, seg, modal, confirm, toast, showSecret, empty, kv, errorBox, fill } from "../ui.js";
import { dateTime, ago, DOC_TYPE_LABELS, label as labelOf } from "../format.js";

const TABS = [["settings", "Settings"], ["clients", "Clients & keys"], ["profiles", "Profiles"], ["rules", "ID rules & document types"],
  ["history", "History"]];

export async function render(root, ctx) {
  const tab = ctx.params.tab || "settings";
  root.append(h("nav", { class: "tabs" }, TABS.map(([id, text]) => h("a", { href: `#/config?tab=${id}`, class: id === tab ? "on" : "" }, text))));
  const body = h("div", { class: "grid" });
  root.append(body);
  const fn = { settings: renderSettings, clients: renderClients, profiles: renderProfiles, rules: renderRules, history: renderHistory }[tab] || renderSettings;
  return fn(body, ctx);
}

function failed(e) {
  toast("Not saved", e.message, "critical");
}

function persistBanner(persistent) {
  return persistent ? null : h("div", { class: "banner" }, "Memory job store (development): changes made here apply now "
    + "but are lost when the service restarts. Use JOB_STORE=postgres to keep them.");
}

// =================================================================================================================
// Settings
// =================================================================================================================
async function renderSettings(body, ctx) {
  const [d, rules] = await Promise.all([api.get("/config/settings"), api.get("/config/rules")]);
  if (ctx.stale()) return;
  const ORDER = ["OCR & languages", "AI models", "ID detection", "Limits & queue", "Retention", "Webhooks",
    "Monitoring & alerts", "Service & infrastructure"];
  const groups = [...new Set(d.items.map((r) => r.group))].sort((a, b) => ORDER.indexOf(a) - ORDER.indexOf(b));
  const group = ctx.params.group || groups.find((g) => d.items.some((r) => r.group === g && r.mode !== "locked")) || groups[0];
  const showLocked = ctx.params.locked === "1";
  const edits = {};
  body.append(...[persistBanner(d.persistent),
    d.error ? h("div", { class: "banner critical" }, `Version ${d.error.version} (${d.error.kind}) could not be applied on this replica: ${d.error.error}`) : null,
    Object.keys(d.pendingRestart).length ? h("div", { class: "banner" }, `Restart required to apply: ${Object.keys(d.pendingRestart).join(", ")}.`) : null].filter(Boolean));
  body.append(h("div", { class: "filters" },
    select(groups.map((g) => [g, g]), group, (v) => ctx.navigate("config", { tab: "settings", group: v, locked: showLocked ? "1" : "" })),
    h("label", { class: "check" }, h("input", { type: "checkbox", checked: showLocked,
      onchange: (e) => ctx.navigate("config", { tab: "settings", group, locked: e.target.checked ? "1" : "" }) }), "Show read-only settings"),
    h("span", { class: "muted" }, "Live settings apply on every replica within seconds; restart settings when a replica starts.")));

  const rows = d.items.filter((r) => r.group === group && (showLocked || r.mode !== "locked"));
  const saveBtn = h("button", { class: "btn primary", type: "button", disabled: true }, "Save changes");
  const comment = h("input", { type: "text", placeholder: "Comment (optional)", maxLength: 500 });
  const refresh = () => {
    const n = Object.keys(edits).length;
    saveBtn.disabled = !n;
    saveBtn.textContent = n ? `Save ${n} change${n === 1 ? "" : "s"}` : "Save changes";
  };
  const ruleRefs = [String(rules.file.path), ...Object.keys(rules.sets).map((n) => `db:${n}`)];
  const list = h("div");
  for (const r of rows) {
    const row = h("div", { class: "setting-row" });
    const editable = ctx.isAdmin && r.mode !== "locked";
    const mark = (value) => {
      const same = String(value) === String(r.value);
      if (same) delete edits[r.key]; else edits[r.key] = value;
      row.classList.toggle("changed", !same);
      refresh();
    };
    let control;
    if (r.secret) control = h("span", { class: "secondary" }, r.value);
    else if (!editable) control = h("span", { class: "mono" }, r.value === null ? "–" : String(r.value));
    else if (r.type === "bool") control = select([["true", "true"], ["false", "false"]], String(r.value), (v) => mark(v === "true"));
    else if (r.key === "id_patterns_path") control = select(ruleRefs.map((x) => [x, x]), String(r.value), (v) => mark(v));
    else if (["int", "float", "int?"].includes(r.type)) {
      control = h("input", { type: "number", value: r.value ?? "", min: r.min ?? undefined, max: r.max ?? undefined,
        step: r.type === "float" ? "any" : "1", oninput: (e) => mark(e.target.value === "" ? null : Number(e.target.value)) });
    } else control = h("input", { type: "text", value: r.value ?? "", oninput: (e) => mark(e.target.value) });
    row.append(
      h("div", null, h("div", { class: "key" }, r.key), h("div", { class: "meta" }, `${r.env}${r.min !== null && r.min !== undefined ? ` · ${r.min}–${r.max}` : ""}`)),
      h("div", null, control, r.pending !== undefined ? h("div", { class: "meta" }, `pending restart: ${r.pending}`) : null,
        r.source === "dashboard" ? h("div", { class: "meta" }, `env value: ${r.envValue ?? "–"} `,
          editable ? h("a", { href: "#", onclick: (e) => { e.preventDefault(); edits[r.key] = null; row.classList.add("changed"); refresh(); } }, "reset") : null) : null),
      h("div", { class: "chips" }, badge(r.mode === "live" ? "Live" : r.mode === "restart" ? "Restart" : "Read-only",
        r.mode === "live" ? "good" : r.mode === "restart" ? "warning" : "neutral", { plain: true }),
      badge(r.source, r.source === "dashboard" ? "info" : "neutral", { plain: true })));
    list.append(row);
  }
  saveBtn.addEventListener("click", async () => {
    try {
      const res = await api.put("/config/settings", { patch: edits, comment: comment.value, baseVersion: d.version });
      toast("Settings saved", `version ${res.version}${Object.keys(res.pendingRestart).length ? " · some need a restart" : ""}`, "good");
      ctx.rerender();
    } catch (e) { failed(e); }
  });
  body.append(card(null, h("div", null, rows.length ? list : empty("Nothing to edit in this group (show read-only settings to see them)."),
    ctx.isAdmin ? h("div", { class: "sticky-bar" }, comment, saveBtn) : null), { cls: "flush" }));
}

// =================================================================================================================
// Clients & keys
// =================================================================================================================
async function renderClients(body, ctx) {
  const d = await api.get("/config/registry");
  if (ctx.stale()) return;
  const reg = d.registry;
  const can = ctx.isAdmin && d.editable;
  body.append(...[persistBanner(d.persistent), !d.editable ? h("div", { class: "banner" }, d.reason) : null,
    d.source === "file" ? h("div", { class: "banner info" }, "The registry comes from CLIENTS_FILE. The first change saved here "
      + "moves it into the dashboard (versioned in the database); CLIENTS_FILE is then no longer read.") : null].filter(Boolean));
  body.append(h("div", { class: "filters" },
    can ? h("button", { class: "btn primary", type: "button", onclick: () => editClient(ctx, d, null) }, "New client") : null,
    ctx.isAdmin ? h("a", { class: "btn", href: "/admin/api/config/export/registry", download: "clients.yaml" }, "Export YAML") : null,
    h("span", { class: "muted" }, `Source: ${d.source}${d.version ? ` · version ${d.version}` : ""}`)));
  const rows = Object.entries(reg.clients || {}).map(([id, c]) => ({ id, ...c }));
  body.append(card(null, table([
    { key: "id", label: "Client", render: (c) => h("div", null, h("div", null, c.name), h("div", { class: "mono muted" }, c.id)) },
    { key: "enabled", label: "Status", render: (c) => (c.enabled === false ? badge("Disabled", "neutral") : badge("Enabled", "good")) },
    { key: "scopes", label: "Scopes", render: (c) => (c.scopes || ["documents", "extract"]).join(", ") },
    { key: "profiles", label: "Profiles", render: (c) => (c.profiles || ["default"]).join(", ") },
    { key: "keys", label: "Keys", render: (c) => h("div", { class: "chips" }, (c.keys || []).map((k) => h("span", { class: "chip mono" }, `${k}…`))) },
    { key: "webhook", label: "Webhook", render: (c) => (c.webhook ? `${(c.webhook.allowed_hosts || []).join(", ")} · ${c.webhook.managed_secret ? "secret in dashboard" : `secret ${c.webhook.secret_env}`}` : "off") },
    { key: "limits", label: "Limits", render: (c) => [c.limits?.max_queued_jobs ? `queue ${c.limits.max_queued_jobs}` : null,
      c.limits?.extract_concurrent ? `sync ${c.limits.extract_concurrent}` : null].filter(Boolean).join(" · ") || "global" },
  ], rows, { onRow: (c) => editClient(ctx, d, c.id), emptyText: "No clients yet." }), { cls: "flush" }));
  if (ctx.params.client && reg.clients?.[ctx.params.client]) editClient(ctx, d, ctx.params.client);
}

function editClient(ctx, d, id) {
  const can = ctx.isAdmin && d.editable;
  const c = id ? JSON.parse(JSON.stringify(d.registry.clients[id])) : { name: "", scopes: ["documents", "extract"], profiles: ["default"], enabled: true };
  const profileNames = ["default", ...Object.keys(d.registry.profiles || {})];
  const f = {
    id: h("input", { type: "text", value: id || "", placeholder: "e.g. civil-registry", disabled: !!id, pattern: "[a-z0-9][a-z0-9._-]{0,63}" }),
    name: h("input", { type: "text", value: c.name || "", placeholder: "Display name" }),
    enabled: h("input", { type: "checkbox", checked: c.enabled !== false }),
    scopes: Object.fromEntries(d.scopes.map((sc) => [sc, h("input", { type: "checkbox", checked: (c.scopes || d.scopes).includes(sc) })])),
    profiles: Object.fromEntries(profileNames.map((p) => [p, h("input", { type: "checkbox", checked: (c.profiles || ["default"]).includes(p) })])),
    defaultProfile: null,
    maxQueued: h("input", { type: "number", min: 0, value: c.limits?.max_queued_jobs ?? "", placeholder: "global" }),
    extract: h("input", { type: "number", min: 1, value: c.limits?.extract_concurrent ?? "", placeholder: "global" }),
    hooks: h("input", { type: "text", value: (c.webhook?.allowed_hosts || []).join(", "), placeholder: "hooks.example.internal (comma separated)" }),
    window: h("input", { type: "number", min: 0, value: c.webhook?.retry_window_s ?? "", placeholder: "default" }),
    secretEnv: h("input", { type: "text", value: c.webhook?.secret_env || "", placeholder: c.webhook?.managed_secret ? "managed in the dashboard" : "WEBHOOK_SECRET_MY_CLIENT" }),
  };
  f.defaultProfile = select(profileNames.map((p) => [p, p]), (c.profiles || ["default"])[0], () => {});
  const keys = h("div", { class: "chips" });
  const drawKeys = (list) => {
    fill(keys, ...(list || []).map((k) => h("span", { class: "chip mono" }, `${k}…`, can && id ? h("button", {
      class: "btn ghost sm danger", type: "button", title: "Revoke", onclick: async () => {
        if (!await confirm("Revoke this key?", `Requests with key ${k}… get 401 at once, on every replica within seconds.`, { okText: "Revoke", danger: true })) return;
        try { await api.del(`/config/clients/${encodeURIComponent(id)}/keys/${k}`); toast("Key revoked", `${k}…`, "good"); list.splice(list.indexOf(k), 1); drawKeys(list); } catch (e) { failed(e); }
      } }, "×") : null)));
  };
  drawKeys(c.keys);
  const field = (text, el, hint) => h("label", { class: "field" }, h("span", null, text), el, hint ? h("span", { class: "muted" }, hint) : null);
  const form = h("div", { class: "grid" },
    h("div", { class: "grid cols-2" }, field("Client id", f.id, "Owns its jobs and appears in the logs: keep it stable."), field("Name", f.name)),
    h("div", { class: "filters" }, h("label", { class: "check" }, f.enabled, "Enabled (a disabled client's keys get 401)"),
      ...d.scopes.map((sc) => h("label", { class: "check" }, f.scopes[sc], `scope: ${sc}`))),
    h("div", { class: "filters" }, h("span", { class: "secondary" }, "Profiles:"), ...profileNames.map((p) => h("label", { class: "check" }, f.profiles[p], p)),
      h("span", { class: "secondary" }, "default:"), f.defaultProfile),
    h("div", { class: "grid cols-2" }, field("Max queued jobs", f.maxQueued), field("Max simultaneous sync extracts", f.extract)),
    h("h3", null, "Webhook"),
    h("div", { class: "grid cols-2" }, field("Allowed callback hosts", f.hooks, "Empty = callbacks off."), field("Retry window (s)", f.window)),
    field("Signing secret environment variable", f.secretEnv, c.webhook?.managed_secret ? "This client uses a secret generated in the dashboard. Leave empty to keep it." : "Or save first, then generate a secret here."),
    id ? h("div", null, h("h3", null, "API keys"), keys) : h("p", { class: "muted" }, "An API key is generated when the client is created; it is shown once."));
  const actions = [{ text: "Close" }];
  if (can && id) {
    actions.unshift({ text: "Delete client", cls: "danger", onClick: async () => {
      if (!await confirm(`Delete ${id}?`, "Its keys stop working and its webhooks stop. Its jobs and audit records stay. Prefer disabling it.", { okText: "Delete", danger: true })) return true;
      try { await api.del(`/config/clients/${encodeURIComponent(id)}`); toast("Client deleted", id, "good"); ctx.rerender(); return false; } catch (e) { failed(e); return true; }
    } });
    actions.splice(1, 0, { text: "Add API key", onClick: async () => {
      try { const r = await api.post(`/config/clients/${encodeURIComponent(id)}/keys`); showSecret(`New API key for ${id}`, r.key, "Give it to the client over a secure channel. It is shown only now; the service stores its hash."); c.keys = [...(c.keys || []), r.fingerprint]; drawKeys(c.keys); } catch (e) { failed(e); }
      return true;
    } }, { text: "Generate webhook secret", onClick: async () => {
      const hosts = f.hooks.value.split(",").map((x) => x.trim()).filter(Boolean);
      if (!hosts.length) { toast("Allowed hosts needed", "Enter the callback host names first.", "warning"); return true; }
      try { const r = await api.post(`/config/clients/${encodeURIComponent(id)}/webhook-secret`, { allowed_hosts: hosts }); showSecret(`Webhook secret for ${id}`, r.secret, "Give it to the receiver to verify X-Webhook-Signature. Stored encrypted; shown only now."); } catch (e) { failed(e); }
      return true;
    } });
  }
  if (can) {
    actions.push({ text: id ? "Save" : "Create client", cls: "primary", onClick: async () => {
      const cid = f.id.value.trim();
      const chosen = profileNames.filter((p) => f.profiles[p].checked);
      const def = f.defaultProfile.value;
      const profiles = chosen.includes(def) ? [def, ...chosen.filter((p) => p !== def)] : chosen;
      const hosts = f.hooks.value.split(",").map((x) => x.trim()).filter(Boolean);
      const limits = {};
      if (f.maxQueued.value !== "") limits.max_queued_jobs = Number(f.maxQueued.value);
      if (f.extract.value !== "") limits.extract_concurrent = Number(f.extract.value);
      const client = { name: f.name.value.trim(), enabled: f.enabled.checked, scopes: d.scopes.filter((sc) => f.scopes[sc].checked),
        profiles: profiles.length ? profiles : ["default"], limits,
        webhook: hosts.length ? { allowed_hosts: hosts, retry_window_s: f.window.value === "" ? null : Number(f.window.value), secret_env: f.secretEnv.value.trim() || null } : null };
      try {
        const r = await api.put(`/config/clients/${encodeURIComponent(cid)}`, { client, baseVersion: d.version });
        toast(id ? "Client saved" : "Client created", `version ${r.version}`, "good");
        if (r.key) showSecret(`API key for ${cid}`, r.key, "Give it to the client over a secure channel. It is shown only now; the service stores its hash.");
        ctx.navigate("config", { tab: "clients" });
        return false;
      } catch (e) { failed(e); return true; }
    } });
  }
  modal(id ? `Client ${id}` : "New client", form, actions, { wide: true });
}

// =================================================================================================================
// Profiles
// =================================================================================================================
async function renderProfiles(body, ctx) {
  const d = await api.get("/config/registry");
  if (ctx.stale()) return;
  const can = ctx.isAdmin && d.editable;
  const profiles = d.registry.profiles || {};
  const users = (name) => Object.entries(d.registry.clients || {}).filter(([, c]) => (c.profiles || ["default"]).includes(name)).map(([id]) => id);
  body.append(...[persistBanner(d.persistent), !d.editable ? h("div", { class: "banner" }, d.reason) : null].filter(Boolean));
  body.append(h("div", { class: "filters" }, can ? h("button", { class: "btn primary", type: "button", onclick: () => editProfile(ctx, d, null) }, "New profile") : null,
    h("span", { class: "muted" }, "A profile changes how a client's documents are processed, what is returned and kept. Unset fields keep the global value.")));
  const rows = [{ name: "default", spec: {}, builtin: true }, ...Object.entries(profiles).map(([name, spec]) => ({ name, spec: spec || {} }))];
  body.append(card(null, table([
    { key: "name", label: "Profile", render: (r) => h("span", { class: "mono" }, r.name) },
    { key: "spec", label: "Settings", nosort: true, render: (r) => (r.builtin ? h("span", { class: "muted" }, "built in: the global settings")
      : h("div", { class: "chips" }, Object.entries(r.spec).map(([k, v]) => h("span", { class: "chip" }, `${k}: ${v}`)))) },
    { key: "users", label: "Used by", render: (r) => users(r.name).join(", ") || "–" },
  ], rows, { onRow: (r) => (r.builtin ? null : editProfile(ctx, d, r.name)) }), { cls: "flush" }));
}

function editProfile(ctx, d, name) {
  const can = ctx.isAdmin && d.editable;
  const spec = name ? { ...(d.registry.profiles[name] || {}) } : {};
  const nameInput = h("input", { type: "text", value: name || "", disabled: !!name, placeholder: "e.g. civil-ids" });
  const inputs = {};
  const fields = d.profileSchema.map((f) => {
    let el;
    if (f.type.includes("bool")) {
      el = select([["", "inherit / default"], ["true", "true"], ["false", "false"]], spec[f.key] === undefined ? "" : String(spec[f.key]), () => {});
    } else if (f.key === "id_patterns") {
      el = select([["", "inherit (ID_PATTERNS_PATH)"], ...d.ruleSets.map((r) => [r, r]), ...(spec.id_patterns && !d.ruleSets.includes(spec.id_patterns) ? [[spec.id_patterns, spec.id_patterns]] : [])],
        spec.id_patterns || "", () => {});
    } else {
      el = h("input", { type: f.type.includes("int") ? "number" : "text", value: spec[f.key] ?? "", placeholder: "inherit" });
    }
    el.disabled = !can;
    inputs[f.key] = { el, type: f.type };
    return h("label", { class: "field" }, h("span", null, labelOf(f.key)), el, f.description ? h("span", { class: "muted" }, f.description) : null);
  });
  const actions = [{ text: "Close" }];
  if (can && name) {
    actions.unshift({ text: "Delete", cls: "danger", onClick: async () => {
      if (!await confirm(`Delete profile ${name}?`, "Only possible when no client uses it.", { okText: "Delete", danger: true })) return true;
      try { await api.del(`/config/profiles/${encodeURIComponent(name)}`); toast("Profile deleted", name, "good"); ctx.rerender(); return false; } catch (e) { failed(e); return true; }
    } });
  }
  if (can) {
    actions.push({ text: "Save", cls: "primary", onClick: async () => {
      const out = {};
      for (const [k, { el, type }] of Object.entries(inputs)) {
        const v = el.value;
        if (v === "") continue;
        out[k] = type.includes("bool") ? v === "true" : type.includes("int") ? Number(v) : v;
      }
      try {
        const r = await api.put(`/config/profiles/${encodeURIComponent(nameInput.value.trim())}`, { profile: out, baseVersion: d.version });
        toast("Profile saved", `version ${r.version}`, "good");
        ctx.rerender();
        return false;
      } catch (e) { failed(e); return true; }
    } });
  }
  modal(name ? `Profile ${name}` : "New profile", h("div", { class: "grid" }, h("label", { class: "field" }, h("span", null, "Name"), nameInput),
    h("div", { class: "grid cols-2" }, fields)), actions, { wide: true });
}

// =================================================================================================================
// ID rules & document types
// =================================================================================================================
async function renderRules(body, ctx) {
  const d = await api.get("/config/rules");
  if (ctx.stale()) return;
  const sum = d.summary;
  body.append(...[persistBanner(d.persistent)].filter(Boolean));
  body.append(h("div", { class: "grid cols-2" },
    card("Active rules", sum.error ? errorBox(sum.error) : kv([["In use", h("span", { class: "mono" }, d.active)], ["Version", d.activeVersion || "–"],
      ["Keywords", `${sum.keywords} (+${sum.negativeKeywords} negative)`], ["ID-number patterns", `${sum.patterns} · countries ${sum.countries.join(", ") || "–"}`]]),
    { actions: ctx.isAdmin ? h("a", { href: "#/config?tab=settings&group=ID%20detection" }, "Change") : null }),
    card("Document types", sum.byDocType ? table([
      { key: "type", label: "Type", render: (r) => DOC_TYPE_LABELS[r.type] || r.type },
      { key: "keywords", label: "Keywords", cls: "r" }, { key: "patterns", label: "Patterns", cls: "r" },
    ], d.docTypes.map((t) => ({ type: t, ...(sum.byDocType[t] || { keywords: 0, patterns: 0 }) }))) : empty(), { cls: "flush",
      sub: "the classes the detector decides between (fixed); rules give them evidence" })));

  const sets = Object.entries(d.sets);
  body.append(h("div", { class: "filters" }, ctx.isAdmin ? h("button", { class: "btn primary", type: "button",
    onclick: () => editRules(ctx, null, d.file.yaml) }, "New rule set from the file") : null,
  h("span", { class: "muted" }, "Rule sets are referenced as db:<name> by ID_PATTERNS_PATH or a profile.")));
  body.append(card("Rule sets", table([
    { key: "name", label: "Name", render: (r) => h("span", { class: "mono" }, r.ref) },
    { key: "src", label: "Stored in", render: (r) => (r.file ? "file (read-only here)" : `dashboard · v${r.id}`) },
    { key: "at", label: "Changed", render: (r) => (r.at ? `${ago(r.at)} by ${r.actor}` : "–") },
    { key: "usedBy", label: "Used by", render: (r) => (r.usedBy || []).join(", ") || (r.ref === d.active ? "settings: id_patterns_path" : "–") },
  ], [{ ref: d.file.path, file: true }, ...sets.map(([name, v]) => ({ ref: `db:${name}`, name, ...v }))],
  { onRow: (r) => (r.file ? editRules(ctx, null, d.file.yaml, true) : editRules(ctx, r.name, r.yaml, false, r)) }), { cls: "flush" }));
}

function editRules(ctx, name, yaml, readOnly = false, meta = null) {
  const can = ctx.isAdmin && !readOnly;
  const nameInput = h("input", { type: "text", value: name || "", disabled: !!name || readOnly, placeholder: "e.g. civil" });
  const area = h("textarea", { rows: 22, spellcheck: false, readOnly: !can }, yaml || "");
  const result = h("div");
  const validate = async () => {
    const r = await api.post("/config/rules/validate", { yaml: area.value });
    fill(result, r.ok ? h("div", { class: "banner info" }, `Valid: ${r.summary.keywords} keywords, ${r.summary.negativeKeywords} negative, `
      + `${r.summary.patterns} patterns (${r.summary.countries.join(", ") || "no countries"}).`) : h("div", { class: "banner critical" }, r.error));
    return r.ok;
  };
  const actions = [{ text: "Close" }, { text: "Validate", onClick: async () => { await validate(); return true; } }];
  if (ctx.isAdmin && name && meta) {
    actions.unshift({ text: "Delete", cls: "danger", onClick: async () => {
      if (!await confirm(`Delete rule set ${name}?`, "Only possible when nothing uses it.", { okText: "Delete", danger: true })) return true;
      try { await api.del(`/config/rules/${encodeURIComponent(name)}`); toast("Rule set deleted", name, "good"); ctx.rerender(); return false; } catch (e) { failed(e); return true; }
    } });
    actions.push({ text: "Use as default", onClick: async () => {
      try { await api.put("/config/settings", { patch: { id_patterns_path: `db:${name}` }, comment: `rules ${name} as default` }); toast("Default rules changed", `db:${name}`, "good"); ctx.rerender(); return false; } catch (e) { failed(e); return true; }
    } });
  }
  if (can) {
    actions.push({ text: "Save", cls: "primary", onClick: async () => {
      if (!await validate()) return true;
      const n = nameInput.value.trim();
      try {
        await api.put(`/config/rules/${encodeURIComponent(n)}`, { yaml: area.value, baseVersion: meta?.id });
        toast("Rule set saved", `db:${n}`, "good");
        ctx.rerender();
        return false;
      } catch (e) { failed(e); return true; }
    } });
  }
  modal(readOnly ? "Rules file (read-only)" : name ? `Rule set ${name}` : "New rule set", h("div", { class: "grid" },
    h("label", { class: "field" }, h("span", null, "Name"), nameInput), area, result,
    h("p", { class: "muted" }, "keywords / negative_keywords: {text, weight 0-1, doc_type}; countries: {CC: [{name, pattern, weight, doc_type}]}. "
      + "The file is validated before it is saved; every replica picks it up within seconds.")), actions, { wide: true });
}

// =================================================================================================================
// History
// =================================================================================================================
function diffLines(a, b) {
  const x = a.split("\n"); const y = b.split("\n");
  if (x.length * y.length > 4e6) return null;
  const dp = Array.from({ length: x.length + 1 }, () => new Uint32Array(y.length + 1));
  for (let i = x.length - 1; i >= 0; i--) for (let j = y.length - 1; j >= 0; j--) dp[i][j] = x[i] === y[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0; let j = 0;
  while (i < x.length && j < y.length) {
    if (x[i] === y[j]) { out.push([" ", x[i]]); i++; j++; } else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push(["-", x[i++]]); } else { out.push(["+", y[j++]]); }
  }
  while (i < x.length) out.push(["-", x[i++]]);
  while (j < y.length) out.push(["+", y[j++]]);
  return out;
}

async function renderHistory(body, ctx) {
  const kind = ctx.params.kind || "";
  ctx.controls.append(seg([["", "All"], ["settings", "Settings"], ["registry", "Clients & profiles"], ["rules:", "Rules"]], kind,
    (v) => ctx.navigate("config", { tab: "history", kind: v })));
  const d = await api.get("/config/history", { kind });
  if (ctx.stale()) return;
  if (d.error) body.append(h("div", { class: "banner critical" }, `Not applied on this replica: version ${d.error.version} (${d.error.kind}): ${d.error.error}`));
  body.append(card(null, table([
    { key: "id", label: "Version", cls: "r" },
    { key: "at", label: "Saved", render: (r) => dateTime(r.at) },
    { key: "kind", label: "What", render: (r) => (r.kind === "registry" ? "clients & profiles" : r.kind) },
    { key: "actor", label: "By" },
    { key: "comment", label: "Comment" },
    { key: "applied", label: "", nosort: true, render: (r) => (d.applied[r.kind] === r.id ? badge("current", "good") : null) },
  ], d.items, { onRow: (r) => showVersion(ctx, r, d.items), emptyText: "No configuration was changed in the dashboard yet." }), { cls: "flush" }));
}

async function showVersion(ctx, row, items) {
  const v = await api.get(`/config/versions/${row.id}`);
  const prevRow = items.find((x) => x.kind === row.kind && x.id < row.id);
  const prev = prevRow ? await api.get(`/config/versions/${prevRow.id}`) : null;
  const now = JSON.stringify(v.content, null, 2);
  const before = prev ? JSON.stringify(prev.content, null, 2) : "";
  const lines = prev ? diffLines(before, now) : null;
  const view = lines ? h("div", { class: "json" }, lines.filter(([op], i) => op !== " " || lines.slice(Math.max(0, i - 2), i + 3).some(([o]) => o !== " "))
    .map(([op, text]) => h("div", { class: op === "+" ? "level-WARNING" : op === "-" ? "level-ERROR" : "muted" }, `${op} ${text}`)))
    : h("div", { class: "json" }, now);
  const actions = [{ text: "Close" }];
  if (ctx.isAdmin) {
    actions.push({ text: `Restore version ${row.id}`, cls: "primary", onClick: async () => {
      if (!await confirm(`Restore version ${row.id}?`, "Its content is saved as a new version and applied on every replica.", { okText: "Restore" })) return true;
      try { const r = await api.post(`/config/versions/${row.id}/restore`); toast("Restored", `as version ${r.version}`, "good"); ctx.rerender(); return false; } catch (e) { failed(e); return true; }
    } });
  }
  modal(`Version ${row.id} · ${row.kind}`, h("div", { class: "grid" }, h("p", { class: "secondary" },
    `${dateTime(row.at)} by ${row.actor}${row.comment ? ` · ${row.comment}` : ""}${prev ? ` · changes since version ${prev.id}` : " · first version"}`), view), actions, { wide: true });
}
