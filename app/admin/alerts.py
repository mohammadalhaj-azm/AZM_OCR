"""Alerts for administrators, shown in the dashboard (no outbound notifications).

Every replica evaluates the rules every ALERT_EVAL_S. An alert is identified by (rule, scope); the store keeps at most
one open alert per pair, so replicas evaluating the same global rule update one alert instead of adding duplicates.

| kind | scope | evaluated by |
|---|---|---|
| global | '' | every replica (shared data: job store, audit trail) |
| client | client:<id> | every replica (audit trail) |
| instance | <instance>[|<part>] | the replica itself (its dependencies and resources) |
| peer | <other instance> | every replica (a replica stopped sending samples) |

Resource rules (CPU, memory, disk, GPU memory) fire only after the threshold was exceeded for ALERT_FOR_S. A rule
that couldn't be evaluated (e.g. the database is down) leaves its alerts as they are.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger("app.admin")

Kind = Literal["global", "client", "instance", "peer"]
FAILURE_WINDOW_S = 900
SHEDDING_WINDOW_S, SHEDDING_MIN = 600, 20
WEBHOOK_WINDOW_S = 3600
WEBHOOKS_PILING = 100
AUDIT_FAILURE_HOLD_S = 600
REPLICA_MISSING_S = 60


@dataclass(frozen=True)
class Rule:
    name: str
    kind: Kind
    severity: str
    title: str


RULES = {r.name: r for r in [
    Rule("ocr_models_missing", "instance", "critical", "OCR models missing"),
    Rule("worker_down", "instance", "critical", "Job worker not running"),
    Rule("llm_unavailable", "instance", "warning", "Text LLM unavailable"),
    Rule("vision_unavailable", "instance", "warning", "Vision model unavailable"),
    Rule("audit_write_failures", "instance", "critical", "Audit events could not be written"),
    Rule("config_apply_failed", "instance", "critical", "Configuration change could not be applied"),
    Rule("cpu_high", "instance", "warning", "CPU usage high"),
    Rule("memory_high", "instance", "warning", "Memory usage high"),
    Rule("disk_high", "instance", "warning", "Disk almost full"),
    Rule("gpu_memory_high", "instance", "warning", "GPU memory almost full"),
    Rule("queue_backlog", "global", "warning", "Queue backlog"),
    Rule("job_stuck", "global", "critical", "Job stuck in processing"),
    Rule("job_failure_rate", "global", "warning", "High job failure rate"),
    Rule("server_errors", "global", "warning", "Server errors (500)"),
    Rule("model_fallbacks", "global", "warning", "AI model fallbacks"),
    Rule("webhooks_piling", "global", "warning", "Webhook deliveries piling up"),
    Rule("load_shedding", "client", "warning", "Client requests rejected (503)"),
    Rule("webhooks_failed", "client", "warning", "Webhook deliveries given up"),
    Rule("replica_missing", "peer", "warning", "Replica stopped reporting"),
]}


@dataclass
class Finding:
    rule: str
    scope: str
    firing: bool
    summary: str = ""
    details: dict | None = None


class AlertEvaluator:
    def __init__(self, settings, store, jobs, instance: str, latest_sample: Callable[[], dict | None],
                 audit_failures: Callable[[], float], config_error: Callable[[], dict | None],
                 clients: Callable[[], list[str]]):
        self.s, self.store, self.jobs, self.instance = settings, store, jobs, instance
        self._sample, self._audit_failures, self._config_error, self._clients = (latest_sample, audit_failures,
                                                                                 config_error, clients)
        self._breach: dict[tuple[str, str], float] = {}
        self._audit_seen = (audit_failures(), 0.0)  # (counter value, last time it grew)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- rules -------------------------------------------------------------------------------------------------
    def _sustained(self, rule: str, scope: str, breached: bool, now: float) -> bool:
        key = (rule, scope)
        if not breached:
            self._breach.pop(key, None)
            return False
        first = self._breach.setdefault(key, now)
        return now - first >= self.s.alert_for_s

    def _instance_rules(self, now: float) -> list[Finding]:
        out, me, s = [], self.instance, self.s
        sample = self._sample() or {}
        health = sample.get("health") or {}
        if health:
            out.append(Finding("ocr_models_missing", me, health.get("ocrModels") is False,
                               "The OCR models are not on disk: scanned pages can't be read"))
            out.append(Finding("worker_down", me, health.get("worker") is False,
                               "No job worker thread is running: queued documents are not processed"))
            out.append(Finding("llm_unavailable", me, s.llm_enabled and health.get("llm") is False,
                               f"{s.llm_text_model} is not available: ID detection runs on rules only"))
            out.append(Finding("vision_unavailable", me,
                               s.llm_enabled and s.vision_enabled and health.get("vision") is False,
                               f"{s.llm_vision_model} is not available: no vision fallback for uncertain documents"))
        value = self._audit_failures()
        seen, grew_at = self._audit_seen
        if value > seen:
            self._audit_seen = (value, now)
            grew_at = now
        out.append(Finding("audit_write_failures", me, bool(grew_at) and now - grew_at < AUDIT_FAILURE_HOLD_S,
                           "Audit events could not be written (database?): the audit trail has gaps",
                           {"total": value}))
        err = self._config_error()
        out.append(Finding("config_apply_failed", me, err is not None,
                           f"Configuration version {err['version']} ({err['kind']}) was not applied: {err['error']}"
                           if err else "", err))
        if sample:
            cpu = (sample.get("cpu") or {}).get("percent")
            out.append(Finding("cpu_high", me, self._sustained("cpu_high", me, (cpu or 0) >= s.alert_cpu_pct, now),
                               f"CPU at {cpu}% for over {s.alert_for_s // 60} min", {"percent": cpu}))
            mem = (sample.get("memory") or {}).get("percent")
            out.append(Finding("memory_high", me, self._sustained("memory_high", me, (mem or 0) >= s.alert_mem_pct,
                                                                  now),
                               f"Memory at {mem}% for over {s.alert_for_s // 60} min", {"percent": mem}))
            for d in sample.get("disks") or []:
                scope = f"{me}|{d['label']}"
                out.append(Finding("disk_high", scope,
                                   self._sustained("disk_high", scope, d["percent"] >= s.alert_disk_pct, now),
                                   f"Disk {d['label']} is {d['percent']}% full", {"percent": d["percent"],
                                                                                   "freeBytes": d["free"]}))
            for g in sample.get("gpus") or []:
                if g.get("memPercent") is None:
                    continue
                scope = f"{me}|gpu{g['index']}"
                out.append(Finding("gpu_memory_high", scope, self._sustained(
                    "gpu_memory_high", scope, g["memPercent"] >= s.alert_gpu_mem_pct, now),
                    f"GPU {g['index']} memory at {g['memPercent']}%", {"percent": g["memPercent"]}))
        return out

    def _global_rules(self, now: float) -> list[Finding]:
        s, out = self.s, []
        ages = self.jobs.ages()
        out.append(Finding("queue_backlog", "", ages["queued"] > s.alert_queue_age_s,
                           f"The oldest queued job has waited {round(ages['queued'] / 60)} min",
                           {"oldestQueuedS": round(ages["queued"])}))
        out.append(Finding("job_stuck", "", ages["processing"] > s.alert_job_stuck_s,
                           f"A job has been processing for {round(ages['processing'] / 60)} min",
                           {"oldestProcessingS": round(ages["processing"])}))
        pending = self.jobs.count_pending_webhooks()
        out.append(Finding("webhooks_piling", "", pending > WEBHOOKS_PILING,
                           f"{pending} webhook deliveries are waiting for their receivers", {"pending": pending}))
        st = self.store.stats(now - FAILURE_WINDOW_S)
        done, failed = st["events"].get("job.done", 0), st["events"].get("job.failed", 0)
        rate = failed / (done + failed) if done + failed else 0.0
        out.append(Finding("job_failure_rate", "", done + failed >= 5 and rate > s.alert_failure_rate,
                           f"{round(rate * 100)}% of jobs failed in the last 15 min ({failed} of {done + failed})",
                           {"failed": failed, "finished": done + failed, "errors": st["dist"]["error"]}))
        n500 = st["dist"]["httpStatus"].get("500", 0)
        out.append(Finding("server_errors", "", n500 > 0, f"{n500} requests failed with 500 in the last 15 min",
                           {"count": n500}))
        fb = st["dist"]["fallback"]
        out.append(Finding("model_fallbacks", "", bool(fb),
                           "ID detection went on without " + ", ".join(f"{k} ({n}x)" for k, n in fb.items())
                           + " in the last 15 min", {"fallbacks": fb}))
        return out

    def _client_rules(self, now: float) -> list[Finding]:
        out = []
        recent, hour = self.store.client_usage(now - SHEDDING_WINDOW_S), self.store.client_usage(now - WEBHOOK_WINDOW_S)
        for client in set(self._clients()) | set(recent) | set(hour):
            if not client:
                continue
            scope = f"client:{client}"
            n503 = (recent.get(client) or {}).get("httpStatus", {}).get("503", 0)
            out.append(Finding("load_shedding", scope, n503 >= SHEDDING_MIN,
                               f"{client} got {n503} × 503 (queue full / busy) in 10 min", {"count": n503}))
            failed = (hour.get(client) or {}).get("events", {}).get("webhook.failed", 0)
            out.append(Finding("webhooks_failed", scope, failed > 0,
                               f"{failed} webhook deliveries to {client} were given up in the last hour",
                               {"count": failed}))
        return out

    def _peer_rules(self, now: float) -> list[Finding]:
        out = []
        retention = self.s.sample_retention_h * 3600
        for row in self.store.latest_samples():
            if row["instance"] == self.instance:
                continue
            age = now - row["at"]
            stopped = bool(row["data"].get("stopping"))
            missing = not stopped and REPLICA_MISSING_S < age < retention
            host = row["data"].get("host", "?")
            out.append(Finding("replica_missing", row["instance"], missing,
                               f"Replica {host} ({row['instance']}) sent no sample for {round(age)} s",
                               {"lastSeen": row["at"], "host": host}))
        return out

    # --- evaluation --------------------------------------------------------------------------------------------
    def evaluate(self, now: float | None = None) -> tuple[list[Finding], set[str]]:
        """All findings, and the rule kinds that could be evaluated."""
        now = time.time() if now is None else now
        findings, ok = [], set()
        for kind, fn in (("instance", self._instance_rules), ("global", self._global_rules),
                         ("client", self._client_rules), ("peer", self._peer_rules)):
            try:
                findings += fn(now)
                ok.add(kind)
            except Exception as e:  # noqa: BLE001 - e.g. database down: keep those alerts as they are
                log.warning("alert rules not evaluated kind=%s error=%s", kind, type(e).__name__)
        return findings, ok

    def _owns(self, rule: Rule, scope: str) -> bool:
        if rule.kind == "instance":
            return scope == self.instance or scope.startswith(self.instance + "|")
        return rule.kind in ("global", "client") or (rule.kind == "peer" and scope != self.instance)

    def run_once(self, now: float | None = None) -> list[int]:
        """Evaluate, open/refresh firing alerts, resolve the rest. Returns the ids of newly opened alerts."""
        findings, ok = self.evaluate(now)
        opened, firing = [], set()
        for f in findings:
            if not f.firing:
                continue
            rule = RULES[f.rule]
            firing.add((f.rule, f.scope))
            try:
                alert_id, new = self.store.alert_fire(f.rule, f.scope, rule.severity, f.summary, f.details or {})
            except Exception as e:  # noqa: BLE001
                log.warning("alert write failed rule=%s error=%s", f.rule, type(e).__name__)
                continue
            if new:
                opened.append(alert_id)
                log.warning("alert opened rule=%s scope=%s severity=%s", f.rule, f.scope or "-", rule.severity)
        try:
            open_alerts = self.store.alerts(state="open", limit=1000)
        except Exception:  # noqa: BLE001
            return opened
        for a in open_alerts:
            rule = RULES.get(a["rule"])
            if rule is None or rule.kind not in ok or not self._owns(rule, a["scope"]):
                continue
            if (a["rule"], a["scope"]) not in firing and self.store.alert_resolve(a["rule"], a["scope"]):
                log.info("alert resolved rule=%s scope=%s", a["rule"], a["scope"] or "-")
        return opened

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="admin-alerts", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(max(1.0, self.s.alert_eval_s)):
            try:
                self.run_once()
            except Exception as e:  # noqa: BLE001
                log.error("alert evaluation failed error=%s", type(e).__name__)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
