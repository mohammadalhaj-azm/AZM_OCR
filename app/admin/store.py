"""The admin dashboard's data access: document history and statistics from the audit trail, replica samples, alerts,
stored log records and dashboard-managed configuration.

- `PgAdminStore` shares the PostgreSQL job store's connection pool (tables from migration 006).
- `MemoryAdminStore` (development) reads the in-memory job store and keeps everything else in RAM.

Everything returned here is metadata: ids, sizes, states, timings, decisions, error codes. Never document text,
file names, MRZ fields or caller metadata (the audit trail and job columns used here don't hold them).
"""

import itertools
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict, deque

from app.api.store import InMemoryJobStore, JobStore

DOC_EVENTS = ("job.received", "extract.done")
FINAL_EVENTS = ("job.done", "job.failed")
# A request.error on an upload route with one of these statuses is a submitted document that was rejected
REJECT_ROUTES = ("/v1/documents", "/v1/extract")
REJECT_STATUSES = (400, 409, 413, 415, 500, 503)
STAGES = {"extract": "extract", "llm": "llm", "vision": "vision", "llmQueue": "llmQueue"}
DIMENSIONS = ("docType", "decidedBy", "fallback", "method", "error", "httpError", "httpStatus", "confidence",
              "language")
SERIES_EVENTS = ("job.received", "job.done", "job.failed", "extract.done", "request.error")
MEMORY_SAMPLES = 50_000
MEMORY_LOGS = 20_000


def percentiles(values: list[float]) -> dict:
    """p50/p95/avg/max/n like PostgreSQL percentile_cont (linear interpolation)."""
    s = sorted(v for v in values if v is not None)
    if not s:
        return {"n": 0, "p50": None, "p95": None, "avg": None, "max": None}

    def pct(q: float) -> float:
        pos = (len(s) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    return {"n": len(s), "p50": round(pct(0.5), 1), "p95": round(pct(0.95), 1), "avg": round(sum(s) / len(s), 1),
            "max": round(s[-1], 1)}


def confidence_bin(c) -> str | None:
    if not isinstance(c, (int, float)):
        return None
    return str(min(int(c * 10), 9) + 1)  # 1..10, like width_bucket(c, 0, 1, 10)


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def document_view(event: dict, final: dict | None = None, live: str | None = None) -> dict:
    """One row of Document Tracking from its first audit event (+ the final event and live status of a job)."""
    kind = {"job.received": "job", "extract.done": "extract"}.get(event["event"], "rejected")
    data, fdata = event.get("data") or {}, (final or {}).get("data") or {}
    facts = fdata if kind == "job" else data
    if kind == "extract":
        status = "done"
    elif kind == "rejected":
        status = "rejected"
    elif final is not None:
        status = "done" if final["event"] == "job.done" else "failed"
    else:
        status = live or "deleted"
    det = facts.get("idDetection") or {}
    doc_id = event["jobId"] if kind == "job" else f"evt-{event['id']}"
    finished = final["at"] if final else (event["at"] if kind != "job" else None)
    return {
        "id": doc_id, "kind": kind, "jobId": event["jobId"] if kind == "job" else None,
        "clientId": event["clientId"], "requestId": event["requestId"], "receivedAt": event["at"],
        "finishedAt": finished, "status": status, "endpoint": data.get("endpoint") or (
            "extract" if (data.get("route") or "").endswith("extract") else "documents"),
        "size": data.get("size"), "mimeType": data.get("mimeType"), "sha256": data.get("sha256"),
        "profile": data.get("profile"), "ms": facts.get("ms"), "pages": facts.get("pages"),
        "pageMethods": facts.get("pageMethods"), "languages": facts.get("languages"),
        "docType": det.get("docType"), "isId": det.get("isId"), "confidence": det.get("confidence"),
        "decidedBy": det.get("decidedBy"), "errorCode": fdata.get("error") if kind == "job" else data.get("code"),
        "httpStatus": data.get("status") if kind == "rejected" else None, "attempt": fdata.get("attempt"),
    }


class AdminStore(ABC):
    def __init__(self, jobs: JobStore):
        self.jobs = jobs

    @property
    def persistent(self) -> bool:
        return False

    # --- documents (from the audit trail) ------------------------------------------------------------------
    @abstractmethod
    def documents(self, *, since: float | None = None, until: float | None = None, client_id: str | None = None,
                  kind: str | None = None, status: str | None = None, doc_type: str | None = None,
                  error: str | None = None, q: str | None = None, q_hash: str | None = None,
                  before: int | None = None, limit: int = 50) -> list[dict]:
        """Newest first. `before`: the cursor (the first event's id) of the last row of the previous page; the
        rows carry it as `cursor`. `q` matches a job id, request id or sha256; `q_hash` an externalIdHash."""

    @abstractmethod
    def event(self, event_id: int) -> dict | None: ...

    def job_events(self, job_id: str) -> list[dict]:
        return self.jobs.audit_events(job_id=job_id, limit=1000)

    # --- audit explorer --------------------------------------------------------------------------------------
    @abstractmethod
    def audit_page(self, *, event: str | None = None, client_id: str | None = None, job_id: str | None = None,
                   request_id: str | None = None, actor: str | None = None, sha256: str | None = None,
                   external_hash: str | None = None, since: float | None = None, until: float | None = None,
                   before: int | None = None, limit: int = 100) -> list[dict]:
        """Newest first; `event` is an exact name or a prefix ending in '.' (e.g. 'job.'); `before` an event id."""

    def latest_event_id(self) -> int:
        rows = self.audit_page(limit=1)
        return rows[0]["id"] if rows else 0

    # --- statistics ------------------------------------------------------------------------------------------
    @abstractmethod
    def stats(self, since: float, until: float | None = None, client_id: str | None = None) -> dict:
        """{events: {event: n}, pages, ids, durations: {job, extract}, stages: {ocrPerPage, extract, llm, vision,
        llmQueue}, dist: {dimension: {key: n}}}"""

    @abstractmethod
    def series(self, since: float, bucket_s: int, client_id: str | None = None) -> list[dict]:
        """[{t, event, n, pages, ms}] per bucket (t = bucket start, unix time) for SERIES_EVENTS."""

    @abstractmethod
    def client_usage(self, since: float) -> dict:
        """{client_id or '': {events: {event: n}, lastSeen, avgMs, httpStatus: {status: n}, errors: {code: n}}}"""

    @abstractmethod
    def live_by_owner(self) -> dict:
        """{owner: {queued, processing, webhooksPending}} from the job table."""

    def db_info(self) -> dict | None:
        return None

    # --- replica samples -------------------------------------------------------------------------------------
    @abstractmethod
    def write_sample(self, instance: str, data: dict) -> None: ...

    @abstractmethod
    def latest_samples(self) -> list[dict]:
        """[{instance, at, data}] the latest sample of every instance seen within the retention."""

    @abstractmethod
    def samples(self, instance: str, since: float, max_points: int = 600) -> list[dict]:
        """[{at, data}] oldest first, thinned to about max_points."""

    @abstractmethod
    def purge_samples_before(self, cutoff: float) -> int: ...

    # --- alerts ----------------------------------------------------------------------------------------------
    @abstractmethod
    def alert_fire(self, rule: str, scope: str, severity: str, summary: str, details: dict) -> tuple[int, bool]:
        """Open the alert, or refresh the open one. Returns (id, newly opened)."""

    @abstractmethod
    def alert_resolve(self, rule: str, scope: str) -> bool: ...

    @abstractmethod
    def alerts(self, *, state: str | None = None, limit: int = 200) -> list[dict]:
        """state: open (firing or acknowledged), resolved, or None (all). Newest first."""

    @abstractmethod
    def alert_ack(self, alert_id: int, actor: str) -> bool: ...

    # --- stored log records (WARNING+, every replica) --------------------------------------------------------
    @abstractmethod
    def write_logs(self, records: list[dict]) -> None: ...

    @abstractmethod
    def logs(self, *, level: str | None = None, logger: str | None = None, text: str | None = None,
             request_id: str | None = None, job_id: str | None = None, client_id: str | None = None,
             instance: str | None = None, since: float | None = None, before: int | None = None,
             limit: int = 200) -> list[dict]: ...

    @abstractmethod
    def purge_logs_before(self, cutoff: float) -> int: ...

    # --- dashboard-managed configuration ----------------------------------------------------------------------
    @abstractmethod
    def config_save(self, kind: str, content: dict, actor: str, comment: str = "") -> int: ...

    @abstractmethod
    def config_latest(self, kind: str) -> dict | None:
        """{id, at, kind, content, actor, comment} or None."""

    @abstractmethod
    def config_heads(self) -> dict[str, int]:
        """{kind: latest version id} of every kind."""

    @abstractmethod
    def config_history(self, kind: str | None = None, limit: int = 100) -> list[dict]:
        """Newest first, without content."""

    @abstractmethod
    def config_version(self, version_id: int) -> dict | None: ...

    @abstractmethod
    def secret_set(self, client_id: str, secret: str) -> None: ...

    @abstractmethod
    def secrets(self) -> dict[str, str]:
        """{client_id: webhook secret} generated in the dashboard."""


# =====================================================================================================================
class MemoryAdminStore(AdminStore):
    """Development only: reads the in-memory job store; samples, alerts, logs and config live in RAM."""

    def __init__(self, jobs: InMemoryJobStore):
        super().__init__(jobs)
        self._lock = threading.Lock()
        self._samples: deque[dict] = deque(maxlen=MEMORY_SAMPLES)
        self._alerts: list[dict] = []
        self._alert_ids = itertools.count(1)
        self._logs: deque[dict] = deque(maxlen=MEMORY_LOGS)
        self._log_ids = itertools.count(1)
        self._config: list[dict] = []
        self._config_ids = itertools.count(1)
        self._secrets: dict[str, str] = {}

    def _events(self) -> list[dict]:
        with self.jobs._lock:
            return [dict(e, data=dict(e["data"])) for e in self.jobs._events]

    # --- documents -------------------------------------------------------------------------------------------
    def documents(self, *, since=None, until=None, client_id=None, kind=None, status=None, doc_type=None,
                  error=None, q=None, q_hash=None, before=None, limit=50) -> list[dict]:
        events = self._events()
        finals: dict[str, dict] = {}
        for e in events:
            if e["event"] in FINAL_EVENTS and e["jobId"]:
                finals[e["jobId"]] = e
        with self.jobs._lock:
            live = {j.id: j.status for j in self.jobs._jobs.values()}
        out = []
        for e in reversed(events):
            if not self._is_doc_event(e):
                continue
            if before is not None and e["id"] >= before:
                continue
            if since is not None and e["at"] < since or until is not None and e["at"] >= until:
                continue
            if client_id is not None and e["clientId"] != client_id:
                continue
            if q or q_hash:
                d = e["data"]
                if not ((q and q in (e["jobId"], e["requestId"], d.get("sha256")))
                        or (q_hash and d.get("externalIdHash") == q_hash)):
                    continue
            doc = document_view(e, finals.get(e["jobId"]) if e["event"] == "job.received" else None,
                                live.get(e["jobId"]))
            if (kind and doc["kind"] != kind or status and doc["status"] != status
                    or doc_type and doc["docType"] != doc_type or error and doc["errorCode"] != error):
                continue
            out.append(doc | {"cursor": e["id"]})
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _is_doc_event(e: dict) -> bool:
        if e["event"] in DOC_EVENTS:
            return True
        d = e["data"]
        return e["event"] == "request.error" and d.get("route") in REJECT_ROUTES and d.get("status") in REJECT_STATUSES

    def event(self, event_id: int) -> dict | None:
        return next((e for e in self._events() if e["id"] == event_id), None)

    # --- audit -----------------------------------------------------------------------------------------------
    def audit_page(self, *, event=None, client_id=None, job_id=None, request_id=None, actor=None, sha256=None,
                   external_hash=None, since=None, until=None, before=None, limit=100) -> list[dict]:
        out = []
        for e in reversed(self._events()):
            d = e["data"]
            if (event and not (e["event"].startswith(event) if event.endswith(".") else e["event"] == event)
                    or client_id is not None and e["clientId"] != client_id
                    or job_id is not None and e["jobId"] != job_id
                    or request_id is not None and e["requestId"] != request_id
                    or actor is not None and d.get("actor") != actor
                    or sha256 is not None and d.get("sha256") != sha256
                    or external_hash is not None and d.get("externalIdHash") != external_hash
                    or since is not None and e["at"] < since or until is not None and e["at"] >= until
                    or before is not None and e["id"] >= before):
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return out

    # --- statistics ------------------------------------------------------------------------------------------
    def _window(self, since, until=None, client_id=None) -> list[dict]:
        return [e for e in self._events() if e["at"] >= since and (until is None or e["at"] < until)
                and (client_id is None or e["clientId"] == client_id)]

    def stats(self, since, until=None, client_id=None) -> dict:
        events = self._window(since, until, client_id)
        counts = Counter(e["event"] for e in events)
        dist: dict[str, Counter] = {d: Counter() for d in DIMENSIONS}
        pages = ids = 0
        job_ms, extract_ms = [], []
        stage_values: dict[str, list] = defaultdict(list)
        for e in events:
            d, ev = e["data"], e["event"]
            if ev in ("job.done", "extract.done"):
                pages += d.get("pages") or 0
                for method, n in (d.get("pageMethods") or {}).items():
                    dist["method"][method] += n
                for lang in d.get("languages") or []:
                    dist["language"][lang] += 1
                (job_ms if ev == "job.done" else extract_ms).append(_num(d.get("ms")))
                t = d.get("timingsMs") or {}
                for key, name in STAGES.items():
                    if _num(t.get(key)) is not None:
                        stage_values[name].append(t[key])
                if _num(t.get("ocr")) is not None and (_num(t.get("ocrPages")) or 0) > 0:
                    stage_values["ocrPerPage"].append(t["ocr"] / t["ocrPages"])
            if ev == "job.done":
                det = d.get("idDetection")
                if det:
                    ids += bool(det.get("isId"))
                    dist["docType"][det.get("docType")] += 1
                    if det.get("decidedBy"):
                        dist["decidedBy"][det["decidedBy"][-1]] += 1
                    if (b := confidence_bin(det.get("confidence"))) is not None:
                        dist["confidence"][b] += 1
                for f in d.get("fallbacks") or []:
                    dist["fallback"][f] += 1
            elif ev == "job.failed":
                dist["error"][d.get("error")] += 1
            elif ev == "request.error":
                dist["httpError"][d.get("code")] += 1
                dist["httpStatus"][str(d.get("status"))] += 1
        return {"events": dict(counts), "pages": pages, "ids": ids,
                "durations": {"job": percentiles(job_ms), "extract": percentiles(extract_ms)},
                "stages": {k: percentiles(stage_values.get(k, [])) for k in ("ocrPerPage", *STAGES.values())},
                "dist": {k: {str(key): n for key, n in v.items() if key is not None} for k, v in dist.items()}}

    def series(self, since, bucket_s, client_id=None) -> list[dict]:
        acc: dict[tuple, list] = {}
        for e in self._window(since, None, client_id):
            if e["event"] not in SERIES_EVENTS:
                continue
            t = math.floor(e["at"] / bucket_s) * bucket_s
            row = acc.setdefault((t, e["event"]), [0, 0, []])
            row[0] += 1
            row[1] += e["data"].get("pages") or 0
            if _num(e["data"].get("ms")) is not None:
                row[2].append(e["data"]["ms"])
        return [{"t": t, "event": ev, "n": n, "pages": p, "ms": round(sum(ms) / len(ms), 1) if ms else None}
                for (t, ev), (n, p, ms) in sorted(acc.items())]

    def client_usage(self, since) -> dict:
        out: dict = {}
        for e in self._window(since):
            u = out.setdefault(e["clientId"] or "", {"events": Counter(), "lastSeen": None, "ms": [],
                                                     "httpStatus": Counter(), "errors": Counter()})
            u["events"][e["event"]] += 1
            u["lastSeen"] = max(u["lastSeen"] or 0, e["at"])
            d = e["data"]
            if e["event"] == "job.done" and _num(d.get("ms")) is not None:
                u["ms"].append(d["ms"])
            if e["event"] == "request.error":
                u["httpStatus"][str(d.get("status"))] += 1
                u["errors"][d.get("code") or "-"] += 1
            if e["event"] == "job.failed":
                u["errors"][d.get("error") or "-"] += 1
        return {k: {"events": dict(u["events"]), "lastSeen": u["lastSeen"],
                    "avgMs": round(sum(u["ms"]) / len(u["ms"]), 1) if u["ms"] else None,
                    "httpStatus": dict(u["httpStatus"]), "errors": dict(u["errors"])} for k, u in out.items()}

    def live_by_owner(self) -> dict:
        out: dict = {}
        with self.jobs._lock:
            for j in self.jobs._jobs.values():
                o = out.setdefault(j.owner, {"queued": 0, "processing": 0, "webhooksPending": 0})
                if j.status in ("queued", "processing"):
                    o[j.status] += 1
                if j.webhook_next_at is not None:
                    o["webhooksPending"] += 1
        return out

    # --- samples ---------------------------------------------------------------------------------------------
    def write_sample(self, instance, data) -> None:
        with self._lock:
            self._samples.append({"instance": instance, "at": time.time(), "data": data})

    def latest_samples(self) -> list[dict]:
        with self._lock:
            latest = {}
            for s in self._samples:
                latest[s["instance"]] = s
            return list(latest.values())

    def samples(self, instance, since, max_points=600) -> list[dict]:
        with self._lock:
            rows = [{"at": s["at"], "data": s["data"]} for s in self._samples
                    if s["instance"] == instance and s["at"] >= since]
        step = max(1, math.ceil(len(rows) / max_points))
        return rows[::step]

    def purge_samples_before(self, cutoff) -> int:
        with self._lock:
            keep = [s for s in self._samples if s["at"] >= cutoff]
            n = len(self._samples) - len(keep)
            self._samples.clear()
            self._samples.extend(keep)
            return n

    # --- alerts ----------------------------------------------------------------------------------------------
    def _open(self, rule, scope) -> dict | None:
        return next((a for a in self._alerts if a["rule"] == rule and a["scope"] == scope
                     and a["resolvedAt"] is None), None)

    def alert_fire(self, rule, scope, severity, summary, details) -> tuple[int, bool]:
        now = time.time()
        with self._lock:
            a = self._open(rule, scope)
            if a is not None:
                a.update(severity=severity, summary=summary, details=details, lastSeenAt=now)
                return a["id"], False
            a = {"id": next(self._alert_ids), "rule": rule, "scope": scope, "severity": severity, "summary": summary,
                 "details": details, "startedAt": now, "lastSeenAt": now, "resolvedAt": None, "ackedBy": None,
                 "ackedAt": None}
            self._alerts.append(a)
            return a["id"], True

    def alert_resolve(self, rule, scope) -> bool:
        with self._lock:
            a = self._open(rule, scope)
            if a is None:
                return False
            a["resolvedAt"] = time.time()
            return True

    def alerts(self, *, state=None, limit=200) -> list[dict]:
        with self._lock:
            rows = [dict(a) for a in reversed(self._alerts)
                    if state is None or (state == "open") == (a["resolvedAt"] is None)]
        return rows[:limit]

    def alert_ack(self, alert_id, actor) -> bool:
        with self._lock:
            a = next((a for a in self._alerts if a["id"] == alert_id and a["resolvedAt"] is None), None)
            if a is None:
                return False
            a.update(ackedBy=actor, ackedAt=time.time())
            return True

    # --- logs ------------------------------------------------------------------------------------------------
    def write_logs(self, records) -> None:
        with self._lock:
            for r in records:
                self._logs.append(dict(r, id=next(self._log_ids)))

    def logs(self, *, level=None, logger=None, text=None, request_id=None, job_id=None, client_id=None,
             instance=None, since=None, before=None, limit=200) -> list[dict]:
        with self._lock:
            rows = list(self._logs)
        return filter_logs(reversed(rows), level=level, logger=logger, text=text, request_id=request_id,
                           job_id=job_id, client_id=client_id, instance=instance, since=since, before=before,
                           limit=limit)

    def purge_logs_before(self, cutoff) -> int:
        with self._lock:
            keep = [r for r in self._logs if r["at"] >= cutoff]
            n = len(self._logs) - len(keep)
            self._logs.clear()
            self._logs.extend(keep)
            return n

    # --- config ----------------------------------------------------------------------------------------------
    def config_save(self, kind, content, actor, comment="") -> int:
        with self._lock:
            v = {"id": next(self._config_ids), "at": time.time(), "kind": kind, "content": content, "actor": actor,
                 "comment": comment}
            self._config.append(v)
            return v["id"]

    def config_latest(self, kind) -> dict | None:
        with self._lock:
            return next((dict(v) for v in reversed(self._config) if v["kind"] == kind), None)

    def config_heads(self) -> dict[str, int]:
        with self._lock:
            return {v["kind"]: v["id"] for v in self._config}

    def config_history(self, kind=None, limit=100) -> list[dict]:
        with self._lock:
            rows = [{k: v[k] for k in ("id", "at", "kind", "actor", "comment")} for v in reversed(self._config)
                    if kind is None or v["kind"] == kind or (kind.endswith(":") and v["kind"].startswith(kind))]
        return rows[:limit]

    def config_version(self, version_id) -> dict | None:
        with self._lock:
            return next((dict(v) for v in self._config if v["id"] == version_id), None)

    def secret_set(self, client_id, secret) -> None:
        with self._lock:
            self._secrets[client_id] = secret

    def secrets(self) -> dict[str, str]:
        with self._lock:
            return dict(self._secrets)


LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def filter_logs(rows, *, level=None, logger=None, text=None, request_id=None, job_id=None, client_id=None,
                instance=None, since=None, before=None, limit=200) -> list[dict]:
    """Shared by the RAM buffer and the memory store: rows newest first."""
    floor = LEVELS.get((level or "").upper(), 0)
    text = (text or "").lower()
    out = []
    for r in rows:
        if (LEVELS.get(r["level"], 0) < floor or logger and not r["logger"].startswith(logger)
                or text and text not in r["msg"].lower() or request_id and r.get("request_id") != request_id
                or job_id and r.get("job_id") != job_id or client_id and r.get("client_id") != client_id
                or instance and r.get("instance") != instance or since is not None and r["at"] < since
                or before is not None and r["id"] >= before):
            continue
        out.append(r)
        if len(out) >= limit:
            break
    return out


# =====================================================================================================================
_EV = ("SELECT id, extract(epoch from at)::float8 AS at, event, client_id, job_id, request_id, data "
       "FROM ocr_audit_events")


def _event(r: dict) -> dict:
    return {"id": r["id"], "at": r["at"], "event": r["event"], "clientId": r["client_id"], "jobId": r["job_id"],
            "requestId": r["request_id"], "data": r["data"]}


class PgAdminStore(AdminStore):
    def __init__(self, jobs):
        super().__init__(jobs)
        self.pool = jobs.pool
        self.sealer = jobs.sealer

    @property
    def persistent(self) -> bool:
        return True

    def _all(self, sql: str, params=()) -> list[dict]:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()

    # --- documents -------------------------------------------------------------------------------------------
    def documents(self, *, since=None, until=None, client_id=None, kind=None, status=None, doc_type=None,
                  error=None, q=None, q_hash=None, before=None, limit=50) -> list[dict]:
        where, params = ["(r.event IN ('job.received', 'extract.done') OR (r.event = 'request.error' "
                         "AND r.data->>'route' = ANY(%s) AND (r.data->>'status')::int = ANY(%s)))"], \
            [list(REJECT_ROUTES), list(REJECT_STATUSES)]
        for cond, value in (("r.at >= to_timestamp(%s)", since), ("r.at < to_timestamp(%s)", until),
                            ("r.client_id = %s", client_id), ("r.id < %s", before)):
            if value is not None:
                where.append(cond)
                params.append(value)
        if kind:
            where.append("r.event = %s")
            params.append({"job": "job.received", "extract": "extract.done",
                           "rejected": "request.error"}.get(kind, "-"))
        if q or q_hash:
            where.append("(r.job_id = %s OR r.request_id = %s OR r.data->>'sha256' = %s "
                         "OR r.data->>'externalIdHash' = %s)")
            params += [q or "", q or "", q or "", q_hash or ""]
        outer, oparams = [], []
        for cond, value in (("status = %s", status),
                            ("coalesce(fdata, data)->'idDetection'->>'docType' = %s", doc_type),
                            ("coalesce(fdata->>'error', data->>'code') = %s", error)):
            if value:
                outer.append(cond)
                oparams.append(value)
        sql = f"""
            SELECT * FROM (
              SELECT r.id, extract(epoch from r.at)::float8 AS at, r.event, r.client_id, r.job_id, r.request_id,
                     r.data, f.id AS fid, f.event AS fevent, extract(epoch from f.at)::float8 AS fat, f.data AS fdata,
                     j.status AS live,
                     CASE WHEN r.event = 'extract.done' THEN 'done' WHEN r.event = 'request.error' THEN 'rejected'
                          WHEN f.event = 'job.done' THEN 'done' WHEN f.event = 'job.failed' THEN 'failed'
                          ELSE coalesce(j.status, 'deleted') END AS status
              FROM ocr_audit_events r
              LEFT JOIN LATERAL (SELECT e.id, e.event, e.at, e.data FROM ocr_audit_events e
                                 WHERE r.event = 'job.received' AND e.job_id = r.job_id
                                   AND e.event IN ('job.done', 'job.failed') ORDER BY e.id DESC LIMIT 1) f ON true
              LEFT JOIN ocr_jobs j ON r.event = 'job.received' AND j.id = r.job_id
              WHERE {' AND '.join(where)}
            ) d {('WHERE ' + ' AND '.join(outer)) if outer else ''}
            ORDER BY id DESC LIMIT %s"""
        out = []
        for r in self._all(sql, (*params, *oparams, limit)):
            final = ({"id": r["fid"], "event": r["fevent"], "at": r["fat"], "data": r["fdata"]}
                     if r["fevent"] else None)
            out.append(document_view(_event(r), final, r["live"]) | {"cursor": r["id"]})
        return out

    def event(self, event_id) -> dict | None:
        rows = self._all(_EV + " WHERE id = %s", (event_id,))
        return _event(rows[0]) if rows else None

    # --- audit -----------------------------------------------------------------------------------------------
    def audit_page(self, *, event=None, client_id=None, job_id=None, request_id=None, actor=None, sha256=None,
                   external_hash=None, since=None, until=None, before=None, limit=100) -> list[dict]:
        where, params = [], []
        if event:
            if event.endswith("."):
                where.append("event LIKE %s")
                params.append(event.replace("%", "") + "%")
            else:
                where.append("event = %s")
                params.append(event)
        for cond, value in (("client_id = %s", client_id), ("job_id = %s", job_id), ("request_id = %s", request_id),
                            ("data->>'actor' = %s", actor), ("data->>'sha256' = %s", sha256),
                            ("data->>'externalIdHash' = %s", external_hash), ("at >= to_timestamp(%s)", since),
                            ("at < to_timestamp(%s)", until), ("id < %s", before)):
            if value is not None:
                where.append(cond)
                params.append(value)
        sql = _EV + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT %s"
        return [_event(r) for r in self._all(sql, (*params, limit))]

    # --- statistics ------------------------------------------------------------------------------------------
    @staticmethod
    def _where(since, until, client_id) -> tuple[str, list]:
        where, params = ["at >= to_timestamp(%s)"], [since]
        if until is not None:
            where.append("at < to_timestamp(%s)")
            params.append(until)
        if client_id is not None:
            where.append("client_id = %s")
            params.append(client_id)
        return " AND ".join(where), params

    def stats(self, since, until=None, client_id=None) -> dict:
        w, p = self._where(since, until, client_id)
        done = "event IN ('job.done', 'extract.done')"
        pct = "percentile_cont(array[0.5, 0.95]) WITHIN GROUP (ORDER BY {expr})"
        def cols(expr: str, name: str, cond: str) -> str:  # PostgreSQL folds the aliases to lower case
            return (f"{pct.format(expr=expr)} FILTER (WHERE {cond}) AS s_{name}, count({expr}) FILTER (WHERE {cond}) "
                    f"AS n_{name}, avg({expr}) FILTER (WHERE {cond}) AS a_{name}, "
                    f"max({expr}) FILTER (WHERE {cond}) AS m_{name}")

        stage_cols = [cols(f"(data->'timingsMs'->>'{key}')::float8", name, done) for key, name in STAGES.items()]
        per_page = ("(data->'timingsMs'->>'ocr')::float8 / nullif((data->'timingsMs'->>'ocrPages')::float8, 0)")
        sql = f"""
            SELECT coalesce(sum((data->>'pages')::int) FILTER (WHERE {done}), 0) AS pages,
                   count(*) FILTER (WHERE event = 'job.done' AND (data->'idDetection'->>'isId')::boolean) AS ids,
                   {cols("(data->>'ms')::float8", "job", "event = 'job.done'")},
                   {cols("(data->>'ms')::float8", "extractreq", "event = 'extract.done'")},
                   {cols(per_page, "ocrperpage", done)},
                   {', '.join(stage_cols)}
            FROM ocr_audit_events WHERE {w} AND event IN ('job.done', 'extract.done')"""
        with self.pool.connection() as conn:
            counts = {r["event"]: r["n"] for r in conn.execute(
                f"SELECT event, count(*) AS n FROM ocr_audit_events WHERE {w} GROUP BY event", p).fetchall()}
            agg = conn.execute(sql, p).fetchone()
            dist_rows = conn.execute(self._dist_sql(w), p * 9).fetchall()

        def summary(name: str) -> dict:
            k = name.lower()
            s, n, a, m = agg[f"s_{k}"], agg[f"n_{k}"], agg[f"a_{k}"], agg[f"m_{k}"]
            if not n:
                return {"n": 0, "p50": None, "p95": None, "avg": None, "max": None}
            return {"n": n, "p50": round(s[0], 1), "p95": round(s[1], 1), "avg": round(a, 1), "max": round(m, 1)}

        dist: dict[str, dict] = {d: {} for d in DIMENSIONS}
        for r in dist_rows:
            if r["k"] is not None:
                dist[r["dim"]][r["k"]] = int(r["n"])
        return {"events": counts, "pages": int(agg["pages"]), "ids": agg["ids"],
                "durations": {"job": summary("job"), "extract": summary("extractreq")},
                "stages": {k: summary(k) for k in ("ocrPerPage", *STAGES.values())}, "dist": dist}

    @staticmethod
    def _dist_sql(w: str) -> str:
        t = "FROM ocr_audit_events"
        return f"""
            SELECT 'docType' AS dim, data->'idDetection'->>'docType' AS k, count(*) AS n {t}
              WHERE {w} AND event = 'job.done' AND data ? 'idDetection' GROUP BY 2
            UNION ALL SELECT 'decidedBy', data->'idDetection'->'decidedBy'->>-1, count(*) {t}
              WHERE {w} AND event = 'job.done' AND data ? 'idDetection' GROUP BY 2
            UNION ALL SELECT 'confidence', width_bucket(least((data->'idDetection'->>'confidence')::float8, 0.999999),
                                                        0, 1, 10)::text, count(*) {t}
              WHERE {w} AND event = 'job.done' AND data->'idDetection' ? 'confidence' GROUP BY 2
            UNION ALL SELECT 'fallback', f, count(*) {t}, jsonb_array_elements_text(
                CASE WHEN jsonb_typeof(data->'fallbacks') = 'array' THEN data->'fallbacks' ELSE '[]' END) AS f
              WHERE {w} AND event = 'job.done' GROUP BY 2
            UNION ALL SELECT 'method', m.key, sum(m.value::int) {t}, jsonb_each_text(
                CASE WHEN jsonb_typeof(data->'pageMethods') = 'object' THEN data->'pageMethods' ELSE '{{}}' END) AS m
              WHERE {w} AND event IN ('job.done', 'extract.done') GROUP BY 2
            UNION ALL SELECT 'language', l, count(*) {t}, jsonb_array_elements_text(
                CASE WHEN jsonb_typeof(data->'languages') = 'array' THEN data->'languages' ELSE '[]' END) AS l
              WHERE {w} AND event IN ('job.done', 'extract.done') GROUP BY 2
            UNION ALL SELECT 'error', data->>'error', count(*) {t} WHERE {w} AND event = 'job.failed' GROUP BY 2
            UNION ALL SELECT 'httpError', data->>'code', count(*) {t} WHERE {w} AND event = 'request.error' GROUP BY 2
            UNION ALL SELECT 'httpStatus', data->>'status', count(*) {t}
              WHERE {w} AND event = 'request.error' GROUP BY 2"""

    def series(self, since, bucket_s, client_id=None) -> list[dict]:
        w, p = self._where(since, None, client_id)
        rows = self._all(
            f"SELECT (floor(extract(epoch from at) / %s) * %s)::float8 AS t, event, count(*) AS n, "
            f"coalesce(sum((data->>'pages')::int), 0) AS pages, avg((data->>'ms')::float8) AS ms "
            f"FROM ocr_audit_events WHERE {w} AND event = ANY(%s) GROUP BY 1, 2 ORDER BY 1",
            (bucket_s, bucket_s, *p, list(SERIES_EVENTS)))
        return [{"t": r["t"], "event": r["event"], "n": r["n"], "pages": int(r["pages"]),
                 "ms": round(r["ms"], 1) if r["ms"] is not None else None} for r in rows]

    def client_usage(self, since) -> dict:
        out: dict = {}
        rows = self._all("SELECT coalesce(client_id, '') AS c, event, count(*) AS n, "
                         "extract(epoch from max(at))::float8 AS last, avg((data->>'ms')::float8) AS ms "
                         "FROM ocr_audit_events WHERE at >= to_timestamp(%s) GROUP BY 1, 2", (since,))
        for r in rows:
            u = out.setdefault(r["c"], {"events": {}, "lastSeen": None, "avgMs": None, "httpStatus": {},
                                        "errors": {}})
            u["events"][r["event"]] = r["n"]
            u["lastSeen"] = max(u["lastSeen"] or 0, r["last"])
            if r["event"] == "job.done" and r["ms"] is not None:
                u["avgMs"] = round(r["ms"], 1)
        rows = self._all("SELECT coalesce(client_id, '') AS c, event, data->>'status' AS st, "
                         "coalesce(data->>'code', data->>'error', '-') AS code, count(*) AS n FROM ocr_audit_events "
                         "WHERE at >= to_timestamp(%s) AND event IN ('request.error', 'job.failed') "
                         "GROUP BY 1, 2, 3, 4", (since,))
        for r in rows:
            u = out[r["c"]]
            if r["event"] == "request.error":
                u["httpStatus"][r["st"]] = u["httpStatus"].get(r["st"], 0) + r["n"]
            u["errors"][r["code"]] = u["errors"].get(r["code"], 0) + r["n"]
        return out

    def live_by_owner(self) -> dict:
        rows = self._all("SELECT owner, count(*) FILTER (WHERE status = 'queued') AS queued, "
                         "count(*) FILTER (WHERE status = 'processing') AS processing, "
                         "count(*) FILTER (WHERE webhook_next_at IS NOT NULL) AS pending FROM ocr_jobs GROUP BY owner")
        return {r["owner"]: {"queued": r["queued"], "processing": r["processing"], "webhooksPending": r["pending"]}
                for r in rows}

    def db_info(self) -> dict | None:
        try:
            row = self._all("SELECT pg_database_size(current_database()) AS size, "
                            "(SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()) AS conns, "
                            "current_setting('server_version') AS version, "
                            "(SELECT setting::int FROM pg_settings WHERE name = 'max_connections') AS max_conns")[0]
        except Exception:  # noqa: BLE001 - e.g. no permission on pg_stat_activity
            return None
        stats = self.pool.get_stats()
        pool = {"size": stats.get("pool_size"), "available": stats.get("pool_available"),
                "waiting": stats.get("requests_waiting", 0)}
        return {"sizeBytes": row["size"], "connections": row["conns"], "maxConnections": row["max_conns"],
                "version": row["version"], "pool": pool}

    # --- samples ---------------------------------------------------------------------------------------------
    def write_sample(self, instance, data) -> None:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as conn:
            conn.execute("INSERT INTO ocr_instance_samples (instance, data) VALUES (%s, %s)", (instance, Jsonb(data)))

    def latest_samples(self) -> list[dict]:
        rows = self._all("SELECT DISTINCT ON (instance) instance, extract(epoch from at)::float8 AS at, data "
                         "FROM ocr_instance_samples ORDER BY instance, at DESC")
        return [dict(r) for r in rows]

    def samples(self, instance, since, max_points=600) -> list[dict]:
        rows = self._all("SELECT extract(epoch from at)::float8 AS at, data FROM ocr_instance_samples "
                         "WHERE instance = %s AND at >= to_timestamp(%s) ORDER BY at", (instance, since))
        step = max(1, math.ceil(len(rows) / max_points))
        return [dict(r) for r in rows[::step]]

    def purge_samples_before(self, cutoff) -> int:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM ocr_instance_samples WHERE at < to_timestamp(%s)", (cutoff,)).rowcount

    # --- alerts ----------------------------------------------------------------------------------------------
    _ALERT = ("SELECT id, rule, scope, severity, summary, details, extract(epoch from started_at)::float8 AS started, "
              "extract(epoch from last_seen_at)::float8 AS seen, extract(epoch from resolved_at)::float8 AS resolved, "
              "acked_by, extract(epoch from acked_at)::float8 AS acked FROM ocr_alerts")

    @staticmethod
    def _alert(r: dict) -> dict:
        return {"id": r["id"], "rule": r["rule"], "scope": r["scope"], "severity": r["severity"],
                "summary": r["summary"], "details": r["details"], "startedAt": r["started"], "lastSeenAt": r["seen"],
                "resolvedAt": r["resolved"], "ackedBy": r["acked_by"], "ackedAt": r["acked"]}

    def alert_fire(self, rule, scope, severity, summary, details) -> tuple[int, bool]:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as conn:
            row = conn.execute(
                "INSERT INTO ocr_alerts (rule, scope, severity, summary, details) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (rule, scope) WHERE resolved_at IS NULL DO UPDATE SET last_seen_at = now(), "
                "severity = EXCLUDED.severity, summary = EXCLUDED.summary, details = EXCLUDED.details "
                "RETURNING id, (xmax = 0) AS inserted", (rule, scope, severity, summary, Jsonb(details))).fetchone()
        return row["id"], bool(row["inserted"])

    def alert_resolve(self, rule, scope) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("UPDATE ocr_alerts SET resolved_at = now() WHERE rule = %s AND scope = %s "
                                "AND resolved_at IS NULL", (rule, scope)).rowcount > 0

    def alerts(self, *, state=None, limit=200) -> list[dict]:
        where = {"open": " WHERE resolved_at IS NULL", "resolved": " WHERE resolved_at IS NOT NULL"}.get(state, "")
        return [self._alert(r) for r in self._all(self._ALERT + where + " ORDER BY id DESC LIMIT %s", (limit,))]

    def alert_ack(self, alert_id, actor) -> bool:
        with self.pool.connection() as conn:
            return conn.execute("UPDATE ocr_alerts SET acked_by = %s, acked_at = now() WHERE id = %s "
                                "AND resolved_at IS NULL", (actor, alert_id)).rowcount > 0

    # --- logs ------------------------------------------------------------------------------------------------
    def write_logs(self, records) -> None:
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ocr_log_records (at, instance, level, logger, msg, request_id, job_id, client_id, "
                "exc_type) VALUES (to_timestamp(%s), %s, %s, %s, %s, %s, %s, %s, %s)",
                [(r["at"], r["instance"], r["level"], r["logger"], r["msg"][:4000], r.get("request_id"),
                  r.get("job_id"), r.get("client_id"), r.get("exc_type")) for r in records])

    def logs(self, *, level=None, logger=None, text=None, request_id=None, job_id=None, client_id=None,
             instance=None, since=None, before=None, limit=200) -> list[dict]:
        where, params = [], []
        if level and level.upper() in LEVELS:
            where.append("level = ANY(%s)")
            params.append([name for name, n in LEVELS.items() if n >= LEVELS[level.upper()]])
        if logger:
            where.append("logger LIKE %s")
            params.append(logger.replace("%", "") + "%")
        if text:
            where.append("msg ILIKE %s")
            params.append("%" + text.replace("%", "") + "%")
        for cond, value in (("request_id = %s", request_id), ("job_id = %s", job_id), ("client_id = %s", client_id),
                            ("instance = %s", instance), ("at >= to_timestamp(%s)", since), ("id < %s", before)):
            if value is not None:
                where.append(cond)
                params.append(value)
        rows = self._all("SELECT id, extract(epoch from at)::float8 AS at, instance, level, logger, msg, request_id, "
                         "job_id, client_id, exc_type FROM ocr_log_records"
                         + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT %s",
                         (*params, limit))
        return [{k: v for k, v in dict(r).items() if v is not None} for r in rows]

    def purge_logs_before(self, cutoff) -> int:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM ocr_log_records WHERE at < to_timestamp(%s)", (cutoff,)).rowcount

    # --- config ----------------------------------------------------------------------------------------------
    _CFG = "SELECT id, extract(epoch from at)::float8 AS at, kind, actor, comment"

    def config_save(self, kind, content, actor, comment="") -> int:
        from psycopg.types.json import Jsonb

        with self.pool.connection() as conn:
            return conn.execute("INSERT INTO ocr_config_versions (kind, content, actor, comment) "
                                "VALUES (%s, %s, %s, %s) RETURNING id", (kind, Jsonb(content), actor, comment)
                                ).fetchone()["id"]

    def config_latest(self, kind) -> dict | None:
        rows = self._all(self._CFG + ", content FROM ocr_config_versions WHERE kind = %s ORDER BY id DESC LIMIT 1",
                         (kind,))
        return dict(rows[0]) if rows else None

    def config_heads(self) -> dict[str, int]:
        return {r["kind"]: r["id"] for r in self._all("SELECT kind, max(id) AS id FROM ocr_config_versions "
                                                     "GROUP BY kind")}

    def config_history(self, kind=None, limit=100) -> list[dict]:
        if kind is None:
            rows = self._all(self._CFG + " FROM ocr_config_versions ORDER BY id DESC LIMIT %s", (limit,))
        elif kind.endswith(":"):
            rows = self._all(self._CFG + " FROM ocr_config_versions WHERE kind LIKE %s ORDER BY id DESC LIMIT %s",
                             (kind.replace("%", "") + "%", limit))
        else:
            rows = self._all(self._CFG + " FROM ocr_config_versions WHERE kind = %s ORDER BY id DESC LIMIT %s",
                             (kind, limit))
        return [dict(r) for r in rows]

    def config_version(self, version_id) -> dict | None:
        rows = self._all(self._CFG + ", content FROM ocr_config_versions WHERE id = %s", (version_id,))
        return dict(rows[0]) if rows else None

    def secret_set(self, client_id, secret) -> None:
        sealed = self.sealer.seal(secret.encode(), f"client-secret:{client_id}")
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO ocr_client_secrets (client_id, sealed) VALUES (%s, %s) ON CONFLICT (client_id) "
                         "DO UPDATE SET sealed = EXCLUDED.sealed, updated_at = now()", (client_id, sealed))

    def secrets(self) -> dict[str, str]:
        rows = self._all("SELECT client_id, sealed FROM ocr_client_secrets")
        return {r["client_id"]: self.sealer.open(r["sealed"], f"client-secret:{r['client_id']}").decode()
                for r in rows}


def make_admin_store(jobs: JobStore) -> AdminStore:
    if isinstance(jobs, InMemoryJobStore):
        return MemoryAdminStore(jobs)
    from app.api.store_pg import PostgresJobStore

    if isinstance(jobs, PostgresJobStore):
        return PgAdminStore(jobs)
    raise TypeError(f"the admin dashboard doesn't support {type(jobs).__name__}")
