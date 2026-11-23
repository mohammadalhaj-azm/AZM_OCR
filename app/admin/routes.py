"""Admin dashboard HTTP API (/admin/api/*) and pages (/admin). Not part of the client API contract
(docs/openapi.json): every route is left out of the OpenAPI schema.

Read endpoints need a logged-in user (viewer or admin); endpoints that change something need the admin role and
pass the CSRF checks (app/admin/auth.py). Responses hold metadata only: never document text, file names, MRZ fields,
caller metadata, API keys, key hashes or secrets. A newly generated API key or webhook secret is returned once, in
the response of the request that created it.
"""

import asyncio
import copy
import csv
import io
import json
import logging
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)

from app import __version__
from app.admin import ai
from app.admin.alerts import RULES
from app.admin.auth import COOKIE, AdminUser, admin_user, check_csrf, client_address, current_user
from app.admin.managed_config import REGISTRY, SETTINGS, ConfigConflict
from app.admin.managed_config import RULES as RULES_KIND
from app.api import audit
from app.api.clients import key_hash
from app.api.errors import ApiError
from app.api.store import AuditEvent
from app.config import ConfigError
from app.detection.llm import DOC_TYPES
from app.detection.rules import MANAGED_PREFIX, load_rules_text, resolve_rules
from app.profiles import Profile

log = logging.getLogger("app.admin")
STATIC = Path(__file__).with_name("static")
router = APIRouter(include_in_schema=False)
RANGES = {"15m": (900, 30), "1h": (3600, 60), "6h": (21600, 300), "24h": (86400, 900), "7d": (604800, 7200),
          "30d": (2592000, 43200)}
STREAM_MAX_S = 30  # an SSE stream ends after this (the browser reconnects), so a shutdown never waits long
JOB_FIELDS = ("status", "attempts", "created_at", "started_at", "finished_at", "webhook", "error", "result_ttl_s")


def rt(request: Request):
    return request.app.state.admin


def window(range_: str) -> tuple[float, int, str]:
    span, bucket = RANGES.get(range_, RANGES["24h"])
    return time.time() - span, bucket, range_ if range_ in RANGES else "24h"


def record(request: Request, event: str, actor: AdminUser | None, job_id: str | None = None, **data) -> None:
    """An admin action in the audit trail (`data.actor` = the admin user)."""
    ev = AuditEvent(event, None, job_id, getattr(request.state, "request_id", None),
                    {"actor": actor.username if actor else None, **data})
    audit.record_safely(request.app.state.store, ev, request.app.state.metrics)


def config_error(e: Exception) -> ApiError:
    if isinstance(e, ConfigConflict):
        return ApiError(409, "conflict", "someone saved a newer version meanwhile: reload and try again")
    return ApiError(422, "invalid_config", str(e)[:2000])


# =====================================================================================================================
# Pages and session
# =====================================================================================================================
@router.get("/admin")
@router.get("/admin/")
def index(request: Request):
    if rt(request).auth.session_user(request.cookies.get(COOKIE)) is None:
        return RedirectResponse("/admin/login", status_code=303)
    return FileResponse(STATIC / "index.html", media_type="text/html")


@router.get("/admin/login")
def login_page():
    return FileResponse(STATIC / "login.html", media_type="text/html")


@router.post("/admin/api/login")
async def login(request: Request, body: dict = Body(...)):
    check_csrf(request)
    username, password = str(body.get("username") or "")[:64], str(body.get("password") or "")[:1024]
    admin = rt(request)
    user, wait = await run_in_threadpool(admin.auth.login, username, password, client_address(request))
    if user is None:
        known = username in admin.auth.users()
        await run_in_threadpool(record, request, audit.ADMIN_LOGIN_FAILED, None,
                                user=username if known else None,  # never an unknown name (could be a password)
                                reason="throttled" if wait else "invalid")
        log.warning("admin login failed known_user=%s addr=%s", known, client_address(request))
        if wait > 0:
            raise ApiError(429, "too_many_attempts", "too many failed logins, wait and retry",
                           {"Retry-After": str(int(wait) + 1)})
        raise ApiError(401, "unauthorized", "wrong user name or password")
    response = JSONResponse({"user": user.public()})
    response.set_cookie(value=admin.auth.issue(user), **admin.auth.cookie_kwargs())
    await run_in_threadpool(record, request, audit.ADMIN_LOGIN, user)
    log.info("admin login user=%s", user.username)
    return response


@router.post("/admin/api/logout")
async def logout(request: Request, user: AdminUser = Depends(current_user)):
    await run_in_threadpool(record, request, audit.ADMIN_LOGOUT, user)
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/admin")
    return response


@router.get("/admin/api/me")
def me(request: Request, user: AdminUser = Depends(current_user)):
    admin, s = rt(request), request.app.state.settings
    return {"user": user.public(), "version": __version__, "instance": admin.instance, "environment": s.environment,
            "persistent": admin.store.persistent, "jobStore": s.job_store, "ranges": list(RANGES),
            "sampleIntervalS": s.sample_interval_s}


# =====================================================================================================================
# Live (SSE) and overview
# =====================================================================================================================
def live_snapshot(request: Request) -> dict:
    admin, state = rt(request), request.app.state
    jobs = state.store
    open_alerts = admin.store.alerts(state="open", limit=500)
    worker = getattr(state, "worker", None)
    return {"at": time.time(), "instance": admin.instance, "sample": admin.sampler.latest() if admin.sampler else None,
            "jobs": jobs.counts(), "ages": jobs.ages(), "webhooksPending": jobs.count_pending_webhooks(),
            "active": worker.active() if worker else [],
            "alerts": {"open": len(open_alerts), "critical": sum(a["severity"] == "critical" for a in open_alerts),
                       "unacked": sum(a["ackedAt"] is None for a in open_alerts),
                       "latestId": max((a["id"] for a in open_alerts), default=0),
                       "latest": [_alert_view(a) for a in open_alerts[:5]]},
            "latestEventId": admin.store.latest_event_id()}


@router.get("/admin/api/live")
async def live(request: Request, user: AdminUser = Depends(current_user)):
    return await run_in_threadpool(live_snapshot, request)


@router.get("/admin/api/stream")
async def stream(request: Request, once: bool = False, user: AdminUser = Depends(current_user)):
    admin = rt(request)
    interval = max(1.0, request.app.state.settings.sample_interval_s)
    cookie = request.cookies.get(COOKIE)

    async def events():
        yield "retry: 1500\n\n"
        end = time.monotonic() + STREAM_MAX_S
        while time.monotonic() < end and not admin.stopping.is_set():
            if await request.is_disconnected():
                return
            if admin.auth.session_user(cookie) is None:
                yield "event: logout\ndata: {}\n\n"
                return
            try:
                snap = await run_in_threadpool(live_snapshot, request)
                yield f"event: tick\ndata: {json.dumps(snap, default=str)}\n\n"
            except Exception as e:  # noqa: BLE001 - e.g. database down: tell the page, keep the stream
                yield f"event: error\ndata: {json.dumps({'error': type(e).__name__})}\n\n"
            if once:
                return
            await asyncio.sleep(interval)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _instances(request: Request) -> list[dict]:
    """Every replica: its latest stored sample, and this replica's fresh one from RAM."""
    admin = rt(request)
    now, rows = time.time(), {}
    for r in admin.store.latest_samples():
        rows[r["instance"]] = {**r["data"], "at": r["at"]}
    if admin.sampler and (mine := admin.sampler.latest()):
        rows[admin.instance] = mine
    out = []
    for inst, d in rows.items():
        age = now - d.get("at", 0)
        state = "stopped" if d.get("stopping") else ("ok" if age < 60 else ("missing" if age < 3600 else "gone"))
        if state == "gone":
            continue
        out.append({**d, "instance": inst, "ageS": round(age, 1), "state": state, "self": inst == admin.instance})
    return sorted(out, key=lambda r: (not r["self"], r.get("host", ""), r["instance"]))


@router.get("/admin/api/overview")
def overview(request: Request, range: str = "24h", user: AdminUser = Depends(current_user)):  # noqa: A002
    since, bucket, rng = window(range)
    admin, jobs = rt(request), request.app.state.store
    st = admin.store.stats(since)
    ev = st["events"]
    done, failed, extracts = ev.get("job.done", 0), ev.get("job.failed", 0), ev.get("extract.done", 0)
    finished = done + failed + extracts
    return {
        "range": rng, "since": since, "bucketS": bucket,
        "kpis": {"documents": ev.get("job.received", 0) + extracts, "jobsReceived": ev.get("job.received", 0),
                 "extracts": extracts, "done": done + extracts, "failed": failed,
                 "successRate": round((done + extracts) / finished, 4) if finished else None,
                 "requestErrors": ev.get("request.error", 0), "pages": st["pages"], "ids": st["ids"],
                 "jobMs": st["durations"]["job"], "extractMs": st["durations"]["extract"]},
        "live": {"jobs": jobs.counts(), "ages": jobs.ages(), "webhooksPending": jobs.count_pending_webhooks()},
        "series": admin.store.series(since, bucket),
        "errors": {"jobs": st["dist"]["error"], "http": st["dist"]["httpError"]},
        "docTypes": st["dist"]["docType"],
        "instances": [_instance_brief(i) for i in _instances(request)],
        "alerts": [_alert_view(a) for a in admin.store.alerts(state="open", limit=10)],
        "activity": [_event_view(e) for e in admin.store.audit_page(limit=20)],
    }


def _instance_brief(i: dict) -> dict:
    return {k: i.get(k) for k in ("instance", "host", "pid", "version", "state", "ageS", "self", "uptimeS",
                                  "health", "work")} | {
        "cpu": (i.get("cpu") or {}).get("percent"), "memory": (i.get("memory") or {}).get("percent"),
        "gpus": [{"index": g.get("index"), "util": g.get("util"), "memPercent": g.get("memPercent")}
                 for g in i.get("gpus") or []]}


def _event_view(e: dict) -> dict:
    return {"id": e["id"], "at": e["at"], "event": e["event"], "clientId": e["clientId"], "jobId": e["jobId"],
            "requestId": e["requestId"], "data": e["data"]}


# =====================================================================================================================
# Documents
# =====================================================================================================================
@router.get("/admin/api/documents")
def documents(request: Request, range: str = "7d", status: str | None = None, client: str | None = None,  # noqa: A002
              kind: str | None = None, docType: str | None = None, error: str | None = None,
              q: str | None = None, before: int | None = None, limit: int = Query(50, ge=1, le=200),
              user: AdminUser = Depends(current_user)):
    since, _, rng = window(range)
    q = (q or "").strip()[:128] or None
    rows = rt(request).store.documents(since=since, client_id=client or None, kind=kind or None,
                                       status=status or None, doc_type=docType or None, error=error or None, q=q,
                                       q_hash=request.app.state.ref_hash(q) if q else None, before=before,
                                       limit=limit)
    return {"range": rng, "items": rows, "next": rows[-1]["cursor"] if len(rows) == limit else None}


@router.get("/admin/api/documents/{doc_id}")
def document(request: Request, doc_id: str, user: AdminUser = Depends(current_user)):
    admin = rt(request)
    if doc_id.startswith("evt-"):
        try:
            e = admin.store.event(int(doc_id[4:]))
        except ValueError:
            e = None
        if e is None:
            raise ApiError(404, "not_found", "document not found")
        from app.admin.store import document_view

        return {"document": document_view(e), "events": [_event_view(e)], "job": None, "timeline": []}
    events = admin.store.job_events(doc_id[:64])
    job = request.app.state.store.get(doc_id[:64])
    if not events and job is None:
        raise ApiError(404, "not_found", "document not found")
    received = next((e for e in events if e["event"] == "job.received"), None)
    final = next((e for e in reversed(events) if e["event"] in ("job.done", "job.failed")), None)
    from app.admin.store import document_view

    doc = document_view(received, final, job.status if job else None) if received else None
    live = None
    if job is not None:  # metadata columns only: never filename, result, external id or caller metadata
        live = {"status": job.status, "attempts": job.attempts, "createdAt": job.created_at,
                "startedAt": job.started_at, "finishedAt": job.finished_at, "webhook": job.webhook,
                "error": job.error, "callback": bool(job.callback_url), "resultTtlS": job.result_ttl_s,
                "webhookNextAt": job.webhook_next_at, "clientId": job.owner}
    active = next((a for a in (getattr(request.app.state, "worker", None) and request.app.state.worker.active()
                               or []) if a["jobId"] == doc_id), None)
    return {"document": doc, "job": live, "active": active, "events": [_event_view(e) for e in events],
            "timeline": _timeline(events)}


def _timeline(events: list[dict]) -> list[dict]:
    """Lifecycle steps with the time since the previous step (queue wait, processing, delivery...)."""
    out, prev = [], None
    for e in events:
        out.append({"event": e["event"], "at": e["at"], "sincePrevMs": round((e["at"] - prev) * 1000) if prev else None,
                    "data": e["data"]})
        prev = e["at"]
    return out


@router.delete("/admin/api/documents/{job_id}")
def delete_document(request: Request, job_id: str, user: AdminUser = Depends(admin_user)):
    store = request.app.state.store
    job = store.get(job_id[:64])
    if job is None:
        raise ApiError(404, "not_found", "job not found (already deleted or expired)")
    store.delete(job.id, AuditEvent(audit.DELETED, job.owner, job.id, request.state.request_id,
                                    {"by": "admin", "actor": user.username}))
    record(request, audit.ADMIN_JOB_DELETED, user, job.id, client=job.owner, status=job.status)
    return Response(status_code=204)


# =====================================================================================================================
# System resources
# =====================================================================================================================
@router.get("/admin/api/system")
def system(request: Request, user: AdminUser = Depends(current_user)):
    admin = rt(request)
    return {"instances": _instances(request), "database": admin.store.db_info(), "jobStore":
            request.app.state.settings.job_store, "self": admin.instance}


@router.get("/admin/api/system/history")
def system_history(request: Request, instance: str | None = None, range: str = "1h",  # noqa: A002
                   user: AdminUser = Depends(current_user)):
    admin = rt(request)
    since, _, rng = window(range)
    inst = instance or admin.instance
    if inst == admin.instance and admin.sampler and time.time() - since <= 900:
        points = [{"at": s["at"], "data": s} for s in admin.sampler.recent(since)]  # fresh, every SAMPLE_INTERVAL_S
    else:
        points = admin.store.samples(inst, since)
    return {"instance": inst, "range": rng, "points": [_point(p["at"], p["data"]) for p in points]}


def _point(at: float, d: dict) -> dict:
    io_, mem = d.get("io") or {}, d.get("memory") or {}
    return {"at": at, "cpu": (d.get("cpu") or {}).get("percent"), "mem": mem.get("percent"),
            "rss": mem.get("processRss"), "diskRead": io_.get("diskReadBps"), "diskWrite": io_.get("diskWriteBps"),
            "netSent": io_.get("netSentBps"), "netRecv": io_.get("netRecvBps"),
            "gpu": [g.get("util") for g in d.get("gpus") or []],
            "vram": [g.get("memPercent") for g in d.get("gpus") or []],
            "active": (d.get("work") or {}).get("activeJobs")}


# =====================================================================================================================
# AI monitoring
# =====================================================================================================================
@router.get("/admin/api/ai")
def ai_monitoring(request: Request, range: str = "24h", user: AdminUser = Depends(current_user)):  # noqa: A002
    since, bucket, rng = window(range)
    state = request.app.state
    out = ai.ai_overview(state.settings, getattr(state, "services", None), rt(request).store, since)
    out["range"], out["series"], out["bucketS"] = rng, rt(request).store.series(since, bucket), bucket
    return out


# =====================================================================================================================
# Integrations
# =====================================================================================================================
def _health(c, usage: dict) -> tuple[str, list[str]]:
    if not c.enabled:
        return "disabled", ["client is disabled: its keys get 401"]
    ev, st = usage.get("events", {}), usage.get("httpStatus", {})
    total = ev.get("job.received", 0) + ev.get("extract.done", 0) + ev.get("request.error", 0)
    if not total and not ev:
        return "idle", ["no requests in this period"]
    server = sum(n for s, n in st.items() if s.startswith("5") and s != "503") + ev.get("job.failed", 0)
    rate = server / max(1, total)
    reasons = []
    if ev.get("webhook.failed"):
        reasons.append(f"{ev['webhook.failed']} webhook deliveries given up")
    if st.get("503"):
        reasons.append(f"{st['503']} requests rejected with 503 (limits)")
    if rate > 0.05:
        reasons.append(f"{round(rate * 100)}% of documents failed")
    if rate > 0.25 or (ev.get("webhook.failed") and not ev.get("webhook.delivered")):
        return "critical", reasons
    return ("warning", reasons) if reasons else ("ok", [])


def _client_view(c, registry, usage: dict, live: dict) -> dict:
    health, reasons = _health(c, usage)
    return {"id": c.id, "name": c.name, "enabled": c.enabled, "scopes": sorted(c.scopes), "profiles": list(c.profiles),
            "keys": registry.key_count(c.id), "keyFingerprints": registry.key_fingerprints(c.id),
            "webhook": {"configured": bool(c.webhook_secret), "hosts": sorted(c.webhook_hosts),
                        "retryWindowS": c.webhook_retry_window_s},
            "limits": {"maxQueuedJobs": c.max_queued_jobs, "extractConcurrent": c.extract_concurrent},
            "usage": usage, "live": live or {"queued": 0, "processing": 0, "webhooksPending": 0},
            "health": health, "healthReasons": reasons}


@router.get("/admin/api/integrations")
def integrations(request: Request, range: str = "24h", user: AdminUser = Depends(current_user)):  # noqa: A002
    since, _, rng = window(range)
    admin, registry = rt(request), request.app.state.clients
    usage, live = admin.store.client_usage(since), admin.store.live_by_owner()
    clients = [_client_view(c, registry, usage.get(c.id, {}), live.get(c.id)) for c in registry.clients]
    unknown = usage.get("", {})
    return {"range": rng, "source": registry.source, "clients": clients,
            "unauthenticated": {"requests": unknown.get("events", {}).get("request.error", 0),
                                "httpStatus": unknown.get("httpStatus", {}), "lastSeen": unknown.get("lastSeen")},
            "orphans": sorted(k for k in usage if k and registry.get(k) is None)}


@router.get("/admin/api/integrations/{client_id}")
def integration(request: Request, client_id: str, range: str = "24h", before: int | None = None,  # noqa: A002
                user: AdminUser = Depends(current_user)):
    since, bucket, rng = window(range)
    admin, registry = rt(request), request.app.state.clients
    c = registry.get(client_id)
    usage = admin.store.client_usage(since).get(client_id, {})
    if c is None and not usage:
        raise ApiError(404, "not_found", "client not found")
    view = _client_view(c, registry, usage, admin.store.live_by_owner().get(client_id)) if c else {
        "id": client_id, "removed": True, "usage": usage}
    history = admin.store.audit_page(client_id=client_id, before=before, limit=50)
    return {"range": rng, "client": view, "stats": admin.store.stats(since, client_id=client_id),
            "series": admin.store.series(since, bucket, client_id=client_id), "bucketS": bucket,
            "requests": [_event_view(e) for e in history], "next": history[-1]["id"] if len(history) == 50 else None}


# =====================================================================================================================
# Logs, audit, tracing
# =====================================================================================================================
@router.get("/admin/api/logs")
def logs(request: Request, source: str = "local", level: str | None = None, logger: str | None = None,
         text: str | None = None, requestId: str | None = None, jobId: str | None = None,
         clientId: str | None = None, instance: str | None = None, before: int | None = None,
         limit: int = Query(200, ge=1, le=1000), user: AdminUser = Depends(current_user)):
    admin = rt(request)
    filters = {"level": level, "logger": logger, "text": (text or "")[:200] or None, "request_id": requestId or None,
               "job_id": jobId or None, "client_id": clientId or None, "before": before, "limit": limit}
    if source == "all":
        rows = admin.store.logs(instance=instance or None, **filters)
    else:
        rows = admin.logs.recent(**filters) if admin.logs else []
    return {"source": "all" if source == "all" else "local", "instance": admin.instance, "items": rows,
            "next": rows[-1]["id"] if len(rows) == limit else None,
            "dropped": admin.logs.dropped if admin.logs else 0}


def _audit_rows(request: Request, event, client, job, req, actor, sha256, external, range_, before, limit):
    since, _, _ = window(range_)
    external = (external or "").strip()
    return rt(request).store.audit_page(
        event=event or None, client_id=client or None, job_id=job or None, request_id=req or None,
        actor=actor or None, sha256=sha256 or None,
        external_hash=request.app.state.ref_hash(external) if external else None, since=since, before=before,
        limit=limit)


@router.get("/admin/api/audit")
def audit_log(request: Request, event: str | None = None, client: str | None = None, job: str | None = None,
              requestId: str | None = None, actor: str | None = None, sha256: str | None = None,
              externalId: str | None = None, range: str = "7d", before: int | None = None,  # noqa: A002
              limit: int = Query(100, ge=1, le=5000), format: str = "json",  # noqa: A002
              user: AdminUser = Depends(current_user)):
    rows = _audit_rows(request, event, client, job, requestId, actor, sha256, externalId, range, before, limit)
    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id", "at", "event", "client_id", "job_id", "request_id", "data"])
        for e in rows:
            w.writerow([e["id"], datetime.fromtimestamp(e["at"], UTC).isoformat(), e["event"], e["clientId"] or "",
                        e["jobId"] or "", e["requestId"] or "", json.dumps(e["data"], ensure_ascii=False)])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="audit.csv"'})
    return {"items": [_event_view(e) for e in rows], "next": rows[-1]["id"] if len(rows) == limit else None,
            "events": sorted(audit.EVENTS)}


@router.get("/admin/api/trace")
def trace(request: Request, requestId: str | None = None, jobId: str | None = None,
          user: AdminUser = Depends(current_user)):
    """Everything about one request id or job id: audit events and log lines (this replica's buffer and the stored
    warnings of all replicas)."""
    if not requestId and not jobId:
        raise ApiError(422, "invalid_request", "give requestId or jobId")
    admin = rt(request)
    events = admin.store.jobs.audit_events(request_id=requestId or None, job_id=jobId or None, limit=1000)
    job_ids = {e["jobId"] for e in events if e["jobId"]} | ({jobId} if jobId else set())
    request_ids = {e["requestId"] for e in events if e["requestId"]} | ({requestId} if requestId else set())
    if requestId and not jobId:  # the request's job: its later events (worker, webhook) too
        for jid in list(job_ids):
            events += [e for e in admin.store.job_events(jid) if e["id"] not in {x["id"] for x in events}]
    events.sort(key=lambda e: e["id"])
    lines: dict[tuple, dict] = {}
    for rid in request_ids:
        for r in (admin.logs.recent(request_id=rid, limit=500) if admin.logs else []) + admin.store.logs(
                request_id=rid, limit=500):
            lines[(r.get("instance"), r["at"], r["msg"])] = r
    for jid in job_ids:
        for r in (admin.logs.recent(job_id=jid, limit=500) if admin.logs else []) + admin.store.logs(
                job_id=jid, limit=500):
            lines[(r.get("instance"), r["at"], r["msg"])] = r
    return {"requestIds": sorted(request_ids), "jobIds": sorted(job_ids), "events": [_event_view(e) for e in events],
            "logs": sorted(lines.values(), key=lambda r: r["at"])}


# =====================================================================================================================
# Configuration
# =====================================================================================================================
def _cfg(request: Request):
    return rt(request).config


@router.get("/admin/api/config/settings")
def config_settings(request: Request, user: AdminUser = Depends(current_user)):
    cfg = _cfg(request)
    head = cfg.store.config_latest(SETTINGS)
    return {"items": cfg.settings_view(), "version": head["id"] if head else None,
            "pendingRestart": cfg.pending_restart, "error": cfg.error, "persistent": cfg.store.persistent}


@router.put("/admin/api/config/settings")
def save_settings(request: Request, body: dict = Body(...), user: AdminUser = Depends(admin_user)):
    patch = body.get("patch")
    if not isinstance(patch, dict) or not patch:
        raise ApiError(422, "invalid_request", "patch must be a non-empty object")
    try:
        vid = _cfg(request).save_settings(patch, user.username, str(body.get("comment") or "")[:500],
                                          body.get("baseVersion"))
    except (ConfigError, ConfigConflict) as e:
        raise config_error(e) from None
    record(request, audit.ADMIN_CONFIG_SAVED, user, kind=SETTINGS, version=vid, keys=sorted(patch))
    return {"version": vid, "pendingRestart": _cfg(request).pending_restart}


def _sanitize_registry(content: dict) -> dict:
    """The registry for the browser: key hashes replaced by fingerprints."""
    out = copy.deepcopy(content)
    for c in (out.get("clients") or {}).values():
        c["keys"] = sorted(h[:8] for h in c.pop("key_sha256", []) or [])
    return out


@router.get("/admin/api/config/registry")
def config_registry(request: Request, user: AdminUser = Depends(current_user)):
    cfg = _cfg(request)
    editable, reason = cfg.registry_editable()
    try:
        content, version, source = cfg.registry_content()
    except ConfigError as e:
        raise config_error(e) from None
    secrets_ = set(cfg.store.secrets())
    view = _sanitize_registry(content)
    for cid, c in (view.get("clients") or {}).items():
        if c.get("webhook"):
            c["webhook"]["secretStored"] = cid in secrets_
    schema = [{"key": k, "type": str(f.annotation).replace("typing.", ""), "description": f.description or ""}
              for k, f in Profile.model_fields.items() if k != "name"]
    return {"editable": editable, "reason": reason, "source": source, "version": version, "registry": view,
            "profileSchema": schema, "scopes": ["documents", "extract"],
            "ruleSets": [MANAGED_PREFIX + n for n in cfg.rule_sets()], "persistent": cfg.store.persistent}


def _mutate_registry(request: Request, user: AdminUser, body: dict, change, comment: str) -> int:
    cfg = _cfg(request)
    try:
        content, version, _ = cfg.registry_content()
        base = body.get("baseVersion", version)
        change(content)
        vid = cfg.save_registry(content, user.username, str(body.get("comment") or comment)[:500], base)
    except (ConfigError, ConfigConflict) as e:
        raise config_error(e) from None
    record(request, audit.ADMIN_CONFIG_SAVED, user, kind=REGISTRY, version=vid, change=comment[:200])
    return vid


CLIENT_FIELDS = {"name", "scopes", "profiles", "webhook", "limits", "enabled"}


@router.put("/admin/api/config/clients/{client_id}")
def save_client(request: Request, client_id: str, body: dict = Body(...), user: AdminUser = Depends(admin_user)):
    spec = body.get("client")
    if not isinstance(spec, dict) or set(spec) - CLIENT_FIELDS:
        raise ApiError(422, "invalid_request", f"client must be an object with {', '.join(sorted(CLIENT_FIELDS))}")
    new_key: list[str] = []

    def change(content: dict) -> None:
        clients = content.setdefault("clients", {})
        old = clients.get(client_id)
        item = {k: v for k, v in spec.items() if v is not None}
        if old is None:
            key = secrets.token_urlsafe(32)
            new_key.append(key)
            item["key_sha256"] = [key_hash(key)]
        else:
            item["key_sha256"] = old["key_sha256"]
        hook = item.get("webhook")
        if hook:
            hook = {k: v for k, v in hook.items() if k in ("allowed_hosts", "retry_window_s", "secret_env") and v}
            if not hook.get("secret_env"):
                if not ((old or {}).get("webhook") or {}).get("managed_secret"):
                    raise ConfigError("webhook: give secret_env, or save the client without a webhook and generate "
                                      "a webhook secret")
                hook["managed_secret"] = True
            item["webhook"] = hook
        clients[client_id] = item

    vid = _mutate_registry(request, user, body, change, f"client {client_id} saved")
    if new_key:
        record(request, audit.ADMIN_KEY_CREATED, user, client=client_id, fingerprint=key_hash(new_key[0])[:8])
    return {"version": vid, "key": new_key[0] if new_key else None}


@router.delete("/admin/api/config/clients/{client_id}")
def delete_client(request: Request, client_id: str, user: AdminUser = Depends(admin_user)):
    def change(content: dict) -> None:
        if client_id not in (content.get("clients") or {}):
            raise ConfigError(f"client {client_id} doesn't exist")
        del content["clients"][client_id]

    return {"version": _mutate_registry(request, user, {}, change, f"client {client_id} deleted")}


@router.post("/admin/api/config/clients/{client_id}/keys")
def create_key(request: Request, client_id: str, user: AdminUser = Depends(admin_user)):
    key = secrets.token_urlsafe(32)

    def change(content: dict) -> None:
        c = (content.get("clients") or {}).get(client_id)
        if c is None:
            raise ConfigError(f"client {client_id} doesn't exist")
        c["key_sha256"] = [*c.get("key_sha256", []), key_hash(key)]

    vid = _mutate_registry(request, user, {}, change, f"client {client_id}: key added")
    record(request, audit.ADMIN_KEY_CREATED, user, client=client_id, fingerprint=key_hash(key)[:8])
    return {"version": vid, "key": key, "fingerprint": key_hash(key)[:8]}


@router.delete("/admin/api/config/clients/{client_id}/keys/{fingerprint}")
def revoke_key(request: Request, client_id: str, fingerprint: str, user: AdminUser = Depends(admin_user)):
    def change(content: dict) -> None:
        c = (content.get("clients") or {}).get(client_id)
        if c is None:
            raise ConfigError(f"client {client_id} doesn't exist")
        fp = fingerprint.lower()
        hits = [h for h in c.get("key_sha256", []) if h.startswith(fp)] if len(fp) >= 8 else []
        if len(hits) != 1:
            raise ConfigError("no single key matches that fingerprint")
        if len(c["key_sha256"]) == 1:
            raise ConfigError("this is the client's last key: add a new key first, or disable the client")
        c["key_sha256"] = [h for h in c["key_sha256"] if h != hits[0]]

    vid = _mutate_registry(request, user, {}, change, f"client {client_id}: key {fingerprint[:8]} revoked")
    record(request, audit.ADMIN_KEY_REVOKED, user, client=client_id, fingerprint=fingerprint[:8])
    return {"version": vid}


@router.post("/admin/api/config/clients/{client_id}/webhook-secret")
def create_webhook_secret(request: Request, client_id: str, body: dict = Body(default={}),
                          user: AdminUser = Depends(admin_user)):
    cfg = _cfg(request)
    content, _, _ = cfg.registry_content()
    c = (content.get("clients") or {}).get(client_id)
    if c is None:
        raise ApiError(404, "not_found", "client not found")
    hosts = body.get("allowed_hosts") or ((c.get("webhook") or {}).get("allowed_hosts"))
    if not hosts:
        raise ApiError(422, "invalid_config", "give allowed_hosts (the callback host names) for the webhook")
    secret = secrets.token_urlsafe(32)
    cfg.store.secret_set(client_id, secret)

    def change(content: dict) -> None:
        cl = content["clients"][client_id]
        old = cl.get("webhook") or {}
        cl["webhook"] = {"managed_secret": True, "allowed_hosts": list(hosts),
                         **({"retry_window_s": old["retry_window_s"]} if old.get("retry_window_s") else {})}

    vid = _mutate_registry(request, user, {}, change, f"client {client_id}: webhook secret generated")
    record(request, audit.ADMIN_SECRET_CREATED, user, client=client_id)
    return {"version": vid, "secret": secret}


@router.put("/admin/api/config/profiles/{name}")
def save_profile(request: Request, name: str, body: dict = Body(...), user: AdminUser = Depends(admin_user)):
    spec = body.get("profile")
    if not isinstance(spec, dict):
        raise ApiError(422, "invalid_request", "profile must be an object")

    def change(content: dict) -> None:
        content.setdefault("profiles", {})[name] = {k: v for k, v in spec.items() if v is not None and k != "name"}

    return {"version": _mutate_registry(request, user, body, change, f"profile {name} saved")}


@router.delete("/admin/api/config/profiles/{name}")
def delete_profile(request: Request, name: str, user: AdminUser = Depends(admin_user)):
    def change(content: dict) -> None:
        if name not in (content.get("profiles") or {}):
            raise ConfigError(f"profile {name} doesn't exist")
        users = [cid for cid, c in (content.get("clients") or {}).items() if name in (c.get("profiles") or [])]
        if users:
            raise ConfigError(f"profile {name} is still used by: {', '.join(users)}")
        del content["profiles"][name]

    return {"version": _mutate_registry(request, user, {}, change, f"profile {name} deleted")}


def _rules_summary(rules) -> dict:
    by_type: dict[str, dict] = {t: {"keywords": 0, "patterns": 0} for t in DOC_TYPES}
    for kw in rules.keywords:
        if kw.doc_type in by_type:
            by_type[kw.doc_type]["keywords"] += 1
    for p in rules.patterns:
        if p.doc_type in by_type:
            by_type[p.doc_type]["patterns"] += 1
    return {"keywords": len(rules.keywords), "negativeKeywords": len(rules.negatives), "patterns": len(rules.patterns),
            "countries": sorted({p.country for p in rules.patterns}), "byDocType": by_type}


@router.get("/admin/api/config/rules")
def config_rules(request: Request, user: AdminUser = Depends(current_user)):
    cfg, s = _cfg(request), request.app.state.settings
    file_path = Path(cfg.base.id_patterns_path)
    try:
        file_text = file_path.read_text(encoding="utf-8") if not str(file_path).startswith(MANAGED_PREFIX) else ""
    except OSError:
        file_text = ""
    try:
        active, version = resolve_rules(s.id_patterns_path)
        summary = _rules_summary(active)
    except Exception as e:  # noqa: BLE001
        summary, version = {"error": str(e)[:300]}, None
    sets = cfg.rule_sets()
    for name, v in sets.items():
        v["usedBy"] = cfg.rule_set_users(name)
    return {"active": str(s.id_patterns_path), "activeVersion": version, "summary": summary,
            "file": {"path": str(file_path), "yaml": file_text}, "sets": sets, "docTypes": DOC_TYPES,
            "persistent": cfg.store.persistent}


@router.post("/admin/api/config/rules/validate")
def validate_rules(request: Request, body: dict = Body(...), user: AdminUser = Depends(current_user)):
    try:
        return {"ok": True, "summary": _rules_summary(load_rules_text(str(body.get("yaml") or "")))}
    except ValueError as e:
        return {"ok": False, "error": str(e)[:1000]}


@router.put("/admin/api/config/rules/{name}")
def save_rules(request: Request, name: str, body: dict = Body(...), user: AdminUser = Depends(admin_user)):
    try:
        vid = _cfg(request).save_rules(name, str(body.get("yaml") or ""), user.username,
                                       str(body.get("comment") or "")[:500], body.get("baseVersion"))
    except (ConfigError, ConfigConflict, ValueError) as e:
        raise config_error(e) from None
    record(request, audit.ADMIN_CONFIG_SAVED, user, kind=RULES_KIND + name, version=vid)
    return {"version": vid, "ref": MANAGED_PREFIX + name}


@router.delete("/admin/api/config/rules/{name}")
def delete_rules(request: Request, name: str, user: AdminUser = Depends(admin_user)):
    try:
        vid = _cfg(request).delete_rules(name, user.username)
    except ConfigError as e:
        raise config_error(e) from None
    record(request, audit.ADMIN_CONFIG_SAVED, user, kind=RULES_KIND + name, version=vid, deleted=True)
    return {"version": vid}


@router.get("/admin/api/config/history")
def config_history(request: Request, kind: str | None = None, user: AdminUser = Depends(current_user)):
    return {"items": _cfg(request).store.config_history(kind or None), "applied": _cfg(request).applied,
            "error": _cfg(request).error}


@router.get("/admin/api/config/versions/{version_id}")
def config_version(request: Request, version_id: int, user: AdminUser = Depends(current_user)):
    v = _cfg(request).store.config_version(version_id)
    if v is None:
        raise ApiError(404, "not_found", "version not found")
    if v["kind"] == REGISTRY:
        v = {**v, "content": _sanitize_registry(v["content"])}
    return v


@router.post("/admin/api/config/versions/{version_id}/restore")
def restore_version(request: Request, version_id: int, user: AdminUser = Depends(admin_user)):
    try:
        vid = _cfg(request).restore(version_id, user.username)
    except KeyError:
        raise ApiError(404, "not_found", "version not found") from None
    except (ConfigError, ConfigConflict, ValueError) as e:
        raise config_error(e) from None
    record(request, audit.ADMIN_CONFIG_SAVED, user, kind=_cfg(request).store.config_version(vid)["kind"].rstrip(":"),
           version=vid, restored=version_id)
    return {"version": vid}


@router.get("/admin/api/config/export/registry")
def export_registry(request: Request, user: AdminUser = Depends(admin_user)):
    """The registry as a CLIENTS_FILE (key hashes only, no secrets), e.g. to keep it in version control."""
    content, version, source = _cfg(request).registry_content()
    header = (f"# Client registry exported from the admin dashboard ({source}, version {version or '-'}) on "
              f"{datetime.now(UTC).isoformat(timespec='seconds')}.\n# Holds key hashes only. Clients with "
              "managed_secret: true need the dashboard registry (their webhook secret is in the database).\n")
    return PlainTextResponse(header + yaml.safe_dump(content, sort_keys=False, allow_unicode=True),
                             media_type="application/x-yaml",
                             headers={"Content-Disposition": 'attachment; filename="clients.yaml"'})


# =====================================================================================================================
# Alerts
# =====================================================================================================================
def _alert_view(a: dict) -> dict:
    rule = RULES.get(a["rule"])
    state = "resolved" if a["resolvedAt"] else ("acknowledged" if a["ackedAt"] else "firing")
    return {**a, "state": state, "title": rule.title if rule else a["rule"], "kind": rule.kind if rule else None}


@router.get("/admin/api/alerts")
def alerts(request: Request, state: str | None = None, limit: int = Query(200, ge=1, le=1000),
           user: AdminUser = Depends(current_user)):
    rows = rt(request).store.alerts(state=state if state in ("open", "resolved") else None, limit=limit)
    return {"items": [_alert_view(a) for a in rows],
            "rules": [{"name": r.name, "kind": r.kind, "severity": r.severity, "title": r.title}
                      for r in RULES.values()]}


@router.post("/admin/api/alerts/{alert_id}/ack")
def ack_alert(request: Request, alert_id: int, user: AdminUser = Depends(admin_user)):
    if not rt(request).store.alert_ack(alert_id, user.username):
        raise ApiError(404, "not_found", "no open alert with this id")
    record(request, audit.ADMIN_ALERT_ACKED, user, alert=alert_id)
    return {"ok": True}
