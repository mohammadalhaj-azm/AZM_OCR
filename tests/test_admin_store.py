"""AdminStore contract (app/admin/store.py): the same tests for the in-memory and the PostgreSQL store.

The PostgreSQL variant runs only with TEST_DATABASE_URL (see tests/test_store.py)."""

import time
import uuid

import pytest

from app.admin.store import MemoryAdminStore, PgAdminStore, confidence_bin, percentiles
from app.api.store import AuditEvent, InMemoryJobStore, Job
from tests.test_store import PG_URL, _drop_tables, pg_store


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(
    not PG_URL, reason="TEST_DATABASE_URL not set"))])
def stores(request, tmp_path):
    if request.param == "memory":
        jobs = InMemoryJobStore(tmp_path / "uploads")
        jobs.open()
        yield jobs, MemoryAdminStore(jobs)
    else:
        _drop_tables()
        jobs = pg_store()
        jobs.open()
        yield jobs, PgAdminStore(jobs)
    jobs.close()


def facts(ms=1000, pages=1, doc_type="passport", conf=0.93, decided=("rules", "llm"), timings=None, **kw):
    out = {"ms": ms, "pages": pages, "pageMethods": {"ocr": pages}, "languages": ["ar", "en"],
           "timingsMs": timings or {"extract": 400, "ocr": 300, "ocrPages": pages, "llm": 500, "total": ms}}
    if doc_type:
        out["idDetection"] = {"isId": doc_type != "not_id", "docType": doc_type, "confidence": conf,
                              "decidedBy": list(decided), "mrzFound": False, "mrzValid": False}
    return {**out, **kw}


def _add(jobs, **kw):
    """A job whose job.received event carries its id (as the upload route does)."""
    owner, status, rid = kw.get("owner", "c1"), kw.get("status", "done"), kw.get("rid")
    job_id = uuid.uuid4().hex
    data = {"endpoint": "documents", "size": 10, "sha256": kw.get("sha", "ab" * 32), "mimeType": "image/png"}
    if "ext" in kw:
        data["externalIdHash"] = kw["ext"]
    received = AuditEvent("job.received", owner, job_id, rid, data)
    jobs.add(Job(id=job_id, owner=owner, filename="SECRET_NAME.pdf", size=10, request_id=rid), b"x", audit=received)
    if status in ("done", "failed", "processing"):
        claimed = jobs.claim(2, "w")
        assert claimed is not None and claimed.id == job_id
    if status in ("done", "failed"):
        data = kw.get("data") or (facts() if status == "done" else {"ms": 5, "error": "corrupt"})
        jobs.finish(job_id, "w", audit=AuditEvent("job.done" if status == "done" else "job.failed", owner, job_id,
                                                   rid, data), status=status, finished_at=time.time())
    return job_id


def test_percentiles_and_bins():
    assert percentiles([]) == {"n": 0, "p50": None, "p95": None, "avg": None, "max": None}
    p = percentiles([1, 2, 3, 4])
    assert p["p50"] == 2.5 and p["max"] == 4 and p["n"] == 4 and abs(p["p95"] - 3.85) <= 0.051
    assert confidence_bin(0.0) == "1" and confidence_bin(0.95) == "10" and confidence_bin(1.0) == "10"
    assert confidence_bin(None) is None


def test_documents_statuses_and_filters(stores):
    jobs, admin = stores
    done = _add(jobs, status="done", rid="r-done", ext="hash-1")
    failed = _add(jobs, status="failed", owner="c2")
    queued = _add(jobs, status="queued")
    jobs.record(AuditEvent("extract.done", "c1", None, "r-ex", {"endpoint": "extract", "size": 5, "pages": 2,
                                                                   "ms": 30, "sha256": "cd" * 32}))
    jobs.record(AuditEvent("request.error", "c1", None, "r-bad", {"method": "POST", "route": "/v1/extract",
                                                                     "status": 415, "code": "unsupported_type"}))
    jobs.record(AuditEvent("request.error", None, None, "r-401", {"method": "GET", "route": "/v1/jobs/{job_id}",
                                                                     "status": 401, "code": "unauthorized"}))
    docs = admin.documents(limit=50)
    by_id = {d["id"]: d for d in docs}
    assert by_id[done]["status"] == "done" and by_id[done]["docType"] == "passport" and by_id[done]["pages"] == 1
    assert by_id[failed]["status"] == "failed" and by_id[failed]["errorCode"] == "corrupt"
    assert by_id[queued]["status"] == "queued"
    kinds = sorted(d["kind"] for d in docs)
    assert kinds == ["extract", "job", "job", "job", "rejected"]  # the 401 on /v1/jobs is not a document
    rejected = next(d for d in docs if d["kind"] == "rejected")
    assert rejected["status"] == "rejected" and rejected["errorCode"] == "unsupported_type"
    assert rejected["httpStatus"] == 415
    assert [d["id"] for d in admin.documents(status="failed")] == [failed]
    assert [d["id"] for d in admin.documents(client_id="c2")] == [failed]
    assert [d["id"] for d in admin.documents(doc_type="passport")] == [done]
    assert [d["id"] for d in admin.documents(q=done)] == [done]
    assert [d["id"] for d in admin.documents(q="r-done")] == [done]
    assert [d["id"] for d in admin.documents(q="nothing", q_hash="hash-1")] == [done]
    assert {d["kind"] for d in admin.documents(kind="extract")} == {"extract"}
    # newest first, cursor paging
    page1 = admin.documents(limit=2)
    page2 = admin.documents(limit=10, before=page1[-1]["cursor"])
    assert len(page1) == 2 and not {d["id"] for d in page1} & {d["id"] for d in page2}
    assert len(page1) + len(page2) == len(docs)
    # deleted before finishing
    jobs.delete(queued, AuditEvent("job.deleted", "c1", queued, None, {"by": "client"}))
    assert next(d for d in admin.documents() if d["id"] == queued)["status"] == "deleted"
    assert "SECRET_NAME" not in repr(admin.documents())


def test_stats_series_and_usage(stores):
    jobs, admin = stores
    _add(jobs, data=facts(ms=1000, timings={"extract": 400, "ocr": 300, "ocrPages": 2, "llm": 600, "total": 1000},
                          fallbacks=["vision"]))
    _add(jobs, data=facts(ms=3000, doc_type="not_id", conf=0.55, decided=("rules",),
                          timings={"extract": 2000, "ocr": 1000, "ocrPages": 1, "total": 3000}))
    _add(jobs, status="failed", owner="c2")
    jobs.record(AuditEvent("extract.done", "c1", None, None, {"pages": 3, "pageMethods": {"text_layer": 3}, "ms": 50,
                                                                 "timingsMs": {"extract": 50, "total": 50}}))
    jobs.record(AuditEvent("request.error", "c2", None, None, {"route": "/v1/documents", "status": 503,
                                                                  "code": "queue_full"}))
    st = admin.stats(time.time() - 60)
    assert st["events"]["job.done"] == 2 and st["events"]["job.failed"] == 1 and st["events"]["extract.done"] == 1
    assert st["pages"] == 5 and st["ids"] == 1
    assert st["durations"]["job"]["n"] == 2 and st["durations"]["job"]["p50"] == 2000.0
    assert st["durations"]["extract"]["n"] == 1
    assert st["stages"]["ocrPerPage"]["n"] == 2 and st["stages"]["ocrPerPage"]["max"] == 1000.0
    assert st["stages"]["llm"]["n"] == 1 and st["stages"]["llm"]["p50"] == 600.0
    assert st["stages"]["vision"]["n"] == 0
    d = st["dist"]
    assert d["docType"] == {"passport": 1, "not_id": 1} and d["decidedBy"] == {"llm": 1, "rules": 1}
    assert d["fallback"] == {"vision": 1} and d["error"] == {"corrupt": 1}
    assert d["method"] == {"ocr": 2, "text_layer": 3} and d["httpStatus"] == {"503": 1}
    assert d["confidence"] == {"10": 1, "6": 1} and d["language"] == {"ar": 2, "en": 2}
    assert admin.stats(time.time() - 60, client_id="c2")["events"] == {"job.received": 1, "job.started": 1,
                                                                        "job.failed": 1, "request.error": 1}
    series = admin.series(time.time() - 3600, 60)
    assert sum(r["n"] for r in series if r["event"] == "job.done") == 2
    usage = admin.client_usage(time.time() - 60)
    assert usage["c2"]["httpStatus"] == {"503": 1} and usage["c2"]["errors"] == {"queue_full": 1, "corrupt": 1}
    assert usage["c1"]["avgMs"] == 2000.0 and usage["c1"]["lastSeen"] > 0
    assert all(v == {"queued": 0, "processing": 0, "webhooksPending": 0} for v in admin.live_by_owner().values())


def test_audit_page_filters(stores):
    jobs, admin = stores
    for i in range(5):
        jobs.record(AuditEvent("admin.login", None, None, f"r{i}", {"actor": "alice" if i % 2 else "bob"}))
    jobs.record(AuditEvent("job.fetched", "c1", "j1", "rx", {"status": "done"}))
    assert [e["event"] for e in admin.audit_page(limit=2)] == ["job.fetched", "admin.login"]
    assert len(admin.audit_page(event="admin.")) == 5 and len(admin.audit_page(event="admin.login")) == 5
    assert len(admin.audit_page(actor="alice")) == 2
    assert admin.audit_page(job_id="j1")[0]["requestId"] == "rx"
    first = admin.audit_page(limit=3)
    rest = admin.audit_page(before=first[-1]["id"])
    assert len(first) + len(rest) == 6
    assert admin.latest_event_id() == first[0]["id"]


def test_samples_alerts_logs_config_secrets(stores):
    jobs, admin = stores
    admin.write_sample("i1", {"cpu": {"percent": 10}})
    admin.write_sample("i1", {"cpu": {"percent": 20}})
    admin.write_sample("i2", {"cpu": {"percent": 30}})
    latest = {r["instance"]: r["data"]["cpu"]["percent"] for r in admin.latest_samples()}
    assert latest == {"i1": 20, "i2": 30}
    assert [p["data"]["cpu"]["percent"] for p in admin.samples("i1", time.time() - 60)] == [10, 20]
    assert admin.purge_samples_before(time.time() + 1) == 3 and admin.latest_samples() == []

    a1, new = admin.alert_fire("cpu_high", "i1", "warning", "CPU high", {"percent": 95})
    a2, again = admin.alert_fire("cpu_high", "i1", "warning", "CPU higher", {"percent": 97})
    assert new and not again and a1 == a2
    assert [a["summary"] for a in admin.alerts(state="open")] == ["CPU higher"]
    assert admin.alert_ack(a1, "alice") and admin.alerts(state="open")[0]["ackedBy"] == "alice"
    assert admin.alert_resolve("cpu_high", "i1") and not admin.alert_resolve("cpu_high", "i1")
    assert admin.alerts(state="open") == [] and admin.alerts(state="resolved")[0]["resolvedAt"]
    a3, new = admin.alert_fire("cpu_high", "i1", "warning", "again", {})
    assert new and a3 != a1 and not admin.alert_ack(a1, "bob")

    now = time.time()
    admin.write_logs([{"at": now, "instance": "i1", "level": "WARNING", "logger": "app.api", "msg": "slow thing",
                       "request_id": "r1"},
                      {"at": now, "instance": "i2", "level": "ERROR", "logger": "app.admin", "msg": "broken",
                       "exc_type": "ValueError"}])
    assert [r["msg"] for r in admin.logs()] == ["broken", "slow thing"]
    assert [r["msg"] for r in admin.logs(level="ERROR")] == ["broken"]
    assert [r["msg"] for r in admin.logs(request_id="r1")] == ["slow thing"]
    assert [r["msg"] for r in admin.logs(text="BROK")] == ["broken"]
    assert admin.logs(logger="app.api")[0]["msg"] == "slow thing"
    assert admin.purge_logs_before(now + 1) == 2

    v1 = admin.config_save("settings", {"overrides": {"ocr_dpi": 200}}, "alice", "first")
    v2 = admin.config_save("rules:civil", {"yaml": "keywords: []"}, "bob")
    v3 = admin.config_save("settings", {"overrides": {}}, "alice")
    assert admin.config_heads() == {"settings": v3, "rules:civil": v2}
    assert admin.config_latest("settings")["id"] == v3 and admin.config_latest("registry") is None
    assert [h["id"] for h in admin.config_history("settings")] == [v3, v1]
    assert [h["id"] for h in admin.config_history("rules:")] == [v2]
    assert "content" not in admin.config_history()[0]
    assert admin.config_version(v1)["content"] == {"overrides": {"ocr_dpi": 200}}

    admin.secret_set("c1", "s3cret-value")
    admin.secret_set("c1", "rotated-value")
    assert admin.secrets() == {"c1": "rotated-value"}


@pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set")
def test_pg_secrets_are_sealed():
    import psycopg

    _drop_tables()
    jobs = pg_store()
    jobs.open()
    try:
        PgAdminStore(jobs).secret_set("c1", "PLAIN_WEBHOOK_SECRET")
        with psycopg.connect(PG_URL) as conn:
            raw = conn.execute("SELECT sealed FROM ocr_client_secrets").fetchone()[0]
        assert b"PLAIN_WEBHOOK_SECRET" not in bytes(raw)
        assert PgAdminStore(jobs).db_info()["sizeBytes"] > 0
    finally:
        jobs.close()
