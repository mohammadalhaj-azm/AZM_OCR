"""Admin dashboard (app/admin): authentication, every endpoint on real requests, privacy, live configuration,
alerts, logs and resource sampling. Uses the memory job store (tests/test_admin_store.py covers PostgreSQL)."""

import logging
import os
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from app.admin.alerts import AlertEvaluator
from app.admin.auth import hash_password
from app.admin.logbuffer import LogBuffer, wait_flushed
from app.admin.store import MemoryAdminStore
from app.admin.system import ResourceSampler
from app.api import audit
from app.api.clients import key_hash, registry_from_config
from app.api.main import create_app
from app.api.store import AuditEvent, InMemoryJobStore
from app.config import ConfigError, Settings
from app.extractors import FileRejected
from tests.test_api import KEY, StubServices, fake_analyze, make_settings

ADMIN_PW, VIEWER_PW = "admin password 123", "viewer password 456"
A = {"X-Admin-Request": "1"}
SECRET = "test-session-secret-0123456789-abcdefgh"
CLIENT_KEY = "client-key-for-registry-tests-0123456789abc"


def analyze(data, filename, svc, detect_id=True, **kw):
    if "BAD" in filename:
        raise FileRejected("corrupt", "the file could not be opened")
    if "FALLBACK" in filename:
        audit.note_fallback("llm")
    return fake_analyze(data, filename, svc, detect_id=detect_id, **kw)


def write_users(path, admin_pw=ADMIN_PW):
    path.write_text(yaml.safe_dump({"admins": {
        "alice": {"name": "Alice", "role": "admin", "password_scrypt": hash_password(admin_pw)},
        "victor": {"name": "Victor", "role": "viewer", "password_scrypt": hash_password(VIEWER_PW)},
    }}), encoding="utf-8")


@pytest.fixture
def make_admin(tmp_path):
    clients = []

    def _make(registry: bool = False, **kw):
        users = tmp_path / "admins.yaml"
        if not users.exists():
            write_users(users)
        base = dict(admin_users_file=str(users), admin_session_secret=SECRET, sample_interval_s=0.5,
                    alert_eval_s=3600, config_poll_s=3600)
        if registry:  # a client registry (CLIENTS_FILE) instead of API_KEYS
            f = tmp_path / "clients.yaml"
            f.write_text(yaml.safe_dump({"profiles": {"text-only": {"detect_id": False}}, "clients": {
                "civil": {"name": "Civil", "key_sha256": [key_hash(CLIENT_KEY)], "profiles": ["default", "text-only"]},
            }}), encoding="utf-8")
            base.update(api_keys="", clients_file=str(f))
        s = make_settings(tmp_path, **{**base, **kw})
        c = TestClient(create_app(s, services=StubServices(), analyze_fn=analyze))
        c.__enter__()
        clients.append(c)
        return c, s

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def login(c, user="alice", pw=ADMIN_PW):
    r = c.post("/admin/api/login", json={"username": user, "password": pw}, headers=A)
    assert r.status_code == 200, r.text
    return r


def upload(c, name="doc.png", key=KEY, **form):
    return c.post("/v1/documents", files={"file": (name, b"\x89PNG\r\n\x1a\n" + b"0" * 64)}, data=form,
                  headers={"X-Api-Key": key, "X-Request-Id": "req-" + name[:8]})


def wait_jobs(c, n, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if c.app.state.store.counts()["queued"] == 0 and c.app.state.store.counts()["processing"] == 0 \
                and sum(c.app.state.store.counts().values()) >= n:
            return
        time.sleep(0.05)
    raise AssertionError("jobs did not finish")


# --- authentication ---------------------------------------------------------------------------------------------
def test_dashboard_is_off_without_users_file(tmp_path):
    with TestClient(create_app(make_settings(tmp_path), services=StubServices(), analyze_fn=fake_analyze)) as c:
        assert c.get("/admin").status_code == 404
        assert c.get("/admin/api/me").status_code == 404


def test_login_session_logout(make_admin):
    c, _ = make_admin()
    r = c.get("/admin", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/admin/login"
    assert c.get("/admin/api/me").status_code == 401
    assert c.post("/admin/api/login", json={"username": "alice", "password": "nope"}, headers=A).status_code == 401
    assert c.post("/admin/api/login", json={"username": "my-password?", "password": "x"}, headers=A).status_code == 401
    failed = c.app.state.store.audit_events()
    assert [e["data"].get("user") for e in failed if e["event"] == "admin.login_failed"] == ["alice", None]
    login(c)
    cookie = c.cookies.get("ocr_admin")
    assert cookie and "alice" not in cookie  # signed payload, base64
    me = c.get("/admin/api/me").json()
    assert me["user"] == {"username": "alice", "name": "Alice", "role": "admin"} and me["persistent"] is False
    page = c.get("/admin")
    assert page.status_code == 200 and "<script type=\"module\"" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"] and page.headers["x-frame-options"] == "DENY"
    assert c.get("/admin/static/js/app.js").status_code == 200
    assert c.post("/admin/api/logout", headers=A).status_code == 200
    assert c.get("/admin/api/me").status_code == 401
    # a forged cookie is ignored
    c.cookies.set("ocr_admin", cookie[:-3] + "AAA", path="/admin")
    assert c.get("/admin/api/me").status_code == 401


def test_login_throttling(make_admin):
    c, _ = make_admin()
    for _ in range(5):
        assert c.post("/admin/api/login", json={"username": "alice", "password": "bad"}, headers=A).status_code == 401
    r = c.post("/admin/api/login", json={"username": "alice", "password": ADMIN_PW}, headers=A)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1  # even the right password waits


def test_csrf_and_roles(make_admin):
    c, _ = make_admin()
    assert c.post("/admin/api/login", json={"username": "alice", "password": ADMIN_PW}).status_code == 403
    login(c, "victor", VIEWER_PW)
    assert c.get("/admin/api/overview").status_code == 200
    r = c.put("/admin/api/config/settings", json={"patch": {"ocr_dpi": 200}}, headers=A)
    assert r.status_code == 403 and "admin role" in r.json()["error"]["message"]
    login(c)
    assert c.put("/admin/api/config/settings", json={"patch": {"ocr_dpi": 200}}).status_code == 403
    assert c.put("/admin/api/config/settings", json={"patch": {"ocr_dpi": 200}},
                 headers={**A, "Origin": "https://evil.example"}).status_code == 403
    assert c.put("/admin/api/config/settings", json={"patch": {"ocr_dpi": 200}},
                 headers={**A, "Origin": "http://testserver"}).status_code == 200


def test_password_change_ends_sessions(make_admin, tmp_path):
    c, _ = make_admin()
    login(c)
    users = tmp_path / "admins.yaml"
    write_users(users, admin_pw="a brand new password")
    os.utime(users, (time.time() + 5, time.time() + 5))
    assert c.get("/admin/api/me").status_code == 401
    login(c, pw="a brand new password")


def test_production_needs_session_secret(tmp_path):
    s = make_settings(tmp_path, environment="production", admin_users_file="x.yaml", admin_session_secret="short")
    assert any("ADMIN_SESSION_SECRET" in p for p in s.production_problems())


# --- dashboard data ---------------------------------------------------------------------------------------------
GETS = ["/admin/api/me", "/admin/api/live", "/admin/api/overview?range=1h", "/admin/api/documents",
        "/admin/api/system", "/admin/api/system/history", "/admin/api/ai", "/admin/api/integrations",
        "/admin/api/logs", "/admin/api/logs?source=all", "/admin/api/audit", "/admin/api/audit?format=csv",
        "/admin/api/config/settings", "/admin/api/config/registry", "/admin/api/config/rules",
        "/admin/api/config/history", "/admin/api/alerts", "/admin/api/stream?once=1",
        "/admin/api/trace?requestId=req-MARKER_F"]


def test_every_endpoint_and_no_document_data(make_admin, caplog):
    caplog.set_level(logging.INFO)  # as under uvicorn (app.logs.configure)
    c, _ = make_admin()
    login(c)
    job = upload(c, "MARKER_FILE.png", externalId="EXT-REF-1").json()["jobId"]
    upload(c, "BAD_MARKER.png")
    upload(c, "FALLBACK.png")
    c.post("/v1/extract", files={"file": ("MARKER_SYNC.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)},
           headers={"X-Api-Key": KEY})
    c.post("/v1/documents", files={"file": ("x.bin", b"not a document")}, headers={"X-Api-Key": KEY})
    c.get("/v1/jobs/x", headers={"X-Api-Key": "unknown-key"})
    wait_jobs(c, 3)
    logging.getLogger("app.api").warning("a warning for the stored logs")
    time.sleep(2.5)  # the log writer stores WARNING+ every 2 s
    for path in GETS + [f"/admin/api/documents/{job}", f"/admin/api/integrations/{key_hash(KEY)[:16]}",
                        f"/admin/api/trace?jobId={job}"]:
        r = c.get(path)
        assert r.status_code == 200, (path, r.text[:300])
        for secret in ("SECRET_TEXT", "MARKER_FILE", "BAD_MARKER", "MARKER_SYNC", "EXT-REF-1", KEY, key_hash(KEY),
                       SECRET, "password_scrypt"):
            assert secret not in r.text, (path, secret)

    ov = c.get("/admin/api/overview?range=1h").json()
    assert ov["kpis"]["documents"] == 4 and ov["kpis"]["failed"] == 1 and ov["kpis"]["extracts"] == 1
    assert ov["kpis"]["requestErrors"] == 2 and ov["live"]["jobs"]["done"] == 2
    assert ov["instances"] and ov["instances"][0]["self"]

    docs = c.get("/admin/api/documents").json()["items"]
    assert sorted(d["status"] for d in docs) == ["done", "done", "done", "failed", "rejected"]
    assert c.get("/admin/api/documents", params={"q": "EXT-REF-1"}).json()["items"][0]["id"] == job  # hashed
    assert c.get("/admin/api/documents", params={"status": "failed"}).json()["items"][0]["errorCode"] == "corrupt"

    d = c.get(f"/admin/api/documents/{job}").json()
    assert [t["event"] for t in d["timeline"]][:3] == ["job.received", "job.started", "job.done"]
    assert d["job"]["status"] == "done" and d["document"]["docType"] == "not_id"

    ai = c.get("/admin/api/ai").json()
    assert ai["decisions"]["fallbacks"] == {"llm": 1} and ai["llm"]["models"][0]["installed"] is True
    integ = c.get("/admin/api/integrations").json()
    assert integ["unauthenticated"]["requests"] == 1 and len(integ["clients"]) == 2
    mine = next(x for x in integ["clients"] if x["id"] == key_hash(KEY)[:16])
    assert mine["usage"]["events"]["job.received"] == 3 and mine["health"] in ("warning", "critical", "ok")

    logs = c.get("/admin/api/logs", params={"text": "job finished"}).json()["items"]
    assert logs and all(r["msg"].startswith("job finished") for r in logs)
    assert any(r["msg"] == "a warning for the stored logs" for r in c.get("/admin/api/logs?source=all").json()["items"])
    assert c.get("/admin/api/audit", params={"event": "admin."}).json()["items"][0]["event"] == "admin.login"

    stream = c.get("/admin/api/stream?once=1")
    assert stream.headers["content-type"].startswith("text/event-stream") and "event: tick" in stream.text
    live = c.get("/admin/api/live").json()
    assert live["sample"]["cpu"]["count"] >= 1 and live["sample"]["health"]["worker"] is True
    assert c.get("/admin/api/system/history").json()["points"]


def test_admin_deletes_a_job(make_admin):
    c, _ = make_admin()
    job = upload(c, "one.png").json()["jobId"]
    wait_jobs(c, 1)
    login(c, "victor", VIEWER_PW)
    assert c.delete(f"/admin/api/documents/{job}", headers=A).status_code == 403
    login(c)
    assert c.delete(f"/admin/api/documents/{job}", headers=A).status_code == 204
    assert c.get(f"/v1/jobs/{job}", headers={"X-Api-Key": KEY}).status_code == 404
    deleted = [e for e in c.app.state.store.audit_events(job_id=job) if e["event"] == "job.deleted"]
    assert deleted[0]["data"] == {"by": "admin", "actor": "alice"}
    assert c.get(f"/admin/api/documents/{job}").json()["document"]["status"] == "done"  # history outlives the job


def test_openapi_unchanged_by_the_dashboard(make_admin, tmp_path):
    from scripts.export_openapi import render

    c, _ = make_admin()
    plain_app = create_app(make_settings(tmp_path / "plain"), services=StubServices(), analyze_fn=fake_analyze)
    with TestClient(plain_app) as plain:
        assert render(c.app) == render(plain.app)


# --- configuration ----------------------------------------------------------------------------------------------
def test_live_settings(make_admin):
    c, s = make_admin()
    login(c)
    r = c.put("/admin/api/config/settings", json={"patch": {"max_queued_jobs": 0}, "comment": "maintenance"}, headers=A)
    assert r.status_code == 200 and s.max_queued_jobs == 0
    assert upload(c).status_code == 503  # applied without a restart
    v1 = r.json()["version"]
    assert c.put("/admin/api/config/settings", json={"patch": {"vision_threshold": 3}}, headers=A).status_code == 422
    assert c.put("/admin/api/config/settings", json={"patch": {"database_url": "x"}}, headers=A).status_code == 422
    assert c.put("/admin/api/config/settings", json={"patch": {"ocr_engines": 4}, "baseVersion": 999},
                 headers=A).status_code == 409
    r = c.put("/admin/api/config/settings", json={"patch": {"ocr_engines": 4, "max_queued_jobs": None}}, headers=A)
    assert r.json()["pendingRestart"] == {"ocr_engines": 4} and s.ocr_engines == 2 and s.max_queued_jobs == 100
    assert upload(c).status_code == 202
    rows = {x["key"]: x for x in c.get("/admin/api/config/settings").json()["items"]}
    assert rows["ocr_engines"]["pending"] == 4 and rows["ocr_engines"]["mode"] == "restart"
    assert rows["database_url"]["value"] == "not set" and rows["api_keys"]["value"] == "set"  # secrets never shown
    assert rows["admin_session_secret"]["value"] == "set" and SECRET not in str(rows)
    hist = c.get("/admin/api/config/history").json()["items"]
    assert [h["actor"] for h in hist] == ["alice", "alice"]
    assert c.post(f"/admin/api/config/versions/{v1}/restore", headers=A).status_code == 200
    assert s.max_queued_jobs == 0 and c.get("/admin/api/config/settings").json()["pendingRestart"] == {}
    assert [e["data"]["kind"] for e in c.app.state.store.audit_events() if e["event"] == "admin.config_saved"][:2] \
        == ["settings", "settings"]


def test_registry_not_editable_with_api_keys(make_admin):
    c, _ = make_admin()
    login(c)
    reg = c.get("/admin/api/config/registry").json()
    assert reg["editable"] is False and "API_KEYS" in reg["reason"]
    r = c.put("/admin/api/config/clients/new", json={"client": {"name": "New"}}, headers=A)
    assert r.status_code == 422


def test_registry_management(make_admin):
    c, s = make_admin(registry=True)
    login(c)
    reg = c.get("/admin/api/config/registry").json()
    assert reg["source"] == "file" and reg["editable"] and key_hash(CLIENT_KEY) not in str(reg)
    assert reg["registry"]["clients"]["civil"]["keys"] == [key_hash(CLIENT_KEY)[:8]]
    assert upload(c, key=CLIENT_KEY).status_code == 202

    # a new client: its key works at once
    r = c.put("/admin/api/config/clients/archive", json={"client": {"name": "Archive", "scopes": ["extract"]},
                                                          "baseVersion": None}, headers=A)
    assert r.status_code == 200, r.text
    key = r.json()["key"]
    assert c.app.state.clients.source == "dashboard"
    files = {"file": ("a.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64)}
    assert c.post("/v1/extract", files=files, headers={"X-Api-Key": key}).status_code == 200
    assert upload(c, key=key).status_code == 403  # no documents scope
    # second key, revoke the first; the last key can't be revoked
    k2 = c.post("/admin/api/config/clients/archive/keys", headers=A).json()
    assert c.delete(f"/admin/api/config/clients/archive/keys/{key_hash(key)[:8]}", headers=A).status_code == 200
    assert c.post("/v1/extract", files=files, headers={"X-Api-Key": key}).status_code == 401
    assert c.post("/v1/extract", files=files, headers={"X-Api-Key": k2["key"]}).status_code == 200
    assert c.delete(f"/admin/api/config/clients/archive/keys/{k2['fingerprint']}", headers=A).status_code == 422
    # disable
    r = c.put("/admin/api/config/clients/archive", json={"client": {"name": "Archive", "scopes": ["extract"],
                                                                     "enabled": False}}, headers=A)
    assert r.status_code == 200 and r.json()["key"] is None
    assert c.post("/v1/extract", files=files, headers={"X-Api-Key": k2["key"]}).status_code == 401
    # webhook secret generated in the dashboard
    r = c.post("/admin/api/config/clients/civil/webhook-secret", json={"allowed_hosts": ["hooks.civil.internal"]},
               headers=A)
    assert r.status_code == 200 and len(r.json()["secret"]) >= 32
    civil = c.app.state.clients.get("civil")
    assert civil.webhook_secret == r.json()["secret"] and civil.webhook_hosts == {"hooks.civil.internal"}
    assert upload(c, key=CLIENT_KEY, callbackUrl="https://hooks.civil.internal/done").status_code == 202
    reg = c.get("/admin/api/config/registry").json()
    assert reg["registry"]["clients"]["civil"]["webhook"] == {"managed_secret": True,
                                                               "allowed_hosts": ["hooks.civil.internal"],
                                                               "secretStored": True}
    assert r.json()["secret"] not in str(reg)
    # profiles
    assert c.put("/admin/api/config/profiles/civil-ids", json={"profile": {"max_pages": 5, "bogus": 1}},
                 headers=A).status_code == 422
    r = c.put("/admin/api/config/profiles/civil-ids", json={"profile": {"max_pages": 5}}, headers=A)
    assert r.status_code == 200
    assert c.delete("/admin/api/config/profiles/text-only", headers=A).status_code == 422  # used by civil
    assert c.delete("/admin/api/config/profiles/civil-ids", headers=A).status_code == 200
    # export round-trips through the normal validation
    text = c.get("/admin/api/config/export/registry").text
    rebuilt = registry_from_config(yaml.safe_load(text), s, managed_secret=lambda cid: "x" * 40)
    assert {cl.id for cl in rebuilt.clients} == {"civil", "archive"}
    assert key_hash(CLIENT_KEY) in text and CLIENT_KEY not in text
    # history + key audit
    events = [e["event"] for e in c.app.state.store.audit_events() if e["event"].startswith("admin.")]
    assert events.count("admin.key_created") == 2 and "admin.key_revoked" in events and "admin.secret_created" in events
    assert key not in str(c.app.state.store.audit_events())


def test_rule_sets(make_admin, tmp_path):
    c, s = make_admin()
    login(c)
    bad = c.post("/admin/api/config/rules/validate", json={"yaml": "keywords: [{text: x}]"}, headers=A).json()
    assert bad["ok"] is False and "malformed" in bad["error"]
    good_yaml = "keywords:\n  - {text: 'BADGE NUMBER', weight: 0.5, doc_type: other_id}\n"
    ok = c.post("/admin/api/config/rules/validate", json={"yaml": good_yaml}, headers=A).json()
    assert ok["ok"] and ok["summary"]["byDocType"]["other_id"]["keywords"] == 1
    assert c.put("/admin/api/config/rules/staff", json={"yaml": "keywords: 5"}, headers=A).status_code == 422
    assert c.put("/admin/api/config/rules/staff", json={"yaml": good_yaml}, headers=A).status_code == 200
    assert c.put("/admin/api/config/settings", json={"patch": {"id_patterns_path": "db:missing"}},
                 headers=A).status_code == 422
    assert c.put("/admin/api/config/settings", json={"patch": {"id_patterns_path": "db:staff"}},
                 headers=A).status_code == 200
    rules = c.get("/admin/api/config/rules").json()
    assert rules["active"] == "db:staff" and rules["summary"]["keywords"] == 1
    assert rules["sets"]["staff"]["usedBy"] == ["settings: id_patterns_path"]
    assert c.delete("/admin/api/config/rules/staff", headers=A).status_code == 422
    from app.detection.detector import IdDetector

    assert IdDetector(s, None).rules.keywords[0].text == "BADGE NUMBER"
    c.put("/admin/api/config/settings", json={"patch": {"id_patterns_path": None}}, headers=A)
    assert c.delete("/admin/api/config/rules/staff", headers=A).status_code == 200


# --- alerts -----------------------------------------------------------------------------------------------------
def _evaluator(tmp_path, sample, **kw):
    jobs = InMemoryJobStore(tmp_path / "u")
    jobs.open()
    store = MemoryAdminStore(jobs)
    settings = Settings(_env_file=None, alert_for_s=kw.pop("alert_for_s", 0), **kw)
    state = {"sample": sample, "audit": 0.0, "config": None}
    ev = AlertEvaluator(settings, store, jobs, "me", lambda: state["sample"], lambda: state["audit"],
                        lambda: state["config"], lambda: ["civil"])
    return ev, store, jobs, state


def test_alert_rules_fire_and_resolve(tmp_path):
    healthy = {"health": {"store": True, "ocrModels": True, "worker": True, "llm": True, "vision": True},
               "cpu": {"percent": 10}, "memory": {"percent": 20}, "disks": [{"label": "app", "percent": 50, "free": 1}],
               "gpus": [{"index": 0, "memPercent": 10}]}
    ev, store, jobs, state = _evaluator(tmp_path, healthy)
    assert ev.run_once() == [] and store.alerts() == []
    state["sample"] = {**healthy, "health": {**healthy["health"], "worker": False, "llm": False},
                       "cpu": {"percent": 99}, "gpus": [{"index": 0, "memPercent": 99}]}
    opened = ev.run_once()
    assert {a["rule"] for a in store.alerts(state="open")} == {"worker_down", "llm_unavailable", "cpu_high",
                                                               "gpu_memory_high"}
    assert len(opened) == 4 and ev.run_once() == []  # no duplicates
    assert next(a for a in store.alerts() if a["rule"] == "gpu_memory_high")["scope"] == "me|gpu0"
    state["sample"] = healthy
    ev.run_once()
    assert store.alerts(state="open") == [] and len(store.alerts(state="resolved")) == 4

    # global and client rules from the audit trail
    for _ in range(5):
        jobs.record(AuditEvent("job.failed", "civil", "j", None, {"error": "internal_error"}))
    jobs.record(AuditEvent("webhook.failed", "civil", "j", None, {"attempts": 9}))
    jobs.record(AuditEvent("job.done", "civil", "j", None, {"fallbacks": ["llm"]}))
    jobs.record(AuditEvent("request.error", "civil", None, None, {"status": 500, "code": "internal_error"}))
    state["audit"], state["config"] = 3.0, {"kind": "registry", "version": 7, "error": "bad"}
    ev.run_once()
    rules = {(a["rule"], a["scope"]) for a in store.alerts(state="open")}
    assert rules == {("job_failure_rate", ""), ("webhooks_failed", "client:civil"), ("model_fallbacks", ""),
                     ("server_errors", ""), ("audit_write_failures", "me"), ("config_apply_failed", "me")}

    # another replica stopped reporting; one that stopped cleanly is no alert
    store.write_sample("other", {"host": "h2"})
    store.write_sample("gone-clean", {"host": "h3", "stopping": True})
    ev.run_once(now=time.time() + 120)
    assert ("replica_missing", "other") in {(a["rule"], a["scope"]) for a in store.alerts(state="open")}
    assert not any(a["scope"] == "gone-clean" for a in store.alerts())


def test_resource_alerts_wait_for_the_hold_time(tmp_path):
    hot = {"health": {}, "cpu": {"percent": 99}, "memory": {"percent": 10}, "disks": [], "gpus": []}
    ev, store, _, _ = _evaluator(tmp_path, hot, alert_for_s=300)
    now = time.time()
    ev.run_once(now=now)
    assert store.alerts() == []
    ev.run_once(now=now + 301)
    assert [a["rule"] for a in store.alerts(state="open")] == ["cpu_high"]


def test_alert_api_ack(make_admin):
    c, _ = make_admin()
    store = c.app.state.admin.store
    aid, _ = store.alert_fire("cpu_high", "x", "warning", "CPU high", {})
    login(c, "victor", VIEWER_PW)
    assert c.post(f"/admin/api/alerts/{aid}/ack", headers=A).status_code == 403
    login(c)
    assert c.post(f"/admin/api/alerts/{aid}/ack", headers=A).status_code == 200
    items = c.get("/admin/api/alerts?state=open").json()["items"]
    assert items[0]["state"] == "acknowledged" and items[0]["ackedBy"] == "alice"
    assert c.get("/admin/api/live").json()["alerts"]["open"] == 1


# --- logs and sampling ------------------------------------------------------------------------------------------
class FailingStore:
    def write_logs(self, records):
        logging.getLogger("app.admin").error("this must not loop")
        raise RuntimeError("database down")


def test_log_buffer_filters_and_never_loops():
    buf = LogBuffer("me", 100, FailingStore())
    log = logging.getLogger("test.admin.buffer")
    log.setLevel(logging.DEBUG)
    log.addHandler(buf)
    buf.start()
    try:
        log.info("hello %s", "world")
        try:
            raise ValueError("SECRET_TEXT in an exception message")
        except ValueError:
            log.exception("failed")
        assert [r["msg"] for r in buf.recent()] == ["failed", "hello world"]
        assert buf.recent(level="ERROR")[0]["exc_type"] == "ValueError"
        assert "SECRET_TEXT" not in str(buf.recent())
        wait_flushed(buf)
        time.sleep(2.5)
        assert buf.dropped >= 1  # the failing write was dropped, not retried forever
    finally:
        log.removeHandler(buf)
        buf.stop()


def test_sampler_measures_real_resources(tmp_path):
    jobs = InMemoryJobStore(tmp_path / "u")
    store = MemoryAdminStore(jobs)
    s = Settings(_env_file=None, upload_dir=tmp_path, sample_persist_s=0)
    sampler = ResourceSampler(s, "me", store, lambda: {"worker": True}, lambda: {"activeJobs": 0})
    sample = sampler.tick()
    assert 0 <= sample["cpu"]["percent"] <= 100 and sample["cpu"]["count"] >= 1
    assert sample["memory"]["total"] > 0 and sample["disks"] and sample["gpuStatus"] in ("ok", "none", "error")
    assert sampler.tick()["io"]["netRecvBps"] is not None  # rates need two samples
    assert len(store.latest_samples()) == 1 and sampler.latest() is not None


def test_invalid_users_file_fails_start(tmp_path):
    (tmp_path / "bad.yaml").write_text("admins: {Bad Name: {name: x, password_scrypt: nope}}", encoding="utf-8")
    with pytest.raises(ConfigError):
        create_app(make_settings(tmp_path, admin_users_file=str(tmp_path / "bad.yaml")), services=StubServices(),
                   analyze_fn=fake_analyze)
