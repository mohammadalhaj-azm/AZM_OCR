import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import webhook
from app.api.main import create_app
from app.config import Settings
from app.detection.detector import IdDetector
from app.extractors import FileRejected
from app.pipeline import Services, analyze
from tests.conftest import compact
from tests.test_detector import StubLlm, V

KEY, OTHER_KEY = "key-a", "key-b"
H = {"X-Api-Key": KEY}
RESULT_KEYS = {"jobId", "status", "file", "pages", "fullText", "normalizedText", "languages", "idDetection",
               "timingsMs", "processing"}


class StubServices:
    def __init__(self, models=("qwen3:8b", "qwen3-vl:8b-instruct")):
        self.llm = StubLlm(models=models)


CALLS: list[dict] = []  # keyword arguments each fake_analyze call received


def fake_analyze(data: bytes, filename: str, svc, detect_id: bool = True, max_pages: int | None = None,
                 **options) -> dict:
    CALLS.append({"detect_id": detect_id, "max_pages": max_pages, **options})
    out = {"file": {"name": filename, "mimeType": "image/png", "pages": 1},
           "pages": [{"page": 1, "method": "ocr", "text": "SECRET_TEXT", "confidence": 0.9}],
           "fullText": "SECRET_TEXT", "normalizedText": "SECRET_TEXT", "languages": ["en"],
           "timingsMs": {"total": 1}, "processing": {"version": "test"}}
    if detect_id:
        out["idDetection"] = {"isId": False, "docType": "not_id", "confidence": 0.9, "decidedBy": ["rules"],
                              "reasons": []}
    return out


def make_settings(tmp_path, **kw) -> Settings:
    base = dict(_env_file=None, api_keys=f"{KEY},{OTHER_KEY}", upload_dir=tmp_path / "uploads",
                webhook_backoff_s=0, ocr_warmup=False)
    return Settings(**{**base, **kw})


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def _make(analyze_fn=fake_analyze, services=None, transport=None, **settings_kw):
        s = make_settings(tmp_path, **settings_kw)
        app = create_app(s, services=services or StubServices(), analyze_fn=analyze_fn, webhook_transport=transport)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c, s

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def upload(client, data, name="doc.png", headers=H, **form):
    return client.post("/v1/documents", files={"file": (name, data, "application/octet-stream")}, data=form,
                       headers=headers)


def wait_done(client, job_id, headers=H, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        r = client.get(f"/v1/jobs/{job_id}", headers=headers)
        if r.json()["status"] in ("done", "failed"):
            return r.json()
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_api_key_required(make_client, fixtures):
    c, _ = make_client()
    assert upload(c, fixtures["id_card.png"], headers={}).status_code == 401
    assert upload(c, fixtures["id_card.png"], headers={"X-Api-Key": "wrong"}).status_code == 401
    assert c.get("/v1/jobs/abc").status_code == 401


def test_no_keys_configured_rejects_everything(make_client, fixtures):
    c, _ = make_client(api_keys="")
    assert upload(c, fixtures["id_card.png"], headers={"X-Api-Key": ""}).status_code == 401


def test_upload_poll_done_and_upload_deleted(make_client, fixtures):
    c, s = make_client()
    r = upload(c, fixtures["id_card.png"], name="card.png")
    assert r.status_code == 202
    job_id = r.json()["jobId"]
    out = wait_done(c, job_id)
    assert set(out) == RESULT_KEYS and out["status"] == "done" and out["file"]["name"] == "card.png"
    assert list(s.upload_dir.iterdir()) == []  # upload removed right after processing


def test_unsupported_type_is_415(make_client):
    c, _ = make_client()
    r = upload(c, b"GIF89a not supported")
    assert r.status_code == 415


def test_too_large_is_413(make_client, fixtures):
    c, _ = make_client(max_upload_mb=1)
    big = fixtures["id_card.png"] + b"\0" * (2 * 1024 * 1024)
    assert upload(c, big).status_code == 413


def test_failure_deletes_upload_and_hides_details(make_client, fixtures, caplog):
    def boom(data, filename, svc):
        raise RuntimeError("SECRET_TEXT inside exception")

    caplog.set_level(logging.DEBUG)
    c, s = make_client(analyze_fn=boom)
    out = wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    assert out["status"] == "failed" and out["error"] == {"code": "internal_error", "message": "processing failed"}
    assert list(s.upload_dir.iterdir()) == []
    assert "SECRET_TEXT" not in caplog.text


def test_rejected_file_reported(make_client, fixtures):
    def corrupt(data, filename, svc):
        raise FileRejected("corrupt", "PDF could not be opened (corrupt or encrypted)")

    c, _ = make_client(analyze_fn=corrupt)
    out = wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    assert out["status"] == "failed" and out["error"]["code"] == "corrupt"


def test_jobs_are_private_to_their_api_key(make_client, fixtures):
    c, _ = make_client()
    job_id = upload(c, fixtures["id_card.png"]).json()["jobId"]
    wait_done(c, job_id)
    assert c.get(f"/v1/jobs/{job_id}", headers={"X-Api-Key": OTHER_KEY}).status_code == 404
    assert c.delete(f"/v1/jobs/{job_id}", headers={"X-Api-Key": OTHER_KEY}).status_code == 404


def test_delete_job(make_client, fixtures):
    c, _ = make_client()
    job_id = upload(c, fixtures["id_card.png"]).json()["jobId"]
    wait_done(c, job_id)
    assert c.delete(f"/v1/jobs/{job_id}", headers=H).status_code == 204
    assert c.get(f"/v1/jobs/{job_id}", headers=H).status_code == 404


def test_results_expire_after_ttl(make_client, fixtures):
    c, _ = make_client(result_ttl_s=0)
    job_id = upload(c, fixtures["id_card.png"]).json()["jobId"]
    wait_done(c, job_id)
    time.sleep(0.01)
    assert c.app.state.worker.sweep() == 1
    assert c.get(f"/v1/jobs/{job_id}", headers=H).status_code == 404


def test_webhook_is_signed(make_client, fixtures):
    received = []

    def hook(request: httpx.Request):
        received.append(request)
        return httpx.Response(200)

    c, s = make_client(transport=httpx.MockTransport(hook), webhook_secret="s3cret",
                       webhook_allowed_hosts="hooks.internal")
    job_id = upload(c, fixtures["id_card.png"], callbackUrl="http://hooks.internal/done").json()["jobId"]
    wait_done(c, job_id)
    end = time.time() + 5
    while not received and time.time() < end:
        time.sleep(0.05)
    req = received[0]
    body = json.loads(req.content)
    assert set(body) == {"event", "deliveryId", "jobId", "status"}  # no document data in the webhook
    assert body["event"] == "job.done" and body["jobId"] == job_id and body["status"] == "done"
    assert req.headers["X-Webhook-Id"] == body["deliveryId"]
    ts = req.headers["X-Webhook-Timestamp"]
    assert req.headers["X-Webhook-Signature"] == webhook.sign("s3cret", ts, req.content)


def test_webhook_retried_on_failure(make_client, fixtures):
    calls = []

    def hook(request):
        calls.append(1)
        return httpx.Response(500 if len(calls) < 3 else 200)

    c, _ = make_client(transport=httpx.MockTransport(hook), webhook_secret="s", webhook_allowed_hosts="hooks.internal")
    wait_done(c, upload(c, fixtures["id_card.png"], callbackUrl="https://hooks.internal/x").json()["jobId"])
    end = time.time() + 5
    while len(calls) < 3 and time.time() < end:
        time.sleep(0.05)
    assert len(calls) == 3


@pytest.mark.parametrize("kw,url", [
    ({}, "http://hooks.internal/x"),  # callbacks disabled by default
    ({"webhook_secret": "s", "webhook_allowed_hosts": "hooks.internal"}, "http://evil.example.com/x"),
    ({"webhook_secret": "s", "webhook_allowed_hosts": "hooks.internal"}, "file:///etc/passwd"),
    ({"webhook_allowed_hosts": "hooks.internal"}, "http://hooks.internal/x"),  # no secret
])
def test_callback_url_rejected(make_client, fixtures, kw, url):
    c, _ = make_client(**kw)
    assert upload(c, fixtures["id_card.png"], callbackUrl=url).status_code == 400


def test_queue_full(make_client, fixtures):
    c, _ = make_client(max_queued_jobs=0)
    assert upload(c, fixtures["id_card.png"]).status_code == 503


def test_health(make_client):
    c, _ = make_client()
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["llm"] == {"reachable": True, "textModel": True, "visionModel": True}


def test_health_degraded_when_llm_unreachable(make_client):
    c, _ = make_client(services=StubServices(models=()))
    r = c.get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded" and r.json()["llm"]["reachable"] is False


def test_logs_never_contain_file_names_or_text(make_client, fixtures, caplog):
    caplog.set_level(logging.DEBUG)
    c, _ = make_client()
    job_id = upload(c, fixtures["id_card.png"], name="SECRET_NAME_ABC.png").json()["jobId"]
    assert wait_done(c, job_id)["file"]["name"] == "SECRET_NAME_ABC.png"
    assert job_id in caplog.text  # job ids are logged ...
    assert "SECRET_NAME_ABC" not in caplog.text and "SECRET_TEXT" not in caplog.text  # ... names/text never


def test_openapi_docs(make_client):
    c, _ = make_client()
    spec = c.get("/openapi.json").json()
    assert {"/v1/documents", "/v1/extract", "/v1/jobs/{job_id}", "/health"} <= set(spec["paths"])
    assert "APIKeyHeader" in spec["components"]["securitySchemes"]
    assert c.get("/docs").status_code == 200


def test_end_to_end_real_pipeline(make_client, fixtures, settings, engine):
    llm = StubLlm(V("not_id", 0.99))  # never called: the passport MRZ decides
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    c, _ = make_client(analyze_fn=analyze, services=svc)
    out = wait_done(c, upload(c, fixtures["passport.png"], name="passport.png").json()["jobId"])
    assert out["status"] == "done" and set(out) == RESULT_KEYS
    assert out["idDetection"]["isId"] and out["idDetection"]["docType"] == "passport"
    assert out["file"] == {"name": "passport.png", "mimeType": "image/png", "pages": 1}
    assert out["pages"][0]["method"] == "ocr" and "PASSPORT" in out["fullText"]


# --- POST /v1/extract (synchronous, text only) -------------------------------------------------------

EXTRACT_KEYS = {"file", "pages", "fullText", "normalizedText", "languages", "timingsMs", "processing"}


def extract(client, data, name="doc.png", headers=H):
    return client.post("/v1/extract", files={"file": (name, data, "application/octet-stream")}, headers=headers)


def test_extract_returns_text_directly_without_id_detection(make_client, fixtures):
    CALLS.clear()
    c, s = make_client()
    r = extract(c, fixtures["id_card.png"], name="card.png")
    assert r.status_code == 200
    out = r.json()
    assert set(out) == EXTRACT_KEYS and out["fullText"] == "SECRET_TEXT" and out["file"]["name"] == "card.png"
    assert CALLS == [{"detect_id": False, "max_pages": s.extract_max_pages}]
    assert not s.upload_dir.exists() or list(s.upload_dir.iterdir()) == []  # processed in memory only
    assert c.get("/health").json()["jobs"] == {"queued": 0, "processing": 0, "done": 0, "failed": 0}  # no job


def test_documents_endpoint_still_detects_ids(make_client, fixtures):
    CALLS.clear()
    c, _ = make_client()
    out = wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    assert "idDetection" in out and CALLS == [{"detect_id": True, "max_pages": None}]


def test_extract_errors(make_client, fixtures):
    c, _ = make_client(max_upload_mb=1)
    assert extract(c, fixtures["id_card.png"], headers={}).status_code == 401
    assert extract(c, fixtures["id_card.png"], headers={"X-Api-Key": "wrong"}).status_code == 401
    assert extract(c, b"GIF89a not supported").status_code == 415
    assert extract(c, b"").status_code == 400
    assert extract(c, fixtures["id_card.png"] + b"\0" * (2 * 1024 * 1024)).status_code == 413


def test_extract_page_limit(make_client, fixtures, settings, engine):
    llm = StubLlm(V("not_id", 0.99))
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    c, _ = make_client(analyze_fn=analyze, services=svc, extract_max_pages=1)
    r = extract(c, fixtures["two_pages.tiff"], name="t.tiff")
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_pages"
    assert "/v1/documents" in r.json()["error"]["message"]


def test_extract_internal_error_hides_details(make_client, fixtures, caplog):
    def boom(data, filename, svc, **kw):
        raise RuntimeError("SECRET_TEXT inside exception")

    caplog.set_level(logging.DEBUG)
    c, _ = make_client(analyze_fn=boom)
    r = extract(c, fixtures["id_card.png"])
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "processing failed",
                                  "requestId": r.headers["X-Request-Id"]}}
    assert "SECRET_TEXT" not in caplog.text and "RuntimeError" in caplog.text


def test_extract_busy_returns_503(make_client, fixtures):
    started, release = threading.Event(), threading.Event()

    def slow(data, filename, svc, **kw):
        started.set()
        release.wait(10)
        return fake_analyze(data, filename, svc, **kw)

    c, _ = make_client(analyze_fn=slow, extract_max_concurrent=1)
    first: list = []
    t = threading.Thread(target=lambda: first.append(extract(c, fixtures["id_card.png"])))
    t.start()
    try:
        assert started.wait(10)
        assert extract(c, fixtures["id_card.png"]).status_code == 503
    finally:
        release.set()
        t.join(10)
    assert first[0].status_code == 200
    assert extract(c, fixtures["id_card.png"]).status_code == 200  # slot released again


def test_extract_logs_never_contain_file_names_or_text(make_client, fixtures, caplog):
    caplog.set_level(logging.DEBUG)
    c, _ = make_client()
    assert extract(c, fixtures["id_card.png"], name="SECRET_NAME_ABC.png").status_code == 200
    assert "extract finished" in caplog.text
    assert "SECRET_NAME_ABC" not in caplog.text and "SECRET_TEXT" not in caplog.text


def test_extract_end_to_end_arabic_english_mixed(make_client, fixtures, settings, engine):
    llm = StubLlm(V("national_id", 0.99))  # must never be called by /v1/extract
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    c, _ = make_client(analyze_fn=analyze, services=svc)

    # PDF: English text layer + scanned Arabic invoice (OCR) + Arabic text layer in visual order
    out = extract(c, fixtures["letter_invoice_arabic.pdf"], name="l.pdf").json()
    assert set(out) == EXTRACT_KEYS and out["file"]["pages"] == 3
    assert [p["method"] for p in out["pages"]] == ["text_layer", "ocr", "text_layer"]
    assert "Thank you for your order" in out["fullText"] and "فاتورةضريبية" in compact(out["fullText"])
    assert set(out["languages"]) == {"ar", "en"}

    # Image: mixed Arabic/English card, OCR only
    out = extract(c, fixtures["id_card.png"]).json()
    assert out["pages"][0]["method"] == "ocr" and out["languages"] == ["ar", "en"]
    assert "بطاقةشخصية" in compact(out["fullText"]) and "بطاقهشخصيه" in compact(out["normalizedText"])
    assert "llm" not in out["timingsMs"] and "rules" not in out["timingsMs"]

    # DOCX: Arabic + English body text and table, plus an embedded image (OCR)
    out = extract(c, fixtures["delivery_note.docx"], name="d.docx").json()
    assert out["pages"][0]["method"] == "text_layer" and "هذه وثيقة تجريبية" in out["fullText"]
    assert "Delivery note" in out["fullText"]

    assert llm.text_calls == 0 and llm.image_calls == 0


# --- Integration contract: errors, request ids, Retry-After, health probes, webhook isolation --------------

def wait_for(cond, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("call,status,code", [
    (lambda c, f: upload(c, f["id_card.png"], headers={}), 401, "unauthorized"),
    (lambda c, f: c.get("/v1/jobs/nope", headers=H), 404, "not_found"),
    (lambda c, f: upload(c, b"GIF89a not supported"), 415, "unsupported_type"),
    (lambda c, f: upload(c, b""), 400, "empty"),
    (lambda c, f: upload(c, f["id_card.png"], callbackUrl="http://evil.example.com/x"), 400, "callback_not_allowed"),
    (lambda c, f: c.post("/v1/documents", headers=H), 422, "invalid_request"),
])
def test_errors_use_one_envelope_with_stable_codes(make_client, fixtures, call, status, code):
    c, _ = make_client()
    r = call(c, fixtures)
    assert r.status_code == status
    assert set(r.json()) == {"error"} and r.json()["error"]["code"] == code
    assert r.json()["error"]["requestId"] == r.headers["X-Request-Id"]


def test_extract_rejected_file_has_its_code(make_client, fixtures):
    def corrupt(data, filename, svc, **kw):
        raise FileRejected("corrupt", "PDF could not be opened (corrupt or encrypted)")

    c, _ = make_client(analyze_fn=corrupt)
    r = extract(c, fixtures["id_card.png"])
    assert r.status_code == 400 and r.json()["error"]["code"] == "corrupt"


def test_validation_error_does_not_echo_values(make_client):
    c, _ = make_client()
    r = c.post("/v1/documents", data={"callbackUrl": "SECRET_VALUE"}, headers=H)
    assert r.status_code == 422 and "SECRET_VALUE" not in r.text and "file" in r.json()["error"]["message"]


def test_request_id_is_echoed_or_generated(make_client, fixtures, caplog):
    caplog.set_level(logging.INFO)
    c, _ = make_client()
    r = upload(c, fixtures["id_card.png"], headers={**H, "X-Request-Id": "caller-123"})
    assert r.headers["X-Request-Id"] == "caller-123"
    wait_done(c, r.json()["jobId"])
    assert "req=caller-123" in caplog.text  # queued + finished lines carry the caller's id
    r = c.get("/health/live", headers={"X-Request-Id": "bad id\nwith newline"})
    assert r.headers["X-Request-Id"] != "bad id\nwith newline" and len(r.headers["X-Request-Id"]) == 32


def test_503s_carry_retry_after(make_client, fixtures):
    c, _ = make_client(max_queued_jobs=0, retry_after_s=7)
    r = upload(c, fixtures["id_card.png"])
    assert r.status_code == 503 and r.json()["error"]["code"] == "queue_full" and r.headers["Retry-After"] == "7"


def test_health_probes_when_llm_is_down(make_client):
    c, _ = make_client(services=StubServices(models=()))
    assert c.get("/health").status_code == 503  # full status: unchanged
    assert c.get("/health/live").json() == {"status": "ok"}
    r = c.get("/health/ready")
    assert r.status_code == 200 and r.json()["status"] == "ready"  # /v1/extract still works without the LLM
    assert r.json()["llm"]["reachable"] is False and r.json()["worker"] == {"alive": True}


def test_slow_webhook_does_not_block_the_queue(make_client, fixtures):
    entered, release = threading.Event(), threading.Event()

    def hook(request):
        entered.set()
        release.wait(10)  # a receiver that hangs
        return httpx.Response(200)

    c, _ = make_client(transport=httpx.MockTransport(hook), webhook_secret="s", webhook_allowed_hosts="hooks.internal")
    try:
        first = upload(c, fixtures["id_card.png"], callbackUrl="http://hooks.internal/x").json()["jobId"]
        assert wait_done(c, first)["webhook"] == {"state": "pending", "attempts": 0}
        assert entered.wait(5)
        second = upload(c, fixtures["id_card.png"]).json()["jobId"]
        assert wait_done(c, second, timeout=5)["status"] == "done"  # processed while the webhook still hangs
    finally:
        release.set()
    assert wait_for(lambda: c.get(f"/v1/jobs/{first}", headers=H).json()["webhook"]["state"] == "delivered")


def test_webhook_failure_recorded_on_job(make_client, fixtures):
    c, _ = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(500)), webhook_secret="s",
                       webhook_allowed_hosts="hooks.internal", webhook_retries=2)
    job_id = upload(c, fixtures["id_card.png"], callbackUrl="http://hooks.internal/x").json()["jobId"]
    wait_done(c, job_id)
    assert wait_for(lambda: c.get(f"/v1/jobs/{job_id}", headers=H).json()["webhook"]["state"] == "failed")
    assert c.get(f"/v1/jobs/{job_id}", headers=H).json()["webhook"] == {"state": "failed", "attempts": 2}


def test_openapi_documents_error_model(make_client):
    spec = make_client()[0].get("/openapi.json").json()
    assert "ErrorResponse" in spec["components"]["schemas"]
    assert "401" in spec["paths"]["/v1/documents"]["post"]["responses"]
    assert {"/health/live", "/health/ready"} <= set(spec["paths"])


# --- Idempotency, caller references, long-poll, metrics ------------------------------------------------------

def test_idempotent_upload_returns_same_job(make_client, fixtures):
    c, _ = make_client()
    h = {**H, "Idempotency-Key": "order-42"}
    r1 = upload(c, fixtures["id_card.png"], headers=h)
    r2 = upload(c, fixtures["id_card.png"], headers=h)
    assert r1.status_code == r2.status_code == 202 and r1.json() == r2.json()
    assert r2.headers["Idempotent-Replayed"] == "true" and "Idempotent-Replayed" not in r1.headers
    r3 = upload(c, fixtures["passport.png"], headers=h)
    assert r3.status_code == 409 and r3.json()["error"]["code"] == "idempotency_conflict"
    r4 = upload(c, fixtures["id_card.png"], headers={"X-Api-Key": OTHER_KEY, "Idempotency-Key": "order-42"})
    assert r4.json()["jobId"] != r1.json()["jobId"]  # keys are scoped per API key
    assert upload(c, fixtures["id_card.png"], headers={**H, "Idempotency-Key": "bad key!"}).status_code == 422


def test_idempotent_replay_allowed_when_queue_full(make_client, fixtures):
    c, _ = make_client()
    h = {**H, "Idempotency-Key": "k1"}
    job_id = upload(c, fixtures["id_card.png"], headers=h).json()["jobId"]
    wait_done(c, job_id)
    c.app.state.store.count_queued = lambda: 10 ** 6  # pretend the queue is full
    assert upload(c, fixtures["id_card.png"], headers=h).json()["jobId"] == job_id
    assert upload(c, fixtures["id_card.png"]).status_code == 503


def test_external_id_and_metadata_echoed(make_client, fixtures):
    received = []
    c, _ = make_client(transport=httpx.MockTransport(lambda r: received.append(r) or httpx.Response(200)),
                       webhook_secret="s", webhook_allowed_hosts="hooks.internal")
    r = upload(c, fixtures["id_card.png"], externalId="case-77", metadata='{"tenant": "a", "n": 1}',
               callbackUrl="http://hooks.internal/x")
    out = wait_done(c, r.json()["jobId"])
    assert out["externalId"] == "case-77" and out["metadata"] == {"tenant": "a", "n": 1}
    assert wait_for(lambda: received)
    body = json.loads(received[0].content)
    assert body["externalId"] == "case-77" and body["metadata"] == {"tenant": "a", "n": 1}


@pytest.mark.parametrize("form", [{"metadata": "[1, 2]"}, {"metadata": "not json"},
                                  {"metadata": '{"a": "' + "x" * 3000 + '"}'}, {"externalId": "x" * 200}])
def test_bad_references_rejected(make_client, fixtures, form):
    c, _ = make_client()
    r = upload(c, fixtures["id_card.png"], **form)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_long_poll_waits_for_result(make_client, fixtures):
    release = threading.Event()

    def slow(data, filename, svc, **kw):
        release.wait(10)
        return fake_analyze(data, filename, svc, **kw)

    c, _ = make_client(analyze_fn=slow, max_wait_s=10)
    job_id = upload(c, fixtures["id_card.png"]).json()["jobId"]
    t0 = time.time()
    assert c.get(f"/v1/jobs/{job_id}?wait=0.3", headers=H).json()["status"] in ("queued", "processing")
    assert 0.25 < time.time() - t0 < 5
    threading.Timer(0.3, release.set).start()
    out = c.get(f"/v1/jobs/{job_id}?wait=10", headers=H).json()
    assert out["status"] == "done"


def test_metrics(make_client, fixtures):
    c, _ = make_client()
    wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    assert extract(c, fixtures["id_card.png"]).status_code == 200
    text = c.get("/metrics").text
    owner = legacy_owner(KEY)
    assert f'ocr_jobs_finished_total{{client="{owner}",status="done"}} 1.0' in text
    assert f'ocr_extract_requests_total{{client="{owner}",status="done"}} 1.0' in text
    assert 'ocr_pages_total{method="ocr"}' in text and 'ocr_jobs{status="done"} 1.0' in text
    assert f'ocr_http_responses_total{{client="{owner}",route="/v1/documents",status="202"}} 1.0' in text
    assert "SECRET" not in text and "doc.png" not in text


def test_metrics_for_queue_health_and_dependencies(make_client, fixtures):
    c, _ = make_client(services=StubServices(models=()))  # the LLM is down
    wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    upload(c, fixtures["id_card.png"], headers={"X-Api-Key": "wrong"})
    text = c.get("/metrics").text
    assert "ocr_job_queue_wait_seconds_count 1.0" in text  # one job taken from the queue
    assert "ocr_queue_oldest_age_seconds 0.0" in text and "ocr_processing_oldest_age_seconds 0.0" in text
    assert "ocr_webhooks_pending 0.0" in text
    assert 'ocr_dependency_up{dependency="store"} 1.0' in text and 'ocr_dependency_up{dependency="worker"} 1.0' in text
    assert 'ocr_dependency_up{dependency="llm"} 0.0' in text
    assert 'ocr_http_responses_total{client="-",route="unmatched",status="401"} 1.0' in text  # no valid key


def test_queue_and_processing_ages_are_measured(make_client, fixtures):
    release = threading.Event()

    def slow(data, filename, svc, **kw):
        release.wait(10)
        return fake_analyze(data, filename, svc, **kw)

    c, _ = make_client(analyze_fn=slow)
    try:
        first = upload(c, fixtures["id_card.png"]).json()["jobId"]
        upload(c, fixtures["id_card.png"])  # waits behind the first (one worker thread)
        assert wait_for(lambda: c.get(f"/v1/jobs/{first}", headers=H).json()["status"] == "processing")
        time.sleep(0.3)
        ages = c.app.state.store.ages()
        assert ages["queued"] >= 0.25 and ages["processing"] >= 0.25
    finally:
        release.set()


def test_llm_fallbacks_are_counted(make_client, fixtures, settings, engine):
    from app.detection.llm import LlmError

    llm = StubLlm(LlmError("unreachable"), LlmError("unreachable"))  # text and vision model both down
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    c, _ = make_client(analyze_fn=analyze, services=svc)
    out = wait_done(c, upload(c, fixtures["invoice.png"]).json()["jobId"])
    assert "llm: unavailable, using rules only" in out["idDetection"]["reasons"]
    assert "vision: unavailable" in out["idDetection"]["reasons"]  # rules alone are unsure: vision was tried
    text = c.get("/metrics").text
    assert 'ocr_llm_fallbacks_total{stage="llm"} 1.0' in text and 'ocr_llm_fallbacks_total{stage="vision"} 1.0' in text


def test_profile_detectors_report_fallbacks_too(settings):
    from app.profiles import Profile

    svc = Services(settings, None, StubLlm(), IdDetector(settings, None))
    seen = []
    svc.report_fallbacks(seen.append)
    later = svc.detector_for(Profile(name="no-vision", vision_enabled=False))  # created after the wiring
    assert svc.detector.on_fallback == later.on_fallback == seen.append


def test_metrics_can_be_disabled(make_client):
    c, _ = make_client(metrics_enabled=False)
    assert c.get("/metrics").status_code == 404


PG_URL = os.environ.get("TEST_DATABASE_URL", "")


@pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set")
def test_postgres_store_end_to_end(make_client, fixtures):
    from tests.test_store import _drop_tables

    _drop_tables()
    c, _ = make_client(job_store="postgres", database_url=PG_URL, db_poll_s=0.05)
    h = {**H, "Idempotency-Key": "pg-1"}
    job_id = upload(c, fixtures["id_card.png"], name="card.png", headers=h).json()["jobId"]
    out = c.get(f"/v1/jobs/{job_id}?wait=10", headers=H).json()
    assert out["status"] == "done" and out["file"]["name"] == "card.png"
    assert upload(c, fixtures["id_card.png"], headers=h).json()["jobId"] == job_id
    assert c.app.state.store.read_upload(job_id) is None  # upload deleted after processing
    assert c.get("/health/ready").json()["store"] == {"backend": "postgres", "reachable": True}

    # a second app on the same database (another replica, or after a restart) sees the job
    c2, _ = make_client(job_store="postgres", database_url=PG_URL)
    assert c2.get(f"/v1/jobs/{job_id}", headers=H).json()["status"] == "done"
    assert c2.delete(f"/v1/jobs/{job_id}", headers=H).status_code == 204
    assert c.get(f"/v1/jobs/{job_id}", headers=H).status_code == 404


@pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set")
def test_postgres_webhook_survives_a_restart(make_client, fixtures):
    """The receiver is down when the job finishes; the service restarts; the new process delivers the callback,
    with the same deliveryId."""
    from tests.test_store import _drop_tables

    _drop_tables()
    down, up = [], []
    pg = dict(job_store="postgres", database_url=PG_URL, db_poll_s=0.05, webhook_secret="s",
              webhook_allowed_hosts="hooks.internal", webhook_backoff_s=600)
    c1, _ = make_client(transport=httpx.MockTransport(lambda r: down.append(r) or httpx.Response(503)), **pg)
    job_id = upload(c1, fixtures["id_card.png"], callbackUrl="http://hooks.internal/x").json()["jobId"]
    assert wait_for(lambda: down)
    c1.__exit__(None, None, None)  # restart: nothing is in memory any more; the retry waits in the database
    c2, _ = make_client(transport=httpx.MockTransport(lambda r: up.append(r) or httpx.Response(200)), **pg)
    c2.app.state.store.update(job_id, webhook_next_at=time.time())  # as if the 10 min backoff had passed
    assert wait_for(lambda: c2.get(f"/v1/jobs/{job_id}", headers=H).json()["webhook"]["state"] == "delivered")
    assert c2.get(f"/v1/jobs/{job_id}", headers=H).json()["webhook"] == {"state": "delivered", "attempts": 2}
    assert up[0].headers["X-Webhook-Id"] == down[0].headers["X-Webhook-Id"]  # receivers can de-duplicate


# --- Phase C: includeBoxes option, staged OCR/LLM worker ------------------------------------------------------

def test_include_boxes_option_reaches_the_pipeline(make_client, fixtures):
    CALLS.clear()
    c, _ = make_client()
    assert c.post("/v1/extract", files={"file": ("a.png", fixtures["id_card.png"])}, data={"includeBoxes": "true"},
                  headers=H).status_code == 200
    wait_done(c, upload(c, fixtures["id_card.png"], includeBoxes="true").json()["jobId"])
    wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    assert [x.get("include_boxes") for x in CALLS] == [True, True, None]


class StagedFake:
    """analyze_fn with .stages, like app.pipeline.analyze; detect blocks until released."""

    def __init__(self):
        self.extracted: list[str] = []
        self.release = threading.Event()
        self.stages = (self.extract_stage, self.detect_stage)

    def __call__(self, data, filename, svc, **kw):  # single-stage mode (unused when staged)
        raise AssertionError("staged mode must use the stages")

    def extract_stage(self, data, filename, svc, **kw):
        self.extracted.append(filename)
        return filename

    def detect_stage(self, extracted, svc):
        assert self.release.wait(10)
        return fake_analyze(b"", extracted, svc)


def test_staged_worker_overlaps_ocr_and_llm(make_client, fixtures):
    fake = StagedFake()
    c, _ = make_client(analyze_fn=fake, llm_worker_threads=1)
    first = upload(c, fixtures["id_card.png"], name="one.png").json()["jobId"]
    second = upload(c, fixtures["id_card.png"], name="two.png").json()["jobId"]
    # job 1 is in the LLM stage (blocked) while job 2's extraction already ran
    assert wait_for(lambda: fake.extracted == ["one.png", "two.png"])
    assert c.get(f"/v1/jobs/{first}", headers=H).json()["status"] == "processing"
    fake.release.set()
    out1, out2 = wait_done(c, first), wait_done(c, second)
    assert out1["status"] == out2["status"] == "done" and out2["file"]["name"] == "two.png"
    assert "llmQueue" in out2["timingsMs"]


def test_staged_worker_reports_extract_errors(make_client, fixtures):
    fake = StagedFake()

    def bad(data, filename, svc, **kw):
        raise FileRejected("corrupt", "PDF could not be opened (corrupt or encrypted)")

    fake.stages = (bad, fake.detect_stage)
    c, _ = make_client(analyze_fn=fake, llm_worker_threads=1)
    out = wait_done(c, upload(c, fixtures["id_card.png"]).json()["jobId"])
    assert out["status"] == "failed" and out["error"]["code"] == "corrupt"


# --- Uploads stay off disk; oversized streamed uploads; production startup guard ----------------------------

def multipart(*files: tuple[str, bytes], boundary: str = "TESTBOUNDARY7f3a") -> tuple[bytes, str]:
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="big.png"\r\n'
                    "Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n" for name, data in files)
    return body + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def chunked(body: bytes, size: int = 256 * 1024):
    """A generator body: httpx sends it without Content-Length (Transfer-Encoding: chunked)."""
    for i in range(0, len(body), size):
        yield body[i:i + size]


@pytest.mark.parametrize("path", ["/v1/documents", "/v1/extract"])
def test_oversized_chunked_request_is_413_without_content_length(make_client, fixtures, path):
    """Each file is under MAX_UPLOAD_MB, but the request isn't: only the streamed body limit can stop it."""
    c, _ = make_client(max_upload_mb=1)
    part = fixtures["id_card.png"] + b"\0" * (600 * 1024)
    body, ctype = multipart(("file", part), ("extra", part))
    r = c.post(path, content=chunked(body), headers={**H, "Content-Type": ctype})
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "too_large" and r.json()["error"]["requestId"] == r.headers["X-Request-Id"]
    assert c.get("/health").json()["jobs"]["queued"] == 0


def test_chunked_upload_within_the_limit_works(make_client, fixtures):
    c, _ = make_client()
    body, ctype = multipart(("file", fixtures["id_card.png"]))
    r = c.post("/v1/extract", content=chunked(body, 1024), headers={**H, "Content-Type": ctype})
    assert r.status_code == 200 and r.json()["file"]["name"] == "big.png"


def test_uploads_over_1mb_are_not_spooled_to_disk(make_client, fixtures, monkeypatch):
    import tempfile

    from starlette.formparsers import MultiPartParser

    monkeypatch.setattr(MultiPartParser, "spool_max_size", 1024 * 1024)  # Starlette's default: 1 MB
    rolled = []
    real_rollover = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover",
                        lambda self: rolled.append(1) or real_rollover(self))  # rollover = written to a disk file
    c, _ = make_client(max_upload_mb=5)
    big = fixtures["id_card.png"] + b"\0" * (3 * 1024 * 1024)
    assert extract(c, big).status_code == 200
    assert wait_done(c, upload(c, big).json()["jobId"])["status"] == "done"
    assert rolled == []


def test_production_refuses_an_unsafe_configuration(tmp_path):
    from app.config import ConfigError

    with pytest.raises(ConfigError, match="JOB_STORE must be postgres"):
        create_app(make_settings(tmp_path, environment="production"), services=StubServices())


# --- Client registry (CLIENTS_FILE): rotation, isolation, scopes, per-client webhooks and limits -------------

SECRET_CIVIL, SECRET_OPS = "civil-" + "s" * 40, "ops-" + "t" * 40
CIVIL_1, CIVIL_2 = "civil-key-1-" + "a" * 32, "civil-key-2-" + "b" * 32
ARCHIVE, OPS = "archive-key-" + "c" * 32, "ops-key-" + "d" * 32


def k(key: str) -> dict:
    return {"X-Api-Key": key}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Write a CLIENTS_FILE and return the settings for it. Keyword arguments update single clients."""
    import yaml

    from app.api.clients import key_hash

    monkeypatch.setenv("OCR_TEST_WEBHOOK_CIVIL", SECRET_CIVIL)
    monkeypatch.setenv("OCR_TEST_WEBHOOK_OPS", SECRET_OPS)

    def _write(profiles: dict | None = None, **changes) -> dict:
        clients = {
            "civil-registry": {"name": "Civil", "key_sha256": [key_hash(CIVIL_1), key_hash(CIVIL_2)],
                               "webhook": {"secret_env": "OCR_TEST_WEBHOOK_CIVIL",
                                           "allowed_hosts": ["hooks.civil.internal"]}},
            "archive": {"name": "Archive", "key_sha256": [key_hash(ARCHIVE)], "scopes": ["extract"]},
            "ops": {"name": "Ops", "key_sha256": [key_hash(OPS)], "scopes": ["documents"],
                    "webhook": {"secret_env": "OCR_TEST_WEBHOOK_OPS", "allowed_hosts": ["hooks.ops.internal"]}},
        }
        for client_id, change in changes.items():
            clients[client_id].update(change)
        path = tmp_path / "clients.yaml"
        path.write_text(yaml.safe_dump({"profiles": profiles or {}, "clients": clients}), encoding="utf-8")
        return {"clients_file": str(path), "api_keys": ""}

    return _write


def test_registry_key_rotation_keeps_jobs_and_isolates_clients(make_client, fixtures, registry):
    c, _ = make_client(**registry())
    job_id = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1)).json()["jobId"]
    assert wait_done(c, job_id, headers=k(CIVIL_2))["status"] == "done"  # the rotated key sees the same jobs
    assert c.get(f"/v1/jobs/{job_id}", headers=k(OPS)).status_code == 404  # another client does not
    assert c.delete(f"/v1/jobs/{job_id}", headers=k(OPS)).status_code == 404
    assert c.delete(f"/v1/jobs/{job_id}", headers=k(CIVIL_1)).status_code == 204
    assert upload(c, fixtures["id_card.png"], headers=k("key-a")).status_code == 401  # API_KEYS is not used


def test_registry_disabled_client_gets_401(make_client, fixtures, registry, caplog):
    caplog.set_level(logging.INFO)
    c, _ = make_client(**registry(**{"civil-registry": {"enabled": False}}))
    r = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1))
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert extract(c, fixtures["id_card.png"], headers=k(CIVIL_2)).status_code == 401
    assert "auth rejected reason=disabled client=civil-registry" in caplog.text
    assert extract(c, fixtures["id_card.png"], headers=k(ARCHIVE)).status_code == 200  # the others still work


@pytest.mark.parametrize("key,path,status", [
    (ARCHIVE, "/v1/documents", 403), (ARCHIVE, "/v1/extract", 200),
    (OPS, "/v1/extract", 403), (OPS, "/v1/documents", 202),
    (CIVIL_1, "/v1/documents", 202), (CIVIL_1, "/v1/extract", 200),
])
def test_registry_scopes(make_client, fixtures, registry, key, path, status):
    c, _ = make_client(**registry())
    r = c.post(path, files={"file": ("a.png", fixtures["id_card.png"])}, headers=k(key))
    assert r.status_code == status
    if status == 403:
        assert r.json()["error"]["code"] == "forbidden" and r.json()["error"]["requestId"] == r.headers["X-Request-Id"]


def test_registry_webhooks_use_the_clients_own_hosts_and_secret(make_client, fixtures, registry):
    received = []
    c, _ = make_client(**registry(), transport=httpx.MockTransport(lambda r: received.append(r) or httpx.Response(200)))
    # another client's callback host, and a client without webhooks: rejected
    r = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1), callbackUrl="http://hooks.ops.internal/x")
    assert r.status_code == 400 and r.json()["error"]["code"] == "callback_not_allowed"
    assert upload(c, fixtures["id_card.png"], headers=k(OPS), callbackUrl="http://hooks.civil.internal/x"
                  ).status_code == 400
    # its own host: delivered, and signed with its own secret only
    job_id = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1), callbackUrl="http://hooks.civil.internal/x"
                    ).json()["jobId"]
    assert wait_for(lambda: received)
    req = received[0]
    ts = req.headers["X-Webhook-Timestamp"]
    assert json.loads(req.content)["jobId"] == job_id
    assert req.headers["X-Webhook-Signature"] == webhook.sign(SECRET_CIVIL, ts, req.content)
    assert req.headers["X-Webhook-Signature"] != webhook.sign(SECRET_OPS, ts, req.content)


def test_registry_queue_quota_is_per_client(make_client, fixtures, registry):
    c, _ = make_client(**registry(**{"civil-registry": {"limits": {"max_queued_jobs": 0}}}))
    r = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1))
    assert r.status_code == 503 and r.json()["error"]["code"] == "queue_full" and "Retry-After" in r.headers
    assert upload(c, fixtures["id_card.png"], headers=k(OPS)).status_code == 202  # other clients are not blocked


def test_registry_extract_concurrency_is_per_client(make_client, fixtures, registry):
    started, release = threading.Event(), threading.Event()

    def slow_for_one_file(data, filename, svc, **kw):
        if filename == "slow.png":
            started.set()
            release.wait(10)
        return fake_analyze(data, filename, svc, **kw)

    c, _ = make_client(**registry(**{"civil-registry": {"limits": {"extract_concurrent": 1}}}),
                       analyze_fn=slow_for_one_file, extract_max_concurrent=3)
    first: list = []
    t = threading.Thread(target=lambda: first.append(extract(c, fixtures["id_card.png"], "slow.png", k(CIVIL_1))))
    t.start()
    try:
        assert started.wait(10)
        r = extract(c, fixtures["id_card.png"], headers=k(CIVIL_2))
        assert r.status_code == 503 and r.json()["error"]["code"] == "busy"  # civil's single slot is taken
        assert extract(c, fixtures["id_card.png"], headers=k(ARCHIVE)).status_code == 200  # archive is not limited
    finally:
        release.set()
        t.join(10)
    assert first[0].status_code == 200
    assert extract(c, fixtures["id_card.png"], headers=k(CIVIL_1)).status_code == 200  # slot released again


def test_registry_logs_name_the_client_never_the_key(make_client, fixtures, registry, caplog):
    caplog.set_level(logging.DEBUG)
    c, _ = make_client(**registry())
    wait_done(c, upload(c, fixtures["id_card.png"], headers=k(CIVIL_1)).json()["jobId"], headers=k(CIVIL_1))
    assert extract(c, fixtures["id_card.png"], headers=k(ARCHIVE)).status_code == 200
    assert upload(c, fixtures["id_card.png"], headers=k("not-a-key")).status_code == 401
    assert "job queued" in caplog.text and "client=civil-registry" in caplog.text
    assert "extract finished client=archive" in caplog.text and "auth rejected reason=unknown_key" in caplog.text
    for secret in (CIVIL_1, ARCHIVE, "not-a-key", SECRET_CIVIL):
        assert secret not in caplog.text


def test_webhook_not_sent_when_the_client_lost_access(tmp_path):
    """The registry is checked again at delivery: a removed or disabled client, or a removed host, gets nothing."""
    from app.api.clients import Client, ClientRegistry

    sent = []
    hosts = frozenset({"h.internal"})
    reg = ClientRegistry([Client("on", "On", webhook_secret="s" * 32, webhook_hosts=hosts),
                          Client("off", "Off", enabled=False, webhook_secret="s" * 32, webhook_hosts=hosts)],
                         {}, "file")
    store, dispatcher = outbox(tmp_path, reg, lambda r: sent.append(r) or httpx.Response(200))
    for owner, url in (("gone", "http://h.internal/x"), ("off", "http://h.internal/x"), ("on", "http://other/x")):
        job = due_delivery(store, owner, url)
        dispatcher.deliver(job)
        assert store.get(job.id).webhook == {"state": "failed", "attempts": 0}
        assert store.get(job.id).webhook_next_at is None  # settled: never retried
        [ev] = [e for e in store.audit_events(job_id=job.id) if e["event"] == "webhook.failed"]
        assert ev["data"] == {"attempts": 0, "reason": "not_allowed"}
    assert sent == []
    job = due_delivery(store, "on", "http://h.internal/x")
    dispatcher.deliver(job)
    assert len(sent) == 1 and store.get(job.id).webhook == {"state": "delivered", "attempts": 1}
    assert sent[0].headers["X-Webhook-Signature"] == webhook.sign("s" * 32, sent[0].headers["X-Webhook-Timestamp"],
                                                                  sent[0].content)  # signed with its own secret


# --- Webhook outbox: retries with backoff, give-up, restart-safe delivery ids, graceful release ------------------

def outbox(tmp_path, reg=None, handler=None, **settings_kw):
    """A memory store and a Dispatcher (not started) that sends through `handler`."""
    from app.api.clients import Client, ClientRegistry
    from app.api.store import InMemoryJobStore

    store = InMemoryJobStore(tmp_path / "outbox")
    store.open()
    reg = reg or ClientRegistry([Client("c1", "C1", webhook_secret="s" * 32,
                                        webhook_hosts=frozenset({"h.internal"}))], {}, "file")
    http = httpx.Client(transport=httpx.MockTransport(handler or (lambda r: httpx.Response(200))))
    return store, webhook.Dispatcher(store, http, reg, make_settings(tmp_path, **settings_kw))


def due_delivery(store, owner="c1", url="http://h.internal/x", finished_ago: float = 0):
    """A finished job whose webhook is due, as the worker leaves it."""
    import uuid

    from app.api.store import Job

    job = store.add(Job(id=uuid.uuid4().hex, owner=owner, filename="", size=1, callback_url=url), b"x")
    now = time.time()
    store.update(job.id, status="done", finished_at=now - finished_ago, webhook={"state": "pending", "attempts": 0},
                 webhook_delivery_id=uuid.uuid4().hex, webhook_next_at=now)
    [claimed] = [j for j in store.claim_webhooks(10, 60) if j.id == job.id]
    return claimed


def test_backoff_doubles_up_to_the_cap(tmp_path):
    s = make_settings(tmp_path, webhook_backoff_s=2, webhook_max_backoff_s=60)
    assert [webhook.backoff(s, n) for n in range(1, 8)] == [2, 4, 8, 16, 32, 60, 60]


def test_failed_attempt_is_rescheduled_with_the_same_delivery_id(tmp_path):
    store, dispatcher = outbox(tmp_path, handler=lambda r: httpx.Response(503), webhook_backoff_s=30)
    job = due_delivery(store)
    t0 = time.time()
    dispatcher.deliver(job)
    after = store.get(job.id)
    assert after.webhook == {"state": "pending", "attempts": 1} and after.webhook_settled_at is None
    assert 29 < after.webhook_next_at - t0 < 32  # WEBHOOK_BACKOFF_S later
    assert after.webhook_delivery_id == job.webhook_delivery_id  # receivers de-duplicate on it
    assert store.claim_webhooks(10, 60) == []  # not due yet
    assert store.count_pending_webhooks() == 1


def test_delivery_gives_up_after_the_retry_window_or_attempts(tmp_path):
    store, dispatcher = outbox(tmp_path, handler=lambda r: httpx.Response(500), webhook_backoff_s=60,
                               webhook_retry_window_s=3600)
    late = due_delivery(store, finished_ago=3590)  # the next attempt (60 s) would be past the 1 h window
    dispatcher.deliver(late)
    assert store.get(late.id).webhook == {"state": "failed", "attempts": 1}
    assert store.get(late.id).webhook_next_at is None and store.get(late.id).webhook_settled_at is not None
    [ev] = [e for e in store.audit_events(job_id=late.id) if e["event"] == "webhook.failed"]
    assert ev["data"] == {"attempts": 1, "reason": "gave_up"}

    store, dispatcher = outbox(tmp_path, handler=lambda r: httpx.Response(500), webhook_backoff_s=0, webhook_retries=2)
    job = due_delivery(store)
    dispatcher.deliver(job)
    dispatcher.deliver(store.claim_webhooks(1, 60)[0])
    assert store.get(job.id).webhook == {"state": "failed", "attempts": 2}


def test_client_retry_window_overrides_the_global_one(tmp_path):
    from app.api.clients import Client, ClientRegistry

    reg = ClientRegistry([Client("c1", "C1", webhook_secret="s" * 32, webhook_hosts=frozenset({"h.internal"}),
                                 webhook_retry_window_s=0)], {}, "file")
    store, dispatcher = outbox(tmp_path, reg, lambda r: httpx.Response(500), webhook_retry_window_s=86400)
    job = due_delivery(store)
    dispatcher.deliver(job)
    assert store.get(job.id).webhook == {"state": "failed", "attempts": 1}  # its window is 0: no retries


def test_a_job_with_a_pending_webhook_is_not_purged(make_client, fixtures):
    c, _ = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(503)), webhook_secret="s",
                       webhook_allowed_hosts="hooks.internal", webhook_backoff_s=3600, result_ttl_s=0)
    job_id = upload(c, fixtures["id_card.png"], callbackUrl="http://hooks.internal/x").json()["jobId"]
    assert wait_for(lambda: c.get(f"/v1/jobs/{job_id}", headers=H).json().get("webhook", {}).get("attempts") == 1)
    time.sleep(0.01)
    assert c.app.state.worker.sweep() == 0  # TTL passed, but the receiver hasn't been told yet
    assert c.get(f"/v1/jobs/{job_id}", headers=H).json()["status"] == "done"
    assert c.get("/health/ready").json()["webhooksPending"] == 1


def test_result_stays_ttl_after_the_webhook_settled(tmp_path):
    store, dispatcher = outbox(tmp_path)
    job = due_delivery(store, finished_ago=7200)  # finished 2 h ago, webhook delivered only now
    dispatcher.deliver(job)
    assert store.get(job.id).webhook["state"] == "delivered"
    assert store.purge_finished_before(time.time() - 3600) == 0  # the receiver gets the full TTL to fetch it
    assert store.purge_finished_before(time.time() + 1) == 1


def test_worker_stop_releases_unfinished_jobs(make_client, fixtures):
    release = threading.Event()

    def slow(data, filename, svc, **kw):
        release.wait(10)
        return fake_analyze(data, filename, svc, **kw)

    c, _ = make_client(analyze_fn=slow)
    worker, store = c.app.state.worker, c.app.state.store
    released = []
    store.release = lambda worker_id: released.append(worker_id) or 1
    job_id = upload(c, fixtures["id_card.png"]).json()["jobId"]
    assert wait_for(lambda: c.get(f"/v1/jobs/{job_id}", headers=H).json()["status"] == "processing")
    worker.stop(timeout=0.2)  # the job is still running: it goes back to the queue
    assert released == [worker.worker_id]
    release.set()


# --- Audit trail ------------------------------------------------------------------------------------------------

def events_of(c, **filters) -> list[dict]:
    return c.app.state.store.audit_events(**filters)


def legacy_owner(key: str) -> str:
    from app.api.clients import key_hash

    return key_hash(key)[:16]


def test_audit_trail_of_a_job(make_client, fixtures):
    import hashlib

    c, _ = make_client()
    job_id = upload(c, fixtures["id_card.png"], headers={**H, "X-Request-Id": "trace-1"},
                    externalId="case-77").json()["jobId"]
    wait_done(c, job_id)
    assert c.delete(f"/v1/jobs/{job_id}", headers={**H, "X-Request-Id": "trace-2"}).status_code == 204
    events = events_of(c, job_id=job_id)
    names = [e["event"] for e in events]
    assert names[:3] == ["job.received", "job.started", "job.done"] and names[-2:] == ["job.fetched", "job.deleted"]
    assert all(e["clientId"] == legacy_owner(KEY) for e in events)
    received, started, done, fetched, deleted = events[0], events[1], events[2], events[-2], events[-1]
    assert received["requestId"] == started["requestId"] == done["requestId"] == "trace-1"  # the upload's request
    assert received["data"] == {
        "endpoint": "documents", "size": len(fixtures["id_card.png"]),
        "sha256": hashlib.sha256(fixtures["id_card.png"]).hexdigest(), "mimeType": "image/png", "callback": False,
        "includeBoxes": False, "idempotent": False, "externalIdHash": c.app.state.ref_hash("case-77"),
        "profile": "default"}
    assert started["data"]["attempt"] == 1
    assert done["data"]["pages"] == 1 and done["data"]["pageMethods"] == {"ocr": 1}
    assert done["data"]["idDetection"] == {"isId": False, "docType": "not_id", "confidence": 0.9,
                                           "decidedBy": ["rules"], "mrzFound": False, "mrzValid": False}
    assert done["data"]["processing"] == {"version": "test"} and done["data"]["ms"] >= 0
    assert fetched["data"] == {"status": "done"}
    assert deleted["requestId"] == "trace-2" and deleted["data"] == {"by": "client"}


def test_audit_trail_of_failures_replays_and_extract(make_client, fixtures):
    def corrupt(data, filename, svc, **kw):
        raise FileRejected("corrupt", "PDF could not be opened (corrupt or encrypted)")

    c, _ = make_client(analyze_fn=corrupt)
    h = {**H, "Idempotency-Key": "k-1"}
    job_id = upload(c, fixtures["id_card.png"], headers=h).json()["jobId"]
    wait_done(c, job_id)
    assert upload(c, fixtures["id_card.png"], headers={**h, "X-Request-Id": "again"}).json()["jobId"] == job_id
    names = [e["event"] for e in events_of(c, job_id=job_id)]
    assert names[:3] == ["job.received", "job.started", "job.failed"] and "job.replayed" in names
    failed = events_of(c, job_id=job_id)[2]
    assert failed["data"]["error"] == "corrupt" and "pages" not in failed["data"]
    assert events_of(c, request_id="again")[0]["event"] == "job.replayed"

    c2, _ = make_client()
    assert extract(c2, fixtures["id_card.png"], headers={**H, "X-Request-Id": "ex-1"}).status_code == 200
    [ev] = events_of(c2, request_id="ex-1")
    assert ev["event"] == "extract.done" and ev["jobId"] is None and ev["clientId"] == legacy_owner(KEY)
    assert ev["data"]["endpoint"] == "extract" and ev["data"]["mimeType"] == "image/png" and ev["data"]["pages"] == 1
    assert "idDetection" not in ev["data"] and ev["data"]["processing"] == {"version": "test"}


def test_audit_trail_records_rejected_requests(make_client, fixtures):
    c, _ = make_client()
    upload(c, fixtures["id_card.png"], headers={"X-Api-Key": "wrong", "X-Request-Id": "bad-key"})
    upload(c, b"GIF89a not supported", headers={**H, "X-Request-Id": "bad-type"})
    c.get("/v1/jobs/" + "f" * 32, headers={**H, "X-Request-Id": "probe"})
    c.post("/v1/documents", headers={**H, "X-Request-Id": "no-file"})
    c.get("/health/live", headers={"X-Request-Id": "not-api"})

    def only(rid):
        [ev] = events_of(c, request_id=rid)
        assert ev["event"] == "request.error"
        return ev

    assert only("bad-key")["clientId"] is None
    assert only("bad-key")["data"] == {"method": "POST", "route": "/v1/documents", "status": 401,
                                       "code": "unauthorized"}
    assert only("bad-type")["clientId"] == legacy_owner(KEY)
    assert only("bad-type")["data"] == {"method": "POST", "route": "/v1/documents", "status": 415,
                                        "code": "unsupported_type"}
    assert only("probe")["jobId"] == "f" * 32  # someone asking for a job that isn't theirs (or doesn't exist)
    assert only("probe")["data"] == {"method": "GET", "route": "/v1/jobs/{job_id}", "status": 404,
                                     "code": "not_found"}
    assert only("no-file")["data"]["status"] == 422 and only("no-file")["data"]["code"] == "invalid_request"
    assert events_of(c, request_id="not-api") == []

    def boom(data, filename, svc, **kw):
        raise RuntimeError("SECRET_TEXT inside exception")

    c2, _ = make_client(analyze_fn=boom)
    assert extract(c2, fixtures["id_card.png"], headers={**H, "X-Request-Id": "boom"}).status_code == 500
    [ev] = events_of(c2, request_id="boom")
    assert ev["data"]["status"] == 500 and ev["data"]["code"] == "internal_error"


def test_audit_trail_records_webhook_outcomes(make_client, fixtures):
    c, _ = make_client(transport=httpx.MockTransport(lambda r: httpx.Response(200)), webhook_secret="s",
                       webhook_allowed_hosts="hooks.internal")
    job_id = upload(c, fixtures["id_card.png"], callbackUrl="http://hooks.internal/x").json()["jobId"]
    assert wait_for(lambda: any(e["event"] == "webhook.delivered" for e in events_of(c, job_id=job_id)))
    [ev] = [e for e in events_of(c, job_id=job_id) if e["event"] == "webhook.delivered"]
    assert ev["data"] == {"attempts": 1} and ev["clientId"] == legacy_owner(KEY)
    assert next(e for e in events_of(c, job_id=job_id) if e["event"] == "job.received")["data"]["callback"] is True


def test_audit_trail_holds_no_document_content(make_client, fixtures):
    def with_personal_data(data, filename, svc, **kw):
        out = fake_analyze(data, filename, svc, **kw)
        if "idDetection" in out:  # LLM reasons can quote the document; MRZ fields are personal data
            out["idDetection"].update(reasons=["llm: holder SECRET_REASON"],
                                      mrz={"format": "TD3", "surname": "SECRET_SURNAME", "documentNumber": "SECRET_NO",
                                           "valid": True, "checks": {"documentNumber": True}})
        return out

    c, _ = make_client(analyze_fn=with_personal_data)
    job_id = upload(c, fixtures["id_card.png"], name="SECRET_NAME_ABC.png", externalId="SECRET_REF_77",
                    metadata='{"note": "SECRET_META"}', headers={**H, "Idempotency-Key": "SECRET_IDEM"}).json()["jobId"]
    wait_done(c, job_id)
    assert extract(c, fixtures["id_card.png"], name="SECRET_NAME_ABC.png").status_code == 200
    done = next(e for e in events_of(c, job_id=job_id) if e["event"] == "job.done")
    assert done["data"]["idDetection"]["mrzFound"] is True and done["data"]["idDetection"]["mrzValid"] is True
    dump = json.dumps(events_of(c))
    for secret in ("SECRET_NAME", "SECRET_TEXT", "SECRET_REF", "SECRET_META", "SECRET_IDEM", "SECRET_REASON",
                   "SECRET_SURNAME", "SECRET_NO", KEY):
        assert secret not in dump


def test_audit_write_failure_never_fails_the_request(make_client, fixtures, caplog):
    c, _ = make_client()

    def broken(event):
        raise ConnectionError("database down")

    c.app.state.store.record = broken
    caplog.set_level(logging.ERROR)
    assert extract(c, fixtures["id_card.png"]).status_code == 200  # extract.done could not be written
    assert upload(c, b"GIF89a not supported").status_code == 415  # request.error could not be written
    assert "audit write failed event=extract.done" in caplog.text
    assert "ocr_audit_write_failures_total 2.0" in c.get("/metrics").text


def test_sweep_purges_old_audit_events(make_client):
    c, s = make_client(audit_retention_days=30)
    cutoffs = []
    c.app.state.store.purge_events_before = lambda cutoff: cutoffs.append(cutoff) or 0
    c.app.state.worker.sweep()
    assert len(cutoffs) == 1 and abs(cutoffs[0] - (time.time() - 30 * 86400)) < 60
    c2, _ = make_client(audit_retention_days=0)  # keep forever
    c2.app.state.store.purge_events_before = lambda cutoff: cutoffs.append(cutoff) or 0
    c2.app.state.worker.sweep()
    assert len(cutoffs) == 1


def test_results_carry_their_provenance(make_client, fixtures, settings, engine):
    import hashlib
    from pathlib import Path

    from app import __version__

    llm = StubLlm(V("not_id", 0.95, "tax invoice"))
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    c, _ = make_client(analyze_fn=analyze, services=svc)
    out = wait_done(c, upload(c, fixtures["invoice.png"]).json()["jobId"])
    p = out["processing"]
    assert p["version"] == __version__ == c.get("/openapi.json").json()["info"]["version"]
    assert p["ocr"] == {"det": settings.ocr_det_model, "recAr": settings.ocr_rec_model_ar,
                        "recEn": settings.ocr_rec_model_en, "preprocess": settings.ocr_preprocess,
                        "dpi": settings.ocr_dpi}
    assert p["rules"] == hashlib.sha256(Path(settings.id_patterns_path).read_bytes()).hexdigest()[:12]
    assert p["llm"] == settings.llm_text_model and "vision" not in p  # vision wasn't needed
    text_only = extract(c, fixtures["invoice.png"]).json()["processing"]
    assert set(text_only) == {"version", "profile", "ocr"} and text_only["profile"] == "default"  # no rules/LLM


# --- Typed API contract: the models describe the JSON exactly; the OpenAPI document is a reviewed snapshot ------

def recording(fn):
    """Wrap analyze: keep (a JSON round trip of) every result it returns."""
    def wrapped(*args, **kw):
        out = fn(*args, **kw)
        wrapped.seen.append(json.loads(json.dumps(out)))
        return out
    wrapped.seen = []
    return wrapped


def test_typed_models_return_the_pipeline_json_unchanged(make_client, fixtures, settings, engine):
    """No field may be dropped, renamed, nulled or added by the response models (nested nulls included)."""
    llm = StubLlm(V("not_id", 0.95, "tax invoice"))
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    rec = recording(analyze)
    c, _ = make_client(analyze_fn=rec, services=svc)

    job = wait_done(c, upload(c, fixtures["passport.png"], externalId="ref-1").json()["jobId"])
    assert "mrz" in job["idDetection"] and "namesSource" in job["idDetection"]["mrz"]
    assert {k: v for k, v in job.items() if k not in ("jobId", "status", "externalId")} == rec.seen[-1]

    job = wait_done(c, upload(c, fixtures["invoice.png"], includeBoxes="true").json()["jobId"])
    assert "mrz" not in job["idDetection"] and job["pages"][0]["lines"]
    assert {k: v for k, v in job.items() if k not in ("jobId", "status")} == rec.seen[-1]

    r = c.post("/v1/extract", files={"file": ("l.pdf", fixtures["letter_invoice_arabic.pdf"])},
               data={"includeBoxes": "true"}, headers=H)
    out = r.json()
    assert out["pages"][0]["confidence"] is None and "lines" not in out["pages"][0]  # text layer: null, no lines
    assert out == rec.seen[-1]


def test_openapi_result_fields_are_typed(make_client):
    spec = make_client()[0].get("/openapi.json").json()
    schemas = spec["components"]["schemas"]
    job = schemas["JobView"]["properties"]
    assert job["status"]["enum"] == ["queued", "processing", "done", "failed"]
    page = schemas["Page"]["properties"]
    assert page["method"]["enum"] == ["text_layer", "ocr"] and "Line" in json.dumps(page["lines"])
    assert "Mrz" in json.dumps(schemas["IdDetectionResult"]["properties"]["mrz"])
    assert set(schemas["IdDetectionResult"]["properties"]["docType"]["enum"]) == {
        "national_id", "passport", "driver_license", "residence_permit", "other_id", "not_id"}
    for name in ("JobView", "ExtractResult"):  # no untyped result objects left
        for field_name, prop in schemas[name]["properties"].items():
            if field_name not in ("metadata", "timingsMs"):  # caller JSON / open set of stage names
                assert prop.get("type") != "object" or "properties" in prop, (name, field_name)
            assert "additionalProperties" not in prop or prop["additionalProperties"] is not False


SNAPSHOT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"


def test_openapi_matches_the_published_snapshot(make_client):
    """docs/openapi.json is the contract client teams generate from. Any API change must update it on purpose:
    `uv run python scripts/export_openapi.py`, then review the diff (only additive changes in /v1)."""
    from scripts.export_openapi import render

    current = render(make_client()[0].app)
    assert SNAPSHOT.read_text(encoding="utf-8") == current, (
        "The API changed: run `uv run python scripts/export_openapi.py` and check the diff of docs/openapi.json "
        "(see docs/INTEGRATION.md, API versioning and compatibility)")


# --- Processing profiles over the API ----------------------------------------------------------------------------

PROFILES = {
    "archive-text": {"detect_id": False, "include_boxes": True, "result_ttl_s": 0},
    "id-only": {"return_text": False, "return_mrz": False, "result_ttl_s": 86400},
}


def with_profiles(registry):
    return registry(profiles=PROFILES, **{"civil-registry": {"profiles": ["default", "archive-text", "id-only"]},
                                         "archive": {"profiles": ["archive-text"]},
                                         "ops": {"profiles": ["id-only"]}})


def test_profile_is_chosen_per_request_and_copied_into_the_job(make_client, fixtures, registry):
    CALLS.clear()
    c, _ = make_client(**with_profiles(registry))
    wait_done(c, upload(c, fixtures["id_card.png"], headers=k(CIVIL_1)).json()["jobId"], headers=k(CIVIL_1))
    job_id = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1), profile="archive-text").json()["jobId"]
    wait_done(c, job_id, headers=k(CIVIL_1))
    assert CALLS[0] == {"detect_id": True, "max_pages": None}  # the default profile: nothing extra
    assert CALLS[1]["profile"] == {"name": "archive-text", "detect_id": False, "include_boxes": True,
                                   "result_ttl_s": 0} and CALLS[1]["include_boxes"] is True  # profile default
    received = next(e for e in events_of(c, job_id=job_id) if e["event"] == "job.received")
    assert received["data"]["profile"] == "archive-text" and received["data"]["includeBoxes"] is True
    upload(c, fixtures["id_card.png"], headers=k(CIVIL_1), profile="archive-text", includeBoxes="false")
    assert wait_for(lambda: len(CALLS) == 3) and "include_boxes" not in CALLS[2]  # the request field wins


def test_a_profile_the_client_may_not_use_is_403(make_client, fixtures, registry):
    c, _ = make_client(**with_profiles(registry))
    for name in ("id-only", "no-such-profile"):  # another client's profile, an unknown one: the same answer
        r = c.post("/v1/extract", files={"file": ("a.png", fixtures["id_card.png"])}, data={"profile": name},
                   headers=k(ARCHIVE))
        assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden" and name not in r.text
    # a text-free profile can't be used to extract text
    r = c.post("/v1/extract", files={"file": ("a.png", fixtures["id_card.png"])}, data={"profile": "id-only"},
               headers=k(CIVIL_1))
    assert r.status_code == 403 and "/v1/documents" in r.json()["error"]["message"]


def test_profile_retention_replaces_result_ttl(make_client, fixtures, registry):
    c, _ = make_client(**with_profiles(registry))  # RESULT_TTL_S: 3600
    short = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1), profile="archive-text").json()["jobId"]
    normal = upload(c, fixtures["id_card.png"], headers=k(CIVIL_1)).json()["jobId"]
    wait_done(c, short, headers=k(CIVIL_1))
    wait_done(c, normal, headers=k(CIVIL_1))
    time.sleep(0.01)
    assert c.app.state.worker.sweep() == 1  # archive-text keeps results 0 s
    assert c.get(f"/v1/jobs/{short}", headers=k(CIVIL_1)).status_code == 404
    assert c.get(f"/v1/jobs/{normal}", headers=k(CIVIL_1)).status_code == 200


def test_minimized_results_are_neither_stored_nor_returned(make_client, fixtures, registry, settings, engine):
    llm = StubLlm(V("national_id", 0.9, "holder name SAMPLE"))
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    c, _ = make_client(**with_profiles(registry), analyze_fn=analyze, services=svc)
    job_id = upload(c, fixtures["passport.png"], headers=k(OPS)).json()["jobId"]  # ops: id-only by default
    out = wait_done(c, job_id, headers=k(OPS))
    assert out["idDetection"]["docType"] == "passport" and "mrz" not in out["idDetection"]
    assert out["fullText"] == "" and all(p["text"] == "" for p in out["pages"])
    assert out["processing"]["profile"] == "id-only" and out["processing"]["withheld"] == ["text", "mrz"]
    stored = c.app.state.store.get(job_id).result
    assert stored["fullText"] == "" and "mrz" not in stored["idDetection"] and "TESTER" not in json.dumps(stored)
    assert c.app.state.store.get(job_id).result_ttl_s == 86400


# --- JSON logs carry the request, job and client on every line ---------------------------------------------------

class JsonLines(logging.Handler):
    def __init__(self):
        from app.logs import JsonFormatter

        super().__init__(logging.INFO)
        self.setFormatter(JsonFormatter())
        self.lines: list[dict] = []

    def emit(self, record):
        self.lines.append(json.loads(self.format(record)))

    def first(self, text: str) -> dict:
        return next(line for line in self.lines if text in line["msg"])


@pytest.fixture
def json_logs():
    handler = JsonLines()
    root = logging.getLogger()
    old_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    yield handler
    root.removeHandler(handler)
    root.setLevel(old_level)


def test_json_logs_carry_request_job_and_client_ids(make_client, fixtures, json_logs):
    def logs_inside(data, filename, svc, **kw):  # e.g. the LLM client, which knows nothing about requests
        logging.getLogger("app.detection.llm").info("llm classify model=stub")
        return fake_analyze(data, filename, svc, **kw)

    c, _ = make_client(analyze_fn=logs_inside, transport=httpx.MockTransport(lambda r: httpx.Response(200)),
                       webhook_secret="s", webhook_allowed_hosts="hooks.internal")
    owner = legacy_owner(KEY)
    job_id = upload(c, fixtures["id_card.png"], name="SECRET_NAME.png", callbackUrl="http://hooks.internal/x",
                    headers={**H, "X-Request-Id": "trace-json"}).json()["jobId"]
    wait_done(c, job_id)
    assert wait_for(lambda: any("webhook delivered" in line["msg"] for line in json_logs.lines))
    queued, finished = json_logs.first("job queued"), json_logs.first("job finished")
    assert queued["request_id"] == "trace-json" and queued["client_id"] == owner  # the request thread
    assert finished["job_id"] == job_id and finished["request_id"] == "trace-json" and finished["client_id"] == owner
    inside = json_logs.first("llm classify")  # the worker's job context reaches every module
    assert inside["job_id"] == job_id and inside["request_id"] == "trace-json"
    assert json_logs.first("webhook delivered")["job_id"] == job_id  # the dispatcher's thread

    assert extract(c, fixtures["id_card.png"], headers={**H, "X-Request-Id": "trace-ex"}).status_code == 200
    in_threadpool = [line for line in json_logs.lines if "llm classify" in line["msg"]][-1]
    assert in_threadpool["request_id"] == "trace-ex" and "job_id" not in in_threadpool  # no job: a sync request

    upload(c, fixtures["id_card.png"], headers={"X-Api-Key": "wrong", "X-Request-Id": "trace-bad"})
    rejected = json_logs.first("auth rejected")
    assert rejected["request_id"] == "trace-bad" and "client_id" not in rejected
    assert "SECRET_NAME" not in json.dumps(json_logs.lines) and "SECRET_TEXT" not in json.dumps(json_logs.lines)


def test_example_alert_rules_use_existing_metrics():
    import re

    import yaml

    from app.api.metrics import Metrics

    rules = yaml.safe_load((Path(__file__).resolve().parents[1] / "docs" / "alerts.yml").read_text(encoding="utf-8"))
    alerts = [r for g in rules["groups"] for r in g["rules"]]
    assert alerts and all({"alert", "expr", "labels", "annotations"} <= set(r) for r in alerts)
    families = {m.name for m in Metrics().registry.collect()}
    used = {re.sub(r"_(total|bucket|count|sum)$", "", name)  # label values ("...") are not metric names
            for r in alerts for name in re.findall(r"\bocr_[a-z_]+", re.sub(r'"[^"]*"', "", r["expr"]))}
    assert used and used <= families, used - families
