"""JobStore contract: the same tests for the in-memory and the PostgreSQL store.

The PostgreSQL variant runs only when TEST_DATABASE_URL points to a database it may freely write to, e.g.
    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=dev postgres:17
    $env:TEST_DATABASE_URL = 'postgresql://postgres:dev@127.0.0.1:5432/postgres'
"""

import base64
import os
import threading
import time
import uuid

import pytest

from app.api.sealer import Sealer, SealError
from app.api.store import WORKER_LOST, IdempotencyConflict, InMemoryJobStore, Job

PG_URL = os.environ.get("TEST_DATABASE_URL", "")
KEY = base64.b64encode(b"k" * 32).decode()


def _drop_tables():
    import psycopg

    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS ocr_idempotency_keys, ocr_job_uploads, ocr_jobs, ocr_audit_events, "
                     "ocr_schema_migrations, ocr_instance_samples, ocr_alerts, ocr_log_records, ocr_config_versions, "
                     "ocr_client_secrets CASCADE")


def pg_store(sealer=None):
    from app.api.store_pg import PostgresJobStore

    return PostgresJobStore(PG_URL, sealer or Sealer(KEY), pool_size=8, poll_s=0.05)


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(
    not PG_URL, reason="TEST_DATABASE_URL not set"))])
def store(request, tmp_path):
    if request.param == "memory":
        s = InMemoryJobStore(tmp_path / "uploads")
    else:
        _drop_tables()
        s = pg_store()
    s.open()
    yield s
    s.close()


def new_job(owner="o1", **kw) -> Job:
    return Job(id=uuid.uuid4().hex, owner=owner, filename="SECRET_NAME.pdf", size=3, **kw)


def test_add_claim_finish_roundtrip(store):
    job = store.add(new_job(external_id="ext-1", metadata={"caseId": 7}, callback_url="http://h/x"), b"abc")
    assert store.count_queued() == 1 and store.counts()["queued"] == 1
    claimed = store.claim(1, "w1")
    assert claimed.id == job.id and claimed.status == "processing" and claimed.attempts == 1
    assert claimed.filename == "SECRET_NAME.pdf" and claimed.external_id == "ext-1"
    assert claimed.metadata == {"caseId": 7} and claimed.callback_url == "http://h/x"
    assert store.read_upload(job.id) == b"abc"
    done = store.finish(job.id, "w1", status="done", result={"fullText": "نص"}, error=None, finished_at=time.time())
    assert done.status == "done" and done.result == {"fullText": "نص"} and done.finished_at is not None
    assert store.read_upload(job.id) is None  # finishing deletes the upload
    assert store.claim(0.1, "w1") is None


def test_update_and_delete(store):
    job = store.add(new_job(), b"abc")
    assert store.update(job.id, webhook={"state": "pending", "attempts": 0}).webhook["state"] == "pending"
    assert store.delete(job.id).id == job.id
    assert store.get(job.id) is None and store.read_upload(job.id) is None
    assert store.update(job.id, status="done") is None
    assert store.claim(0.1, "w1") is None  # deleted while queued


def test_idempotency(store):
    first = store.add(new_job(), b"abc", ("key-1", "sha-a"))
    again = store.add(new_job(), b"abc", ("key-1", "sha-a"))
    assert again.id == first.id and store.count_queued() == 1
    assert store.get_idempotent("o1", "key-1") == (first.id, "sha-a")
    with pytest.raises(IdempotencyConflict):
        store.add(new_job(), b"xyz", ("key-1", "sha-b"))
    other_owner = store.add(new_job(owner="o2"), b"abc", ("key-1", "sha-a"))
    assert other_owner.id != first.id  # keys are per API key
    store.delete(first.id)
    assert store.get_idempotent("o1", "key-1") is None  # the key expires with its job


def test_count_queued_per_owner(store):
    a1, _, _ = store.add(new_job(owner="a"), b"x"), store.add(new_job(owner="a"), b"x"), store.add(new_job("b"), b"x")
    assert store.count_queued() == 3 and store.count_queued_for("a") == 2 and store.count_queued_for("b") == 1
    claimed = store.claim(1, "w")  # the oldest: one of a's
    assert claimed.id == a1.id and store.count_queued_for("a") == 1 and store.count_queued_for("nobody") == 0


def test_purge(store):
    job = store.add(new_job(), b"abc")
    store.claim(1, "w")
    store.finish(job.id, "w", status="done", finished_at=time.time() - 100)
    assert store.purge_finished_before(time.time() - 50) == 1 and store.get(job.id) is None


def test_concurrent_claims_never_share_a_job(store):
    ids = {store.add(new_job(), b"x").id for _ in range(20)}
    got: list[str] = []
    lock = threading.Lock()

    def take(worker):
        while (job := store.claim(0.2, worker)) is not None:
            with lock:
                got.append(job.id)

    threads = [threading.Thread(target=take, args=(f"w{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(got) == sorted(ids)


# --- Audit trail ---------------------------------------------------------------------------------------------

def test_audit_events_follow_the_job_lifecycle(store):
    from app.api.store import AuditEvent

    job = store.add(new_job(owner="c1", request_id="req-1"), b"abc",
                    audit=AuditEvent("job.received", "c1", None, "req-1", {"size": 3}))
    store.claim(1, "w1")
    store.finish(job.id, "w1", audit=AuditEvent("job.done", "c1", job.id, "req-1", {"ms": 5}), status="done",
                 finished_at=time.time() - 100)
    assert store.purge_finished_before(time.time() - 50) == 1
    events = store.audit_events(client_id="c1")
    assert [e["event"] for e in events] == ["job.received", "job.started", "job.done", "job.deleted"]
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)  # oldest first
    assert all(e["requestId"] == "req-1" for e in events)
    assert events[1]["jobId"] == job.id and events[1]["data"] == {"attempt": 1, "worker": "w1"}
    assert events[3]["jobId"] == job.id and events[3]["data"] == {"by": "ttl"}
    assert events[2]["data"] == {"ms": 5} and isinstance(events[0]["at"], float)


def test_audit_events_only_for_changes_that_happened(store):
    from app.api.store import AuditEvent

    first = store.add(new_job(), b"abc", ("key-1", "sha-a"))
    again = new_job()
    assert store.add(again, b"abc", ("key-1", "sha-a"), audit=AuditEvent("job.received", "o1", again.id)).id == first.id
    assert store.audit_events(job_id=again.id) == []  # the replay stored nothing
    assert store.delete("no-such-job", audit=AuditEvent("job.deleted", "o1", "no-such-job")) is None
    assert store.audit_events(job_id="no-such-job") == []
    store.delete(first.id, audit=AuditEvent("job.deleted", "o1", first.id, "req-del", {"by": "client"}))
    assert [(e["event"], e["requestId"], e["data"]) for e in store.audit_events(job_id=first.id)] == [
        ("job.deleted", "req-del", {"by": "client"})]


def test_audit_record_query_and_purge(store):
    from app.api.store import AuditEvent

    store.record(AuditEvent("request.error", None, None, "r1", {"status": 401, "code": "unauthorized"}))
    store.record(AuditEvent("extract.done", "c2", None, "r2", {"pages": 1, "pageMethods": {"ocr": 1}}))
    assert [e["event"] for e in store.audit_events(client_id="c2")] == ["extract.done"]
    assert store.audit_events(request_id="r1")[0]["data"] == {"status": 401, "code": "unauthorized"}
    assert store.audit_events(request_id="r1")[0]["clientId"] is None
    assert len(store.audit_events(since=time.time() - 60)) == 2 and store.audit_events(since=time.time() + 60) == []
    assert [e["event"] for e in store.audit_events(limit=1)] == ["request.error"]
    assert store.purge_events_before(time.time() - 60) == 0
    assert store.purge_events_before(time.time() + 60) == 2 and store.audit_events() == []


# --- Webhook outbox ------------------------------------------------------------------------------------------

def finished_with_webhook(store, worker="w1", finished_at=None):
    job = store.add(new_job(callback_url="http://h/x"), b"abc")
    store.claim(1, worker)
    store.finish(job.id, worker, status="done", finished_at=finished_at or time.time(),
                 webhook={"state": "pending", "attempts": 0}, webhook_delivery_id=uuid.uuid4().hex,
                 webhook_next_at=time.time())
    return store.get(job.id)


def test_webhook_deliveries_are_leased_to_one_dispatcher(store):
    job = finished_with_webhook(store)
    assert store.count_pending_webhooks() == 1
    [claimed] = store.claim_webhooks(10, lease_s=60)
    assert claimed.id == job.id and claimed.webhook_delivery_id == job.webhook_delivery_id
    assert claimed.external_id is None and claimed.callback_url == "http://h/x"
    assert store.claim_webhooks(10, lease_s=60) == []  # leased: no other dispatcher gets it
    assert store.claim_webhooks(0, lease_s=60) == []
    assert store.webhook_attempted(job.id, "another-delivery", state="delivered", attempts=1, next_at=None) is None
    done = store.webhook_attempted(job.id, job.webhook_delivery_id, state="delivered", attempts=1, next_at=None)
    assert done.webhook == {"state": "delivered", "attempts": 1} and done.webhook_next_at is None
    assert done.webhook_settled_at is not None and store.count_pending_webhooks() == 0


def test_webhook_retry_is_due_again_later(store):
    job = finished_with_webhook(store)
    store.claim_webhooks(10, lease_s=60)
    store.webhook_attempted(job.id, job.webhook_delivery_id, state="pending", attempts=1, next_at=time.time() - 1)
    [again] = store.claim_webhooks(10, lease_s=60)
    assert again.webhook == {"state": "pending", "attempts": 1} and again.webhook_delivery_id == job.webhook_delivery_id


def test_purge_spares_pending_and_recently_settled_webhooks(store):
    old = time.time() - 7200
    pending = finished_with_webhook(store, finished_at=old)
    settled = finished_with_webhook(store, finished_at=old)
    store.claim_webhooks(10, lease_s=60)
    store.webhook_attempted(settled.id, settled.webhook_delivery_id, state="failed", attempts=3, next_at=None)
    assert store.purge_finished_before(time.time() - 3600) == 0  # finished long ago, but notified only now / not yet
    assert store.get(pending.id) is not None and store.get(settled.id) is not None
    assert store.purge_finished_before(time.time() + 1) == 1 and store.get(settled.id) is None


# --- PostgreSQL only -----------------------------------------------------------------------------------------

pg_only = pytest.mark.skipif(not PG_URL, reason="TEST_DATABASE_URL not set")


@pg_only
def test_pg_jobs_survive_a_new_store_instance():
    _drop_tables()
    a = pg_store()
    a.open()
    job = a.add(new_job(), b"abc")
    a.close()
    b = pg_store()
    b.open()  # schema creation is idempotent
    try:
        assert b.get(job.id).status == "queued" and b.claim(1, "w").id == job.id
    finally:
        b.close()


@pg_only
def test_pg_stale_jobs_are_requeued_then_failed():
    _drop_tables()
    s = pg_store()
    s.open()
    try:
        job = s.add(new_job(), b"abc")
        assert s.claim(1, "dead-worker").attempts == 1
        time.sleep(1.1)
        assert s.requeue_stale(1, max_attempts=2) == []  # back to the queue
        assert s.get(job.id).status == "queued"
        assert s.claim(1, "dead-again").attempts == 2
        assert s.finish(job.id, "dead-worker", status="done") is None  # the first worker lost the job
        time.sleep(1.1)
        assert s.requeue_stale(1, max_attempts=2) == [job.id]
        failed = s.get(job.id)
        assert failed.status == "failed" and failed.error == WORKER_LOST and s.read_upload(job.id) is None
    finally:
        s.close()


@pg_only
def test_pg_stale_recovery_is_audited_and_a_lost_finish_is_not():
    from app.api.store import AuditEvent

    _drop_tables()
    s = pg_store()
    s.open()
    try:
        job = s.add(new_job(owner="c1", request_id="req-9"), b"abc")
        s.claim(1, "dead-worker")
        time.sleep(1.1)
        s.requeue_stale(1, max_attempts=2)
        s.claim(1, "second")
        assert s.finish(job.id, "dead-worker", audit=AuditEvent("job.done", "c1", job.id), status="done") is None
        time.sleep(1.1)
        s.requeue_stale(1, max_attempts=2)
        events = s.audit_events(job_id=job.id)
        assert [e["event"] for e in events] == ["job.started", "job.requeued", "job.started", "job.failed"]
        assert events[1]["data"] == {"reason": "worker_stale", "attempts": 1}
        assert events[3]["data"] == {"error": "worker_lost", "attempts": 2}
        assert all(e["clientId"] == "c1" and e["requestId"] == "req-9" for e in events)
    finally:
        s.close()


@pg_only
def test_pg_release_requeues_without_counting_the_attempt():
    _drop_tables()
    s = pg_store()
    s.open()
    try:
        job = s.add(new_job(owner="c1"), b"abc")
        assert s.claim(1, "shutting-down").attempts == 1
        other = s.add(new_job(), b"abc")
        s.claim(1, "other-worker")
        assert s.release("shutting-down") == 1
        back = s.get(job.id)
        assert back.status == "queued" and back.attempts == 0 and back.started_at is None
        assert s.get(other.id).status == "processing"  # another worker's job is untouched
        assert s.finish(job.id, "shutting-down", status="done") is None  # a late finish is a no-op
        assert s.claim(1, "next").id == job.id and s.get(job.id).attempts == 1
        [ev] = [e for e in s.audit_events(job_id=job.id) if e["event"] == "job.requeued"]
        assert ev["data"] == {"reason": "shutdown", "attempts": 0} and ev["clientId"] == "c1"
    finally:
        s.close()


@pg_only
def test_pg_worker_lost_job_gets_its_webhook():
    _drop_tables()
    s = pg_store()
    s.open()
    try:
        job = s.add(new_job(callback_url="http://h/x", external_id="ext-9"), b"abc")
        s.claim(1, "dead")
        time.sleep(1.1)
        assert s.requeue_stale(1, max_attempts=1) == [job.id]
        [due] = s.claim_webhooks(10, lease_s=60)
        assert due.id == job.id and due.status == "failed" and due.error["code"] == "worker_lost"
        assert len(due.webhook_delivery_id) == 32 and due.webhook == {"state": "pending", "attempts": 0}
        assert due.external_id == "ext-9"  # the payload can be rebuilt from the row
    finally:
        s.close()


@pg_only
def test_pg_delivery_survives_its_sender_dying():
    """A claimed delivery whose dispatcher died is due again after the lease, from any process."""
    _drop_tables()
    a = pg_store()
    a.open()
    job = finished_with_webhook(a)
    [first] = a.claim_webhooks(10, lease_s=0.5)
    a.close()  # crashed while sending
    b = pg_store()
    b.open()
    try:
        assert b.claim_webhooks(10, lease_s=60) == []  # still leased
        time.sleep(0.7)
        [again] = b.claim_webhooks(10, lease_s=60)
        assert again.id == job.id and again.webhook_delivery_id == first.webhook_delivery_id
    finally:
        b.close()


@pg_only
def test_pg_migration_004_resends_deliveries_stuck_in_an_old_release():
    import psycopg

    _drop_tables()
    s = pg_store()
    s.open()
    stuck, recent = (s.add(new_job(callback_url="http://h/x"), b"x") for _ in range(2))
    s.close()
    with psycopg.connect(PG_URL, autocommit=True) as conn:  # as the in-memory dispatcher left them
        for job_id, ago in ((stuck.id, "10 minutes"), (recent.id, "1 minute")):
            conn.execute("UPDATE ocr_jobs SET status = 'done', finished_at = now() - %s::interval, "
                         """webhook = '{"state": "pending", "attempts": 0}' WHERE id = %s""", (ago, job_id))
        conn.execute("DELETE FROM ocr_schema_migrations WHERE version = 4")  # as if 004 had not run yet
    s = pg_store()
    s.open()  # applies 004 again (its statements are idempotent)
    try:
        assert [j.id for j in s.claim_webhooks(10, lease_s=60)] == [stuck.id]
        assert s.get(recent.id).webhook_next_at is None  # may still be in flight on an old replica
    finally:
        s.close()


@pg_only
def test_pg_heartbeat_keeps_job():
    _drop_tables()
    s = pg_store()
    s.open()
    try:
        job = s.add(new_job(), b"abc")
        s.claim(1, "w")
        time.sleep(1.1)
        s.heartbeat("w")
        assert s.requeue_stale(1, 2) == [] and s.get(job.id).status == "processing"
    finally:
        s.close()


@pg_only
def test_pg_stores_no_plaintext_when_encrypted():
    import psycopg

    _drop_tables()
    s = pg_store()
    s.open()
    try:
        job = s.add(new_job(metadata={"note": "SECRET_META"}), b"SECRET_BYTES")
        s.claim(1, "w")
        s.finish(job.id, "w", status="done", result={"fullText": "SECRET_TEXT"}, finished_at=time.time())
        with psycopg.connect(PG_URL) as conn:
            raw = b"".join(bytes(v) for row in conn.execute(
                "SELECT j.private, j.result, u.data FROM ocr_jobs j JOIN ocr_job_uploads u ON u.job_id = j.id"
            ).fetchall() for v in row)
        for secret in (b"SECRET_BYTES", b"SECRET_TEXT", b"SECRET_META", b"SECRET_NAME"):
            assert secret not in raw
        assert s.get(job.id).result == {"fullText": "SECRET_TEXT"}
    finally:
        s.close()


@pg_only
def test_pg_migrations_applied_once_and_recorded():
    import psycopg

    _drop_tables()
    a, b = pg_store(), pg_store()
    a.open()
    b.open()  # a second replica: nothing left to apply
    try:
        from app.api.store_pg import migrations

        with psycopg.connect(PG_URL) as conn:
            rows = conn.execute("SELECT version, name FROM ocr_schema_migrations ORDER BY version").fetchall()
        assert rows == [(v, name) for v, name, _ in migrations()] and rows[0] == (1, "init")
        assert b.add(new_job(), b"abc").status == "queued"
    finally:
        a.close()
        b.close()


@pg_only
def test_pg_database_from_before_migrations_is_adopted():
    """Tables created by the old schema.sql, without ocr_schema_migrations: 001 runs as a no-op, data stays."""
    import psycopg

    _drop_tables()
    s = pg_store()
    s.open()
    job = s.add(new_job(), b"abc")
    s.close()
    with psycopg.connect(PG_URL, autocommit=True) as conn:
        conn.execute("DROP TABLE ocr_schema_migrations")
    s = pg_store()
    s.open()
    try:
        assert s.get(job.id).status == "queued"
    finally:
        s.close()


# --- Migration files -----------------------------------------------------------------------------------------

def test_migration_files_are_numbered_without_gaps():
    from app.api.store_pg import migrations

    found = migrations()
    assert [v for v, _, _ in found] == list(range(1, len(found) + 1)) and found[0][1] == "init"


@pytest.mark.parametrize("names", [["001_init.sql", "003_x.sql"], ["002_x.sql"], ["001_init.sql", "01_bad.sql"],
                                   ["001_init.sql", "001_again.sql"]])
def test_bad_migration_numbering_is_rejected(tmp_path, names):
    from app.api.store_pg import migrations

    for name in names:
        (tmp_path / name).write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(RuntimeError):
        migrations(tmp_path)


# --- Sealer --------------------------------------------------------------------------------------------------

def test_sealer_roundtrip_and_binding():
    s = Sealer(KEY)
    blob = s.seal(b"data", "job1:upload")
    assert b"data" not in blob and s.open(blob, "job1:upload") == b"data"
    with pytest.raises(SealError):
        s.open(blob, "job2:upload")  # bound to its job
    with pytest.raises(SealError):
        Sealer().open(blob, "job1:upload")  # key missing
    assert Sealer().open(Sealer().seal(b"plain", "x"), "x") == b"plain"
    with pytest.raises(SealError):
        Sealer(base64.b64encode(b"short").decode())


def test_a_job_keeps_its_own_retention(store):
    short, long_, default = (store.add(new_job(result_ttl_s=ttl), b"x") for ttl in (10, 100_000, None))
    for _ in range(3):
        store.claim(1, "w")
    for job in (short, long_, default):
        store.finish(job.id, "w", status="done", finished_at=time.time() - 1000)
    assert store.get(long_.id).result_ttl_s == 100_000
    assert store.purge_finished_before(time.time() - 500) == 2  # short (10 s) and default (500 s) expired
    assert store.get(long_.id) is not None and store.get(short.id) is None and store.get(default.id) is None


def test_ages_of_the_oldest_queued_and_processing_jobs(store):
    assert store.ages() == {"queued": 0.0, "processing": 0.0}
    store.add(new_job(), b"x")
    store.add(new_job(), b"x")
    store.claim(1, "w")
    time.sleep(0.3)
    ages = store.ages()
    assert 0.25 <= ages["queued"] < 30 and 0.25 <= ages["processing"] < 30
