"""PostgreSQL job store: durable jobs, a shared queue, and upload bytes in the database.

Any number of API processes/containers can share one database: a job uploaded to one replica is processed by
whichever worker claims it first (`FOR UPDATE SKIP LOCKED`), and results survive restarts.

Workers send heartbeats; a job whose worker died (crash, OOM, kill) goes back to the queue after JOB_STALE_S and
fails with `worker_lost` after JOB_MAX_ATTEMPTS claims (so a file that crashes the worker can't loop forever).

The schema comes from the numbered migrations in `app/api/migrations/` (see the README there).
"""

import json
import logging
import re
import time
from pathlib import Path

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.api.sealer import Sealer
from app.api.store import FINAL, WORKER_LOST, AuditEvent, IdempotencyConflict, Job, JobStore

log = logging.getLogger(__name__)
MIGRATIONS = Path(__file__).with_name("migrations")
MIGRATION_NAME = re.compile(r"(\d{3})_([a-z0-9_]+)\.sql")
SCHEMA_LOCK = 0x0C5_1D  # pg_advisory_xact_lock id: replicas starting together apply the schema one at a time

_META = """id, owner, size, status, callback_url, request_id, private, error, webhook, attempts, webhook_delivery_id,
    result_ttl_s,
    extract(epoch from created_at)::float8 AS created_at, extract(epoch from started_at)::float8 AS started_at,
    extract(epoch from finished_at)::float8 AS finished_at,
    extract(epoch from webhook_next_at)::float8 AS webhook_next_at,
    extract(epoch from webhook_settled_at)::float8 AS webhook_settled_at"""
_COLUMNS = _META + ", result"  # _META alone where the (large, sealed) result isn't needed, e.g. for webhooks
# A new delivery id in SQL, in the same format as uuid.uuid4().hex
_NEW_DELIVERY_ID = "replace(gen_random_uuid()::text, '-', '')"


class _KeyTaken(Exception):
    pass


def _insert_event(conn, ev: AuditEvent) -> None:
    conn.execute("INSERT INTO ocr_audit_events (event, client_id, job_id, request_id, data) "
                 "VALUES (%s, %s, %s, %s, %s)", (ev.event, ev.client_id, ev.job_id, ev.request_id, Jsonb(ev.data)))


def migrations(folder: Path = MIGRATIONS) -> list[tuple[int, str, Path]]:
    """(version, name, path) of every migration file, in order. Numbering must start at 001 with no gaps."""
    found = []
    for path in sorted(folder.glob("*.sql")):
        m = MIGRATION_NAME.fullmatch(path.name)
        if m is None:
            raise RuntimeError(f"migration file name {path.name!r} doesn't match NNN_name.sql")
        found.append((int(m[1]), m[2], path))
    if [v for v, _, _ in found] != list(range(1, len(found) + 1)):
        raise RuntimeError("migrations must be numbered 001, 002, ... without gaps or duplicates")
    return found


def apply_migrations(conn, folder: Path = MIGRATIONS) -> list[int]:
    """Apply the missing migrations in order; returns the versions applied now. Call inside a transaction."""
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK,))
    conn.execute("CREATE TABLE IF NOT EXISTS ocr_schema_migrations (version integer PRIMARY KEY, "
                 "name text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())")
    done = {r["version"] for r in conn.execute("SELECT version FROM ocr_schema_migrations").fetchall()}
    known = migrations(folder)
    if done and max(done) > len(known):  # a newer release migrated this database: migrations stay compatible
        log.warning("database schema is newer than this release db_version=%d code_version=%d",
                    max(done), len(known))
    applied = []
    for version, name, path in known:
        if version in done:
            continue
        conn.execute(path.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO ocr_schema_migrations (version, name) VALUES (%s, %s)", (version, name))
        log.info("schema migration applied version=%d name=%s", version, name)
        applied.append(version)
    return applied


class PostgresJobStore(JobStore):
    def __init__(self, dsn: str, sealer: Sealer | None = None, pool_size: int = 10, poll_s: float = 0.5):
        if not dsn:
            raise ValueError("JOB_STORE=postgres needs DATABASE_URL")
        self.sealer = sealer or Sealer()
        self.poll_s = poll_s
        self.pool = ConnectionPool(dsn, min_size=1, max_size=max(2, pool_size), open=False,
                                   kwargs={"autocommit": True, "row_factory": dict_row})

    # --- lifecycle -------------------------------------------------------------------------------------------
    def open(self) -> None:
        self.pool.open(wait=True, timeout=30)
        with self.pool.connection() as conn, conn.transaction():  # all migrations, or none
            apply_migrations(conn)
        log.info("job store ready backend=postgres schema=%d encrypted=%s", len(migrations()),
                 self.sealer.encrypting)

    def close(self) -> None:
        self.pool.close()

    def ping(self) -> bool:
        try:
            with self.pool.connection(timeout=3) as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:  # noqa: BLE001 - any failure means not ready
            return False

    # --- (de)serialization -----------------------------------------------------------------------------------
    def _seal_json(self, obj, aad: str) -> bytes:
        return self.sealer.seal(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode(), aad)

    def _open_json(self, blob, aad: str):
        return json.loads(self.sealer.open(blob, aad)) if blob is not None else None

    def _job(self, row: dict) -> Job:
        private = self._open_json(row["private"], row["id"] + ":private")
        return Job(id=row["id"], owner=row["owner"], filename=private.get("filename", ""), size=row["size"],
                   callback_url=row["callback_url"], external_id=private.get("externalId"),
                   metadata=private.get("metadata"), options=private.get("options"),
                   request_id=row["request_id"], status=row["status"],
                   created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"],
                   attempts=row["attempts"], result=self._open_json(row.get("result"), row["id"] + ":result"),
                   error=row["error"], webhook=row["webhook"], webhook_delivery_id=row["webhook_delivery_id"],
                   webhook_next_at=row["webhook_next_at"], webhook_settled_at=row["webhook_settled_at"],
                   result_ttl_s=row["result_ttl_s"])

    def _set_clause(self, job_id: str, changes: dict) -> tuple[str, list]:
        sets, params = [], []
        for name, value in changes.items():
            if name == "result":
                value = None if value is None else self._seal_json(value, job_id + ":result")
            elif name in ("error", "webhook"):
                value = None if value is None else Jsonb(value)
            elif name not in ("status", "finished_at", "started_at", "webhook_delivery_id", "webhook_next_at",
                              "webhook_settled_at"):
                raise ValueError(f"field {name} can't be updated")
            sets.append(f"{name} = to_timestamp(%s)" if name.endswith("_at") else f"{name} = %s")
            params.append(value)
        return ", ".join(sets), params

    # --- jobs ------------------------------------------------------------------------------------------------
    def add(self, job: Job, data: bytes, idempotency: tuple[str, str] | None = None,
            audit: AuditEvent | None = None) -> Job:
        if idempotency and (hit := self._existing(job.owner, idempotency)):
            return hit
        private = {"filename": job.filename, "externalId": job.external_id, "metadata": job.metadata,
                   "options": job.options}
        try:
            with self.pool.connection() as conn, conn.transaction():
                conn.execute(
                    "INSERT INTO ocr_jobs (id, owner, size, status, callback_url, request_id, private, result_ttl_s) "
                    "VALUES (%s, %s, %s, 'queued', %s, %s, %s, %s)",
                    (job.id, job.owner, job.size, job.callback_url, job.request_id,
                     self._seal_json(private, job.id + ":private"), job.result_ttl_s))
                conn.execute("INSERT INTO ocr_job_uploads (job_id, data) VALUES (%s, %s)",
                             (job.id, self.sealer.seal(data, job.id + ":upload")))
                if idempotency:
                    row = conn.execute(
                        "INSERT INTO ocr_idempotency_keys (owner, key, file_sha256, job_id) VALUES (%s, %s, %s, %s) "
                        "ON CONFLICT (owner, key) DO NOTHING RETURNING job_id",
                        (job.owner, idempotency[0], idempotency[1], job.id)).fetchone()
                    if row is None:  # a concurrent request with the same key won: roll this one back
                        raise _KeyTaken
                if audit:
                    _insert_event(conn, audit)
        except _KeyTaken:
            hit = self._existing(job.owner, idempotency)
            if hit is None:
                raise IdempotencyConflict from None  # the winner was deleted in between; let the client retry
            return hit
        return self.get(job.id)

    def _existing(self, owner: str, idempotency: tuple[str, str]) -> Job | None:
        hit = self.get_idempotent(owner, idempotency[0])
        if hit is None:
            return None
        if hit[1] != idempotency[1]:
            raise IdempotencyConflict
        return self.get(hit[0])

    def get_idempotent(self, owner: str, key: str) -> tuple[str, str] | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT job_id, file_sha256 FROM ocr_idempotency_keys WHERE owner = %s AND key = %s",
                               (owner, key)).fetchone()
        return (row["job_id"], row["file_sha256"]) if row else None

    def get(self, job_id: str) -> Job | None:
        with self.pool.connection() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM ocr_jobs WHERE id = %s", (job_id,)).fetchone()
        return self._job(row) if row else None

    def update(self, job_id: str, **changes) -> Job | None:
        sets, params = self._set_clause(job_id, changes)
        with self.pool.connection() as conn:
            row = conn.execute(f"UPDATE ocr_jobs SET {sets} WHERE id = %s RETURNING {_COLUMNS}",
                               (*params, job_id)).fetchone()
        return self._job(row) if row else None

    def finish(self, job_id: str, worker_id: str, audit: AuditEvent | None = None, **changes) -> Job | None:
        sets, params = self._set_clause(job_id, changes)
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                f"UPDATE ocr_jobs SET {sets}, locked_by = NULL WHERE id = %s AND locked_by = %s "
                f"AND status = 'processing' RETURNING {_COLUMNS}", (*params, job_id, worker_id)).fetchone()
            if row and row["status"] in FINAL:  # a finished job never keeps its document
                conn.execute("DELETE FROM ocr_job_uploads WHERE job_id = %s", (job_id,))
            if row and audit:
                _insert_event(conn, audit)
        return self._job(row) if row else None

    def delete(self, job_id: str, audit: AuditEvent | None = None) -> Job | None:
        job = self.get(job_id)
        with self.pool.connection() as conn, conn.transaction():
            # cascades to the upload and the idempotency key
            gone = conn.execute("DELETE FROM ocr_jobs WHERE id = %s RETURNING id", (job_id,)).fetchone()
            if gone and audit:
                _insert_event(conn, audit)
        return job

    # --- queue -----------------------------------------------------------------------------------------------
    def claim(self, timeout: float, worker_id: str) -> Job | None:
        end = time.monotonic() + timeout
        while True:
            with self.pool.connection() as conn:  # one statement: the claim and its job.started event, atomically
                row = conn.execute(
                    "WITH claimed AS ("
                    "  UPDATE ocr_jobs SET status = 'processing', locked_by = %(w)s, started_at = now(), "
                    "    heartbeat_at = now(), attempts = attempts + 1 WHERE id = ("
                    "      SELECT id FROM ocr_jobs WHERE status = 'queued' ORDER BY created_at "
                    "      FOR UPDATE SKIP LOCKED LIMIT 1) "
                    "  RETURNING *), "
                    "started AS (INSERT INTO ocr_audit_events (event, client_id, job_id, request_id, data) "
                    "  SELECT 'job.started', owner, id, request_id, "
                    "    jsonb_build_object('attempt', attempts, 'worker', %(w)s::text) FROM claimed) "
                    f"SELECT {_COLUMNS} FROM claimed", {"w": worker_id}).fetchone()
            if row:
                return self._job(row)
            left = end - time.monotonic()
            if left <= 0:
                return None
            time.sleep(min(self.poll_s, left))

    def heartbeat(self, worker_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE ocr_jobs SET heartbeat_at = now() WHERE locked_by = %s AND status = 'processing'",
                         (worker_id,))

    def requeue_stale(self, stale_s: float, max_attempts: int) -> list[str]:
        with self.pool.connection() as conn, conn.transaction():
            rows = conn.execute(
                "UPDATE ocr_jobs SET locked_by = NULL, "
                "  status = CASE WHEN attempts >= %(max)s THEN 'failed' ELSE 'queued' END, "
                "  error = CASE WHEN attempts >= %(max)s THEN %(err)s::jsonb END, "
                "  finished_at = CASE WHEN attempts >= %(max)s THEN now() END, "
                # a job that fails here gets its webhook scheduled, like one that fails in a worker
                "  webhook = CASE WHEN attempts >= %(max)s AND callback_url IS NOT NULL THEN %(pending)s::jsonb "
                "    ELSE webhook END, "
                f"  webhook_delivery_id = CASE WHEN attempts >= %(max)s AND callback_url IS NOT NULL "
                f"    THEN {_NEW_DELIVERY_ID} ELSE webhook_delivery_id END, "
                "  webhook_next_at = CASE WHEN attempts >= %(max)s AND callback_url IS NOT NULL THEN now() "
                "    ELSE webhook_next_at END "
                "WHERE status = 'processing' AND heartbeat_at < now() - make_interval(secs => %(stale)s) "
                "RETURNING id, status, owner, request_id, attempts",
                {"max": max_attempts, "stale": stale_s, "err": json.dumps(WORKER_LOST),
                 "pending": json.dumps({"state": "pending", "attempts": 0})}).fetchall()
            failed = [r["id"] for r in rows if r["status"] == "failed"]
            if failed:
                conn.execute("DELETE FROM ocr_job_uploads WHERE job_id = ANY(%s)", (failed,))
            for r in rows:
                lost = r["status"] == "failed"
                _insert_event(conn, AuditEvent(
                    "job.failed" if lost else "job.requeued", r["owner"], r["id"], r["request_id"],
                    {"error": WORKER_LOST["code"], "attempts": r["attempts"]} if lost
                    else {"reason": "worker_stale", "attempts": r["attempts"]}))
        if rows:
            log.warning("stale jobs recovered requeued=%d failed=%d", len(rows) - len(failed), len(failed))
        return failed

    def release(self, worker_id: str) -> int:
        with self.pool.connection() as conn:  # one statement: the release and its audit events
            return conn.execute(
                "WITH released AS (UPDATE ocr_jobs SET status = 'queued', locked_by = NULL, started_at = NULL, "
                "    heartbeat_at = NULL, attempts = greatest(attempts - 1, 0) "
                "  WHERE locked_by = %(w)s AND status = 'processing' RETURNING id, owner, request_id, attempts) "
                "INSERT INTO ocr_audit_events (event, client_id, job_id, request_id, data) "
                "SELECT 'job.requeued', owner, id, request_id, "
                "  jsonb_build_object('reason', 'shutdown', 'attempts', attempts) FROM released",
                {"w": worker_id}).rowcount

    # --- uploads ---------------------------------------------------------------------------------------------
    def read_upload(self, job_id: str) -> bytes | None:
        with self.pool.connection() as conn:
            row = conn.execute("SELECT data FROM ocr_job_uploads WHERE job_id = %s", (job_id,)).fetchone()
        return self.sealer.open(row["data"], job_id + ":upload") if row else None

    def delete_upload(self, job_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM ocr_job_uploads WHERE job_id = %s", (job_id,))

    # --- housekeeping ----------------------------------------------------------------------------------------
    def purge_finished_before(self, cutoff: float, now: float | None = None) -> int:
        # a job's own retention (its profile's) replaces the global cutoff
        limit = ("coalesce(to_timestamp(%(now)s) - make_interval(secs => result_ttl_s), "
                 "to_timestamp(%(cutoff)s))")
        with self.pool.connection() as conn:  # one statement: the jobs go and their audit events come together
            return conn.execute(
                f"WITH gone AS (DELETE FROM ocr_jobs WHERE finished_at < {limit} "
                "    AND webhook_next_at IS NULL "  # a pending webhook keeps its job (the receiver will fetch it)
                f"    AND (webhook_settled_at IS NULL OR webhook_settled_at < {limit}) "
                "  RETURNING id, owner, request_id) "
                "INSERT INTO ocr_audit_events (event, client_id, job_id, request_id, data) "
                """SELECT 'job.deleted', owner, id, request_id, '{"by": "ttl"}'::jsonb FROM gone""",
                {"cutoff": cutoff, "now": time.time() if now is None else now}).rowcount

    # --- webhook outbox -----------------------------------------------------------------------------------------
    def claim_webhooks(self, limit: int, lease_s: float) -> list[Job]:
        if limit <= 0:
            return []
        with self.pool.connection() as conn:
            rows = conn.execute(
                "UPDATE ocr_jobs SET webhook_next_at = now() + make_interval(secs => %(lease)s) WHERE id IN ("
                "  SELECT id FROM ocr_jobs WHERE webhook_next_at <= now() ORDER BY webhook_next_at "
                "  FOR UPDATE SKIP LOCKED LIMIT %(n)s) "
                f"RETURNING {_META}", {"lease": lease_s, "n": limit}).fetchall()
        return [self._job(r) for r in rows]

    def webhook_attempted(self, job_id: str, delivery_id: str, *, state: str, attempts: int,
                          next_at: float | None, audit: AuditEvent | None = None) -> Job | None:
        settled = state != "pending"
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "UPDATE ocr_jobs SET webhook = %(webhook)s, webhook_next_at = to_timestamp(%(next)s), "
                "  webhook_settled_at = CASE WHEN %(settled)s THEN now() END "
                f"WHERE id = %(id)s AND webhook_delivery_id = %(delivery)s RETURNING {_META}",
                {"webhook": Jsonb({"state": state, "attempts": attempts}), "next": None if settled else next_at,
                 "settled": settled, "id": job_id, "delivery": delivery_id}).fetchone()
            if row and audit:
                _insert_event(conn, audit)
        return self._job(row) if row else None

    def count_pending_webhooks(self) -> int:
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) AS n FROM ocr_jobs WHERE webhook_next_at IS NOT NULL").fetchone()["n"]

    def count_queued(self) -> int:
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) AS n FROM ocr_jobs WHERE status = 'queued'").fetchone()["n"]

    def count_queued_for(self, owner: str) -> int:
        with self.pool.connection() as conn:
            return conn.execute("SELECT count(*) AS n FROM ocr_jobs WHERE status = 'queued' AND owner = %s",
                                (owner,)).fetchone()["n"]

    def counts(self) -> dict[str, int]:
        out = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
        with self.pool.connection() as conn:
            for r in conn.execute("SELECT status, count(*) AS n FROM ocr_jobs GROUP BY status").fetchall():
                out[r["status"]] = r["n"]
        return out

    def ages(self) -> dict[str, float]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT extract(epoch from now() - min(created_at) FILTER (WHERE status = 'queued'))::float8 AS q, "
                "  extract(epoch from now() - min(started_at) FILTER (WHERE status = 'processing'))::float8 AS p "
                "FROM ocr_jobs WHERE status IN ('queued', 'processing')").fetchone()
        return {"queued": row["q"] or 0.0, "processing": row["p"] or 0.0}

    # --- audit trail -----------------------------------------------------------------------------------------
    def record(self, event: AuditEvent) -> None:
        with self.pool.connection() as conn:
            _insert_event(conn, event)

    def audit_events(self, *, job_id: str | None = None, request_id: str | None = None, client_id: str | None = None,
                     since: float | None = None, limit: int = 1000) -> list[dict]:
        where, params = [], []
        for column, value in (("job_id", job_id), ("request_id", request_id), ("client_id", client_id)):
            if value is not None:
                where.append(f"{column} = %s")
                params.append(value)
        if since is not None:
            where.append("at >= to_timestamp(%s)")
            params.append(since)
        sql = ("SELECT id, extract(epoch from at)::float8 AS at, event, client_id, job_id, request_id, data "
               "FROM ocr_audit_events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id LIMIT %s")
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (*params, limit)).fetchall()
        return [{"id": r["id"], "at": r["at"], "event": r["event"], "clientId": r["client_id"], "jobId": r["job_id"],
                 "requestId": r["request_id"], "data": r["data"]} for r in rows]

    def purge_events_before(self, cutoff: float) -> int:
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM ocr_audit_events WHERE at < to_timestamp(%s)", (cutoff,)).rowcount
