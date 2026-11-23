"""Job store + queue + upload bytes + audit trail.

- `InMemoryJobStore`: single process, nothing survives a restart; uploads are files in UPLOAD_DIR (mode 0600).
- `PostgresJobStore` (`app/api/store_pg.py`): durable, shared by any number of processes/containers.
Choose with JOB_STORE=memory|postgres.

Audit events (see app/api/audit.py) of a job's lifecycle are written by the store itself, together with the state
change they describe (in one transaction with PostgreSQL): received (`add`), started (`claim`), done/failed
(`finish`), requeued/failed (`requeue_stale`, `release`), deleted (`delete`, `purge_finished_before`),
webhook delivered/failed (`webhook_attempted`).

Webhooks use the job row as an outbox: a due delivery has `webhook_next_at` set, any process's dispatcher claims it
(`claim_webhooks`), and the outcome is written back (`webhook_attempted`). See app/api/webhook.py.
"""

import itertools
import os
import queue
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

Status = Literal["queued", "processing", "done", "failed"]
FINAL = ("done", "failed")
WORKER_LOST = {"code": "worker_lost", "message": "processing was interrupted too often"}
MEMORY_AUDIT_EVENTS = 10_000  # the memory store (development) keeps only the latest events


@dataclass(frozen=True)
class Job:
    id: str
    owner: str  # id of the client that created the job (app.api.clients)
    filename: str  # kept only to put in the result; never logged
    size: int
    callback_url: str | None = None
    external_id: str | None = None  # caller's reference, echoed back; never logged
    metadata: dict | None = None  # caller's small opaque JSON object, echoed back; never logged
    request_id: str | None = None  # X-Request-Id of the upload, for log correlation
    options: dict | None = None  # per-request processing options passed to the pipeline, e.g. {"include_boxes": True}
    result_ttl_s: int | None = None  # the profile's retention instead of RESULT_TTL_S
    status: Status = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    attempts: int = 0  # how often a worker claimed it
    result: dict | None = None
    error: dict | None = None
    webhook: dict | None = None  # {"state": pending|delivered|failed, "attempts": n} when a callback is set
    webhook_delivery_id: str | None = None  # the same for every attempt, also across restarts (receivers dedupe)
    webhook_next_at: float | None = None  # the next attempt is due; None = nothing pending
    webhook_settled_at: float | None = None  # delivered or given up: the result is kept RESULT_TTL_S after this


@dataclass(frozen=True)
class AuditEvent:
    """One audit trail entry. `data` is metadata only: never document text, file names, MRZ data or caller
    references in clear (app/api/audit.py builds it)."""

    event: str
    client_id: str | None = None
    job_id: str | None = None
    request_id: str | None = None
    data: dict = field(default_factory=dict)


def started_event(job: Job, worker_id: str) -> AuditEvent:
    return AuditEvent("job.started", job.owner, job.id, job.request_id, {"attempt": job.attempts, "worker": worker_id})


class IdempotencyConflict(Exception):
    """The Idempotency-Key was already used by this API key for a different file."""


class JobStore(ABC):
    def open(self) -> None:
        """Called once at app startup."""

    def close(self) -> None:
        """Called once at app shutdown."""

    def ping(self) -> bool:
        return True

    @abstractmethod
    def add(self, job: Job, data: bytes, idempotency: tuple[str, str] | None = None,
            audit: AuditEvent | None = None) -> Job:
        """Store the job and its upload, and queue it. `idempotency` = (key, sha256 of the file): if this owner
        already used the key, return that job instead and store nothing; IdempotencyConflict if the file differs.
        `audit` is recorded only when the job is stored."""

    @abstractmethod
    def get_idempotent(self, owner: str, key: str) -> tuple[str, str] | None:
        """(job_id, file sha256) previously stored under this owner's key, or None."""

    @abstractmethod
    def get(self, job_id: str) -> Job | None: ...

    @abstractmethod
    def update(self, job_id: str, **changes) -> Job | None:
        """Apply changes; None if the job no longer exists."""

    @abstractmethod
    def finish(self, job_id: str, worker_id: str, audit: AuditEvent | None = None, **changes) -> Job | None:
        """Like update, but only while `worker_id` still holds the job (not deleted, not re-queued meanwhile).
        Reaching done/failed also deletes the upload. `audit` is recorded only when the job was updated."""

    @abstractmethod
    def delete(self, job_id: str, audit: AuditEvent | None = None) -> Job | None:
        """Delete the job, its upload and its idempotency key. `audit` is recorded only if the job existed."""

    @abstractmethod
    def claim(self, timeout: float, worker_id: str) -> Job | None:
        """Take the oldest queued job (status -> processing, attempts + 1), waiting up to `timeout` seconds.
        Records `job.started`."""

    @abstractmethod
    def read_upload(self, job_id: str) -> bytes | None: ...

    @abstractmethod
    def delete_upload(self, job_id: str) -> None: ...

    def heartbeat(self, worker_id: str) -> None:
        """Mark this worker's processing jobs as alive."""

    def requeue_stale(self, stale_s: float, max_attempts: int) -> list[str]:
        """Re-queue jobs whose worker stopped sending heartbeats; fail those over `max_attempts` (and schedule their
        webhook). Returns the ids of the jobs that failed this way. Records `job.requeued` / `job.failed`."""
        return []

    def release(self, worker_id: str) -> int:
        """At shutdown: put this worker's unfinished jobs back in the queue without counting the interrupted
        attempt. Returns how many. Records `job.requeued` {reason: shutdown}."""
        return 0

    @abstractmethod
    def purge_finished_before(self, cutoff: float, now: float | None = None) -> int:
        """Delete jobs finished before `cutoff` (unix time), except those with a webhook still pending or settled
        after `cutoff`; records `job.deleted` {by: ttl}. Returns how many. A job with its own `result_ttl_s` uses
        `now - result_ttl_s` as its cutoff instead (`now`: default the current time)."""

    # --- webhook outbox ------------------------------------------------------------------------------------------
    @abstractmethod
    def claim_webhooks(self, limit: int, lease_s: float) -> list[Job]:
        """Up to `limit` due deliveries (`webhook_next_at` <= now), the longest due first. Each is leased: its
        `webhook_next_at` moves `lease_s` ahead, so no other dispatcher takes it while it is being sent (and it is
        due again if this one dies)."""

    @abstractmethod
    def webhook_attempted(self, job_id: str, delivery_id: str, *, state: str, attempts: int,
                          next_at: float | None, audit: AuditEvent | None = None) -> Job | None:
        """Write a delivery outcome: `pending` with the next attempt at `next_at`, or `delivered` / `failed` (settled:
        nothing more is sent). `audit` is recorded with it. None if the job or that delivery no longer exists."""

    @abstractmethod
    def count_pending_webhooks(self) -> int: ...

    @abstractmethod
    def count_queued(self) -> int: ...

    @abstractmethod
    def count_queued_for(self, owner: str) -> int:
        """Queued jobs of one client (for its MAX_QUEUED_JOBS quota)."""

    @abstractmethod
    def counts(self) -> dict[str, int]: ...

    @abstractmethod
    def ages(self) -> dict[str, float]:
        """Seconds the oldest queued job has waited (`queued`) and the longest-running job has been processing
        (`processing`); 0 when there is none. For monitoring."""

    # --- audit trail ---------------------------------------------------------------------------------------------
    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        """Append one audit event."""

    @abstractmethod
    def audit_events(self, *, job_id: str | None = None, request_id: str | None = None, client_id: str | None = None,
                     since: float | None = None, limit: int = 1000) -> list[dict]:
        """Events matching every given filter, oldest first:
        {id, at (unix time), event, clientId, jobId, requestId, data}."""

    @abstractmethod
    def purge_events_before(self, cutoff: float) -> int:
        """Delete audit events older than `cutoff` (unix time). Returns how many."""


class InMemoryJobStore(JobStore):
    def __init__(self, upload_dir: Path | str | None = None):
        self.upload_dir = Path(upload_dir) if upload_dir else Path(tempfile.gettempdir()) / "ocr-id-service"
        self._jobs: dict[str, Job] = {}
        self._idem: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._events: deque[dict] = deque(maxlen=MEMORY_AUDIT_EVENTS)
        self._event_ids = itertools.count(1)

    def open(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        for leftover in self.upload_dir.glob("*.upload"):  # from a crash: never keep documents around
            leftover.unlink(missing_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.upload_dir / f"{job_id}.upload"  # random name: the original name never hits disk

    def _record_locked(self, ev: AuditEvent) -> None:
        self._events.append({"id": next(self._event_ids), "at": time.time(), "event": ev.event,
                             "clientId": ev.client_id, "jobId": ev.job_id, "requestId": ev.request_id,
                             "data": dict(ev.data)})

    def add(self, job: Job, data: bytes, idempotency: tuple[str, str] | None = None,
            audit: AuditEvent | None = None) -> Job:
        with self._lock:
            if idempotency:
                hit = self._idem.get((job.owner, idempotency[0]))
                if hit and hit[0] in self._jobs:
                    if hit[1] != idempotency[1]:
                        raise IdempotencyConflict
                    return self._jobs[hit[0]]
            fd = os.open(self._path(job.id), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            self._jobs[job.id] = job
            if idempotency:
                self._idem[(job.owner, idempotency[0])] = (job.id, idempotency[1])
            if audit:
                self._record_locked(audit)
        self._queue.put(job.id)
        return job

    def get_idempotent(self, owner: str, key: str) -> tuple[str, str] | None:
        with self._lock:
            hit = self._idem.get((owner, key))
            return hit if hit and hit[0] in self._jobs else None

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **changes) -> Job | None:
        with self._lock:
            return self._update_locked(job_id, changes)

    def _update_locked(self, job_id: str, changes: dict) -> Job | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        self._jobs[job_id] = job = replace(job, **changes)
        return job

    def finish(self, job_id: str, worker_id: str, audit: AuditEvent | None = None, **changes) -> Job | None:
        with self._lock:  # one process: no other worker can hold the job
            job = self._update_locked(job_id, changes)
            if job is not None and audit:
                self._record_locked(audit)
        if job is not None and job.status in FINAL:
            self.delete_upload(job_id)  # a finished job never keeps its document
        return job

    def _forget(self, job_id: str) -> Job | None:
        job = self._jobs.pop(job_id, None)
        if job is not None:
            self._idem = {k: v for k, v in self._idem.items() if v[0] != job_id}
        return job

    def delete(self, job_id: str, audit: AuditEvent | None = None) -> Job | None:
        with self._lock:
            job = self._forget(job_id)
            if job is not None and audit:
                self._record_locked(audit)
        self.delete_upload(job_id)
        return job

    def claim(self, timeout: float, worker_id: str) -> Job | None:
        end = time.monotonic() + timeout
        while (left := end - time.monotonic()) > 0:
            try:
                job_id = self._queue.get(timeout=left)
            except queue.Empty:
                return None
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None and job.status == "queued":  # else: deleted while queued
                    self._jobs[job_id] = job = replace(job, status="processing", started_at=time.time(),
                                                       attempts=job.attempts + 1)
                    self._record_locked(started_event(job, worker_id))
                    return job
        return None

    def read_upload(self, job_id: str) -> bytes | None:
        try:
            return self._path(job_id).read_bytes()
        except FileNotFoundError:
            return None

    def delete_upload(self, job_id: str) -> None:
        try:
            self._path(job_id).unlink(missing_ok=True)
        except OSError:  # Windows: still open by a reader; the worker deletes it after processing anyway
            pass

    def purge_finished_before(self, cutoff: float, now: float | None = None) -> int:
        now = time.time() if now is None else now

        def expired(j: Job) -> bool:
            limit = cutoff if j.result_ttl_s is None else now - j.result_ttl_s
            return (j.finished_at is not None and j.finished_at < limit and j.webhook_next_at is None
                    and (j.webhook_settled_at is None or j.webhook_settled_at < limit))

        with self._lock:
            old = [j for j in self._jobs.values() if expired(j)]
            for job in old:
                self._forget(job.id)
                self._record_locked(AuditEvent("job.deleted", job.owner, job.id, job.request_id, {"by": "ttl"}))
            return len(old)

    def claim_webhooks(self, limit: int, lease_s: float) -> list[Job]:
        now = time.time()
        with self._lock:
            due = sorted((j for j in self._jobs.values() if j.webhook_next_at is not None and j.webhook_next_at <= now),
                         key=lambda j: j.webhook_next_at)[:max(0, limit)]
            for j in due:
                self._jobs[j.id] = replace(j, webhook_next_at=now + lease_s)
            return due

    def webhook_attempted(self, job_id: str, delivery_id: str, *, state: str, attempts: int,
                          next_at: float | None, audit: AuditEvent | None = None) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.webhook_delivery_id != delivery_id:
                return None
            settled = state != "pending"
            self._jobs[job_id] = job = replace(job, webhook={"state": state, "attempts": attempts},
                                               webhook_next_at=None if settled else next_at,
                                               webhook_settled_at=time.time() if settled else None)
            if audit:
                self._record_locked(audit)
            return job

    def count_pending_webhooks(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.webhook_next_at is not None)

    def count_queued(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued")

    def count_queued_for(self, owner: str) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "queued" and j.owner == owner)

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {"queued": 0, "processing": 0, "done": 0, "failed": 0}
            for j in self._jobs.values():
                out[j.status] += 1
            return out

    def ages(self) -> dict[str, float]:
        now = time.time()
        with self._lock:
            queued = [j.created_at for j in self._jobs.values() if j.status == "queued"]
            running = [j.started_at for j in self._jobs.values() if j.status == "processing" and j.started_at]
        return {"queued": now - min(queued) if queued else 0.0, "processing": now - min(running) if running else 0.0}

    def record(self, event: AuditEvent) -> None:
        with self._lock:
            self._record_locked(event)

    def audit_events(self, *, job_id: str | None = None, request_id: str | None = None, client_id: str | None = None,
                     since: float | None = None, limit: int = 1000) -> list[dict]:
        with self._lock:
            hits = [dict(e, data=dict(e["data"])) for e in self._events
                    if (job_id is None or e["jobId"] == job_id) and (request_id is None or e["requestId"] == request_id)
                    and (client_id is None or e["clientId"] == client_id) and (since is None or e["at"] >= since)]
        return hits[:limit]

    def purge_events_before(self, cutoff: float) -> int:
        with self._lock:
            keep = [e for e in self._events if e["at"] >= cutoff]
            n = len(self._events) - len(keep)
            self._events.clear()
            self._events.extend(keep)
            return n


def make_store(settings) -> JobStore:
    if settings.job_store == "postgres":
        from app.api.sealer import Sealer
        from app.api.store_pg import PostgresJobStore

        return PostgresJobStore(settings.database_url, Sealer(settings.result_encryption_key),
                                pool_size=settings.db_pool_size, poll_s=settings.db_poll_s)
    return InMemoryJobStore(settings.upload_dir)
