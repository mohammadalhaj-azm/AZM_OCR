"""Background worker: claims queued jobs, runs the pipeline, and schedules each job's webhook with its result.

With LLM_WORKER_THREADS > 0 a job runs in two stages on different threads: WORKER_THREADS threads claim jobs
and extract text (OCR), then hand the extraction to LLM_WORKER_THREADS threads for ID detection. The handoff
queue holds at most LLM_WORKER_THREADS items, so OCR never runs far ahead of the LLM.

A separate maintenance thread sends heartbeats for the jobs being processed, re-queues jobs of dead workers
(another replica, or this process before a crash) and purges expired results.

At shutdown (`stop`), jobs still being processed go back to the queue without counting the interrupted attempt, so a
deploy never pushes a job towards `worker_lost`.
"""

import logging
import os
import queue
import threading
import time
import uuid
from collections.abc import Callable

from app import logs
from app.api import audit
from app.api.metrics import Metrics
from app.api.store import AuditEvent, Job, JobStore
from app.api.webhook import Dispatcher
from app.config import Settings
from app.extractors import FileRejected

log = logging.getLogger(__name__)

AnalyzeFn = Callable[..., dict]  # analyze(data, filename, services, *, detect_id=True, max_pages=None, ...)
# Optional attribute `analyze_fn.stages = (extract_stage, detect_stage)` enables the staged mode (app.pipeline).
SWEEP_EVERY_S = 30


def job_context(job: Job):
    """The job's ids for the log lines written while working on it: its id, its upload's request, its client."""
    return logs.context(job_id=job.id, request_id=job.request_id, client_id=job.owner)


class Worker:
    def __init__(self, settings: Settings, store: JobStore, analyze_fn: AnalyzeFn, services,
                 dispatcher: Dispatcher | None = None, metrics: Metrics | None = None):
        self.s, self.store, self.analyze_fn, self.services = settings, store, analyze_fn, services
        self.dispatcher, self.metrics = dispatcher, metrics  # the dispatcher is woken when a webhook becomes due
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_sweep = 0.0
        stages = getattr(analyze_fn, "stages", None)
        self.staged = settings.llm_worker_threads > 0 and stages is not None
        self._extract_fn, self._detect_fn = stages if self.staged else (None, None)
        self._handoff: queue.Queue = queue.Queue(maxsize=max(1, settings.llm_worker_threads))
        self._active: dict[str, dict] = {}  # jobs this process is working on now (admin dashboard)
        self._active_lock = threading.Lock()
        self.started_at = time.time()

    def active(self) -> list[dict]:
        """The jobs this process is working on: [{jobId, clientId, stage, since, attempt}] (metadata only)."""
        with self._active_lock:
            return sorted((dict(a) for a in self._active.values()), key=lambda a: a["since"])

    def _set_active(self, job: Job, stage: str) -> None:
        with self._active_lock:
            a = self._active.setdefault(job.id, {"jobId": job.id, "clientId": job.owner, "attempt": job.attempts,
                                                 "since": time.time(), "requestId": job.request_id})
            a["stage"], a["stageSince"] = stage, time.time()

    def start(self) -> None:
        for i in range(max(1, self.s.worker_threads)):
            t = threading.Thread(target=self._loop, name=f"worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        for i in range(self.s.llm_worker_threads if self.staged else 0):
            t = threading.Thread(target=self._detect_loop, name=f"worker-llm-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        m = threading.Thread(target=self._maintenance, name="worker-maintenance", daemon=True)
        m.start()
        self._threads.append(m)

    def stop(self, timeout: float = 5) -> None:
        """Stop claiming, give running jobs `timeout` seconds to finish, then release the rest to the queue."""
        self._stop.set()
        end = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, end - time.monotonic()))
        try:
            n = self.store.release(self.worker_id)  # a late finish() from a thread still running is then a no-op
            if n:
                log.info("jobs released at shutdown count=%d worker=%s", n, self.worker_id)
        except Exception as e:  # noqa: BLE001 - stale-job recovery re-queues them later instead
            log.error("release at shutdown failed error=%s", type(e).__name__)

    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads if t.name != "worker-maintenance")

    def _loop(self) -> None:
        if self.s.ocr_warmup and hasattr(self.services, "engine"):
            try:
                self.services.engine.load()
            except Exception as e:  # noqa: BLE001 - reported per job later
                log.error("ocr warmup failed error=%s", type(e).__name__)
        while not self._stop.is_set():
            try:
                job = self.store.claim(timeout=1.0, worker_id=self.worker_id)
            except Exception as e:  # noqa: BLE001 - e.g. database briefly unreachable: keep the thread alive
                log.error("claim failed error=%s", type(e).__name__)
                self._stop.wait(2)
                continue
            if job is None:
                continue
            if self.metrics and job.started_at is not None:
                self.metrics.queue_wait.observe(max(0.0, job.started_at - job.created_at))
            self._set_active(job, "ocr" if self.staged else "processing")
            with job_context(job):  # every log line of this job carries its ids (JSON logs)
                self._extract(job) if self.staged else self.process(job)

    def _maintenance(self) -> None:
        beat = max(1.0, min(30.0, self.s.job_stale_s / 4))
        while not self._stop.wait(beat):
            try:
                self.store.heartbeat(self.worker_id)
                if time.monotonic() - self._last_sweep > SWEEP_EVERY_S:
                    self.sweep()
            except Exception as e:  # noqa: BLE001
                log.error("maintenance failed error=%s", type(e).__name__)

    def sweep(self) -> int:
        """Purge expired results and recover stale jobs. Returns the number of purged jobs."""
        self._last_sweep = time.monotonic()
        if self.store.requeue_stale(self.s.job_stale_s, self.s.job_max_attempts) and self.dispatcher:
            self.dispatcher.wake()  # jobs failed with worker_lost: their webhooks are scheduled
        n = self.store.purge_finished_before(time.time() - self.s.result_ttl_s)
        if n:
            log.info("purged expired jobs count=%d", n)
        if self.s.audit_retention_days > 0:
            old = self.store.purge_events_before(time.time() - self.s.audit_retention_days * 86400)
            if old:
                log.info("purged audit events count=%d", old)
        return n

    def _read(self, job: Job) -> bytes | None:
        data = self.store.read_upload(job.id)
        if data is None:  # deleted after it was claimed (finish is then a no-op)
            self._complete(job, "failed", None, {"code": "internal_error", "message": "upload missing"},
                           time.perf_counter())
        return data

    def process(self, job: Job) -> None:
        """Single-stage mode: the whole pipeline on this thread."""
        t0 = time.perf_counter()
        status, result, error, fallbacks = "failed", None, None, None
        try:
            data = self._read(job)
            if data is None:
                return
            with audit.collect_fallbacks() as fallbacks:  # models ID detection had to go on without
                result = self.analyze_fn(data, job.filename, self.services, **(job.options or {}))
            status = "done"
        except Exception as e:  # noqa: BLE001
            error = self._error(job, e)
        self._complete(job, status, result, error, t0, fallbacks if status == "done" else None)

    def _extract(self, job: Job) -> None:
        """Staged mode, stage 1 (OCR thread): extract, then hand over to an LLM thread."""
        t0 = time.perf_counter()
        try:
            data = self._read(job)
            if data is None:
                return
            extracted = self._extract_fn(data, job.filename, self.services, **(job.options or {}))
        except Exception as e:  # noqa: BLE001
            self._complete(job, "failed", None, self._error(job, e), t0)
            return
        item = (job, extracted, t0, time.perf_counter())
        self._set_active(job, "llm_queue")
        while not self._stop.is_set():  # blocks while all LLM threads are busy (backpressure)
            try:
                self._handoff.put(item, timeout=1.0)
                return
            except queue.Full:
                continue
        # Stopping: the job stays "processing" and is re-queued by stale-job recovery (postgres store).

    def _detect_loop(self) -> None:
        """Staged mode, stage 2 (LLM thread)."""
        while not self._stop.is_set():
            try:
                job, extracted, t0, handed_at = self._handoff.get(timeout=1.0)
            except queue.Empty:
                continue
            waited_ms = round((time.perf_counter() - handed_at) * 1000)
            status, result, error, fallbacks = "failed", None, None, None
            self._set_active(job, "detect")
            with job_context(job):
                try:
                    with audit.collect_fallbacks() as fallbacks:
                        result = self._detect_fn(extracted, self.services)
                    result.setdefault("timingsMs", {})["llmQueue"] = waited_ms  # OCR done, waiting for an LLM thread
                    status = "done"
                except Exception as e:  # noqa: BLE001
                    error = self._error(job, e)
                self._complete(job, status, result, error, t0, fallbacks if status == "done" else None)

    @staticmethod
    def _error(job: Job, e: Exception) -> dict:
        if isinstance(e, FileRejected):
            return {"code": e.code, "message": str(e)}
        # never leak details: they may contain document content
        log.error("job failed job=%s error=%s", job.id, type(e).__name__)
        return {"code": "internal_error", "message": "processing failed"}

    def _complete(self, job: Job, status: str, result: dict | None, error: dict | None, t0: float,
                  fallbacks: list[str] | None = None) -> None:
        """Store the outcome (this also deletes the upload) and schedule the webhook, then log and count."""
        with self._active_lock:
            self._active.pop(job.id, None)
        ms = round((time.perf_counter() - t0) * 1000)
        changes: dict = {"status": status, "result": result, "error": error, "finished_at": time.time()}
        if job.callback_url:  # schedule the webhook with the result: it can't get lost between the two
            changes.update(webhook={"state": "pending", "attempts": 0}, webhook_delivery_id=uuid.uuid4().hex,
                           webhook_next_at=changes["finished_at"])
        facts = {"ms": ms, "attempt": job.attempts, **(audit.result_facts(result) if result else {})}
        if error:
            facts["error"] = error["code"]
        if fallbacks:
            facts["fallbacks"] = list(fallbacks)
        event = AuditEvent(audit.DONE if status == "done" else audit.FAILED, job.owner, job.id, job.request_id, facts)
        try:
            updated = self.store.finish(job.id, self.worker_id, audit=event, **changes)  # with its audit event
        except Exception as e:  # noqa: BLE001 - the job stays processing; stale recovery retries it
            log.error("finish failed job=%s error=%s", job.id, type(e).__name__)
            return
        log.info("job finished job=%s client=%s status=%s bytes=%d ms=%d req=%s", job.id, job.owner, status,
                 job.size, ms, job.request_id or "-")
        if self.metrics:
            self.metrics.jobs.labels(status, job.owner).inc()
            self.metrics.job_seconds.observe(ms / 1000)
            if result:
                self.metrics.observe_result(result)
        if updated is not None and job.callback_url and self.dispatcher:  # None: deleted, or re-queued elsewhere
            self.dispatcher.wake()
