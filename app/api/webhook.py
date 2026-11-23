"""Signed completion webhooks, delivered from the job store (an outbox).

Headers: X-Webhook-Id: <deliveryId>, X-Webhook-Timestamp: <unix seconds>, X-Webhook-Signature: sha256=<hex>, where the
signature is HMAC-SHA256(secret, "<timestamp>." + raw_body) with the job's client's secret. Receivers should check it,
reject old timestamps, and de-duplicate on the delivery id (every attempt reuses it, also after a restart).

When a job finishes, its delivery is written to the job row (`webhook_next_at` = now) in the same update. A
`Dispatcher` in every process claims due deliveries from the store (`FOR UPDATE SKIP LOCKED` with PostgreSQL, so any
replica sends any delivery exactly once at a time), sends them on a small thread pool, and writes the outcome back:
delivered, failed for good, or the next attempt (exponential backoff). Nothing lives only in memory, so retries
survive restarts and deploys. Job processing never waits for a receiver.
"""

import hashlib
import hmac
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

import httpx

from app import logs
from app.api import audit
from app.api.clients import ClientRegistry
from app.api.store import AuditEvent, Job, JobStore
from app.config import Settings

log = logging.getLogger(__name__)
POLL_S = 1.0  # idle dispatchers look for due deliveries this often (a local job finishing wakes them at once)
LEASE_MARGIN_S = 30.0  # a claimed delivery is due again after WEBHOOK_TIMEOUT_S + this if its sender died


class CallbackNotAllowed(ValueError):
    pass


def check_callback_url(url: str, allowed_hosts: set[str] | frozenset[str], secret: str) -> str:
    """`allowed_hosts` and `secret` are the calling client's (see app.api.clients)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CallbackNotAllowed("callbackUrl must be an http(s) URL")
    if not secret or not allowed_hosts:
        raise CallbackNotAllowed("callbacks are not enabled for this API key")
    if parts.hostname.lower() not in allowed_hosts:
        raise CallbackNotAllowed("callbackUrl host is not allowed for this API key")
    return url


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def deliver_once(client: httpx.Client, url: str, body: bytes, secret: str, delivery_id: str, job_id: str,
                 attempt: int) -> bool:
    """One signed POST. True on a 2xx response."""
    ts = str(int(time.time()))
    headers = {"Content-Type": "application/json", "X-Webhook-Id": delivery_id, "X-Webhook-Timestamp": ts,
               "X-Webhook-Signature": sign(secret, ts, body)}
    try:
        r = client.post(url, content=body, headers=headers)
        if r.status_code < 300:
            log.info("webhook delivered job=%s attempt=%d status=%d", job_id, attempt, r.status_code)
            return True
        log.warning("webhook rejected job=%s attempt=%d status=%d", job_id, attempt, r.status_code)
    except httpx.HTTPError as e:
        log.warning("webhook error job=%s attempt=%d error=%s", job_id, attempt, type(e).__name__)
    return False


def payload(job: Job) -> dict:
    """The callback body. No document data: the receiver fetches the result with its API key."""
    body: dict = {"event": f"job.{job.status}", "deliveryId": job.webhook_delivery_id, "jobId": job.id,
                  "status": job.status}
    if job.external_id:
        body["externalId"] = job.external_id
    if job.metadata:
        body["metadata"] = job.metadata
    if job.error:
        body["error"] = {"code": job.error["code"]}
    return body


def backoff(settings: Settings, attempt: int) -> float:
    """Seconds before the next attempt after `attempt` failed ones: WEBHOOK_BACKOFF_S, doubling, capped."""
    return min(settings.webhook_backoff_s * 2 ** (attempt - 1), settings.webhook_max_backoff_s)


class Dispatcher:
    def __init__(self, store: JobStore, http: httpx.Client, clients: ClientRegistry, settings: Settings,
                 metrics=None):
        self.store, self.http, self.clients, self.s, self.metrics = store, http, clients, settings, metrics
        self.threads = max(1, settings.webhook_threads)
        self._pool = ThreadPoolExecutor(self.threads, thread_name_prefix="webhook")
        self._free = threading.Semaphore(self.threads)  # one per delivery in flight
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="webhooks", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 15) -> None:
        """Stop claiming, and wait for the deliveries in flight (each is bounded by WEBHOOK_TIMEOUT_S). Pending ones
        stay in the store for the next start or another replica."""
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)
        self._pool.shutdown(wait=True, cancel_futures=True)

    def wake(self) -> None:
        """A delivery became due (a job of this process finished): look now instead of at the next poll."""
        self._wake.set()

    def pending(self) -> int:
        return self.store.count_pending_webhooks()

    def _loop(self) -> None:
        while not self._stop.is_set():
            free = 0
            while free < self.threads and self._free.acquire(blocking=False):
                free += 1
            jobs: list[Job] = []
            if free:
                try:
                    jobs = self.store.claim_webhooks(free, self.s.webhook_timeout_s + LEASE_MARGIN_S)
                except Exception as e:  # noqa: BLE001 - e.g. database briefly unreachable: keep the thread alive
                    log.error("webhook claim failed error=%s", type(e).__name__)
            for _ in range(free - len(jobs)):
                self._free.release()
            for job in jobs:
                self._pool.submit(self._run, job)
            if not jobs:
                self._wake.wait(POLL_S)
                self._wake.clear()

    def _run(self, job: Job) -> None:
        try:
            with logs.context(job_id=job.id, request_id=job.request_id, client_id=job.owner):
                self.deliver(job)
        except Exception as e:  # noqa: BLE001 - the delivery stays leased and is retried after the lease
            log.error("webhook delivery failed job=%s error=%s", job.id, type(e).__name__)
        finally:
            self._free.release()
            self._wake.set()  # a slot is free again

    def deliver(self, job: Job) -> None:
        """One attempt for a claimed delivery, then write the outcome."""
        attempt = (job.webhook or {}).get("attempts", 0) + 1
        # Checked again at every attempt: the client may have been disabled or lost the host since the upload.
        client = self.clients.get(job.owner)
        try:
            if client is None or not client.enabled:
                raise CallbackNotAllowed("client unknown or disabled")
            check_callback_url(job.callback_url or "", client.webhook_hosts, client.webhook_secret)
        except CallbackNotAllowed:
            log.warning("webhook not sent job=%s client=%s reason=not_allowed", job.id, job.owner)
            self._settle(job, "failed", attempt - 1, "not_allowed")
            return
        body = json.dumps(payload(job), separators=(",", ":")).encode()
        if deliver_once(self.http, job.callback_url, body, client.webhook_secret, job.webhook_delivery_id, job.id,
                        attempt):
            self._settle(job, "delivered", attempt)
            return
        delay = backoff(self.s, attempt)
        window = client.webhook_retry_window_s if client.webhook_retry_window_s is not None \
            else self.s.webhook_retry_window_s
        out_of_attempts = self.s.webhook_retries > 0 and attempt >= self.s.webhook_retries
        out_of_time = time.time() + delay >= (job.finished_at or time.time()) + window  # window 0: no retries
        if out_of_attempts or out_of_time:
            self._settle(job, "failed", attempt, "gave_up")
        else:
            self.store.webhook_attempted(job.id, job.webhook_delivery_id, state="pending", attempts=attempt,
                                         next_at=time.time() + delay)

    def _settle(self, job: Job, state: str, attempts: int, reason: str | None = None) -> None:
        data = {"attempts": attempts, **({"reason": reason} if reason else {})}
        event = AuditEvent(audit.WEBHOOK_DELIVERED if state == "delivered" else audit.WEBHOOK_FAILED, job.owner,
                           job.id, job.request_id, data)
        self.store.webhook_attempted(job.id, job.webhook_delivery_id, state=state, attempts=attempts, next_at=None,
                                     audit=event)  # the outcome and its audit event together
        if self.metrics is not None:
            self.metrics.webhooks.labels(state, job.owner).inc()
