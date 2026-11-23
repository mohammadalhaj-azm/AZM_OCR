"""Audit trail: who sent which document when, what happened to it, and who read or deleted the result.

Events (table `ocr_audit_events` with PostgreSQL; the latest 10 000 in RAM with the memory store):

| event | when | data |
|---|---|---|
| job.received | POST /v1/documents stored a job | endpoint, size, sha256, mimeType, callback, includeBoxes, |
| | | idempotent, externalIdHash |
| job.replayed | the same Idempotency-Key + file came again | sha256 |
| job.started | a worker claimed the job | attempt, worker |
| job.done / job.failed | processing ended | ms, attempt, result facts (below) / error |
| job.requeued | its worker died; the job is queued again | reason, attempts |
| job.fetched | GET /v1/jobs/{id} returned the finished job | status |
| job.deleted | DELETE /v1/jobs/{id}, or the RESULT_TTL_S purge | by (client / ttl) |
| webhook.delivered / webhook.failed | the callback ended | attempts, reason |
| extract.done | POST /v1/extract returned text | endpoint, size, sha256, mimeType, ms, result facts |
| handwriting.done | POST /v1/handwriting (experimental) returned text | endpoint, size, sha256, mimeType, ms, pages, |
| | | skipped, warnings, languages, timingsMs, model |
| request.error | any /v1 request answered with an error (4xx/5xx) | method, route, status, code |
| admin.* | an action in the admin dashboard (login, config change, key, ...) | actor, and what changed (never |
| | | passwords, keys or secrets) |

Every event has `at`, `client_id` (NULL for an unknown key), `job_id` and `request_id` (the X-Request-Id).
Result facts: pages, pageMethods, languages, timingsMs, processing (versions and models), ocrConfidence,
fallbacks, and for ID detection isId, docType, confidence, decidedBy, mrzFound, mrzValid.

**Never stored:** document text, file names, MRZ fields, the LLM's free-text reasons (they can quote the document),
caller metadata, or `externalId` in clear (only an HMAC, see `ref_hasher`).

Job lifecycle events are written by the store in the same transaction as the state change, so they can't be lost.
The other events are written best effort (`record_safely`): a failed write is logged and counted
(`ocr_audit_write_failures_total`), and never fails the request.
"""

import base64
import hashlib
import hmac
import logging
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.api.store import AuditEvent, JobStore
from app.config import Settings

log = logging.getLogger(__name__)

RECEIVED, REPLAYED, STARTED = "job.received", "job.replayed", "job.started"
DONE, FAILED, REQUEUED = "job.done", "job.failed", "job.requeued"
FETCHED, DELETED = "job.fetched", "job.deleted"
WEBHOOK_DELIVERED, WEBHOOK_FAILED = "webhook.delivered", "webhook.failed"
EXTRACT_DONE, REQUEST_ERROR = "extract.done", "request.error"
HANDWRITING_DONE = "handwriting.done"
# Admin dashboard actions (app/admin); `data.actor` is the admin user name
ADMIN_LOGIN, ADMIN_LOGIN_FAILED, ADMIN_LOGOUT = "admin.login", "admin.login_failed", "admin.logout"
ADMIN_CONFIG_SAVED, ADMIN_KEY_CREATED = "admin.config_saved", "admin.key_created"
ADMIN_KEY_REVOKED = "admin.key_revoked"
ADMIN_SECRET_CREATED, ADMIN_JOB_DELETED, ADMIN_ALERT_ACKED = ("admin.secret_created", "admin.job_deleted",
                                                             "admin.alert_acked")
ADMIN_EVENTS = frozenset({ADMIN_LOGIN, ADMIN_LOGIN_FAILED, ADMIN_LOGOUT, ADMIN_CONFIG_SAVED, ADMIN_KEY_CREATED,
                          ADMIN_KEY_REVOKED, ADMIN_SECRET_CREATED, ADMIN_JOB_DELETED, ADMIN_ALERT_ACKED})
EVENTS = frozenset({RECEIVED, REPLAYED, STARTED, DONE, FAILED, REQUEUED, FETCHED, DELETED, WEBHOOK_DELIVERED,
                    WEBHOOK_FAILED, EXTRACT_DONE, REQUEST_ERROR, HANDWRITING_DONE}) | ADMIN_EVENTS
_REF_CONTEXT = b"ocr-audit-external-id"


def ref_hasher(settings: Settings) -> Callable[[str], str]:
    """externalId -> the hex value stored in the audit trail (`externalIdHash`).

    HMAC-SHA256 with a key derived from RESULT_ENCRYPTION_KEY, so a guessable reference ("order-123") can't be
    looked up without the key. Without the key (development only) it is a plain SHA-256.
    `uv run python -m app.cli audit-ref <externalId>` prints the value to search for."""
    if not settings.result_encryption_key:
        return lambda ref: hashlib.sha256(ref.encode()).hexdigest()
    key = hmac.new(base64.b64decode(settings.result_encryption_key), _REF_CONTEXT, hashlib.sha256).digest()
    return lambda ref: hmac.new(key, ref.encode(), hashlib.sha256).hexdigest()


_fallbacks: ContextVar[list[str] | None] = ContextVar("audit_fallbacks", default=None)


@contextmanager
def collect_fallbacks() -> Iterator[list[str]]:
    """Collect the models ID detection had to go on without ("llm" / "vision") while the code inside runs, so the
    job's audit event records them (`fallbacks`)."""
    found: list[str] = []
    token = _fallbacks.set(found)
    try:
        yield found
    finally:
        _fallbacks.reset(token)


def note_fallback(stage: str) -> None:
    found = _fallbacks.get()
    if found is not None and stage not in found:
        found.append(stage)


def file_facts(data: bytes, mime: str, sha256: str | None = None) -> dict:
    return {"size": len(data), "sha256": sha256 or hashlib.sha256(data).hexdigest(), "mimeType": mime}


def result_facts(result: dict) -> dict:
    """Metadata of a pipeline result. Never text, names, MRZ fields or the LLM's reasons."""
    pages = result.get("pages") or []
    out: dict = {"pages": len(pages), "pageMethods": dict(Counter(p.get("method", "unknown") for p in pages)),
                 "languages": result.get("languages"), "timingsMs": result.get("timingsMs"),
                 "processing": result.get("processing")}
    confidences = [c for p in pages if isinstance(c := p.get("confidence"), (int, float))]
    if confidences:  # mean OCR confidence of the OCR'd pages (a number only)
        out["ocrConfidence"] = round(sum(confidences) / len(confidences), 3)
    det = result.get("idDetection")
    if det:
        mrz = det.get("mrz")
        out["idDetection"] = {"isId": det.get("isId"), "docType": det.get("docType"),
                              "confidence": det.get("confidence"), "decidedBy": det.get("decidedBy"),
                              "mrzFound": mrz is not None, "mrzValid": bool(mrz and mrz.get("valid"))}
    return {k: v for k, v in out.items() if v is not None}


def request_error(request, status: int, code: str | None, client_id: str | None = None) -> AuditEvent:
    """The `request.error` event of a /v1 request answered with an error. `route` is the route template when the
    request was routed (e.g. /v1/jobs/{job_id}), else its path."""
    route = request.scope.get("route")
    client = getattr(request.state, "client", None)  # set by require_client
    job_id = (request.scope.get("path_params") or {}).get("job_id")
    return AuditEvent(REQUEST_ERROR, client_id or (client.id if client else None),
                      job_id[:64] if isinstance(job_id, str) else None, getattr(request.state, "request_id", None),
                      {"method": request.method, "route": getattr(route, "path", None) or request.url.path[:200],
                       "status": status, "code": code})


def record_safely(store: JobStore, event: AuditEvent, metrics=None) -> None:
    """Best-effort write for request-level events: a failure is logged and counted, never raised."""
    try:
        store.record(event)
    except Exception as e:  # noqa: BLE001 - the audit trail must not break the request it describes
        log.error("audit write failed event=%s job=%s req=%s error=%s", event.event, event.job_id or "-",
                  event.request_id or "-", type(e).__name__)
        if metrics is not None:
            metrics.audit_failures.inc()
