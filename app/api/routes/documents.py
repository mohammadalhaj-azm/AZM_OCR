"""POST /v1/documents: upload a document for text extraction + ID detection (async job)."""

import hashlib
import json
import logging
import re
import uuid

from fastapi import APIRouter, Depends, File, Form, Header, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api import audit, webhook
from app.api.clients import Client, ClientRegistry
from app.api.deps import app_clients, app_settings, app_store, read_upload, require_scope, retry_after, use_profile
from app.api.errors import ERROR_RESPONSES, ApiError
from app.api.models import JobCreated
from app.api.store import AuditEvent, IdempotencyConflict, Job, JobStore
from app.config import Settings
from app.profiles import Profile

log = logging.getLogger("app.api")
router = APIRouter()
IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9._:-]{1,128}")
MAX_EXTERNAL_ID = 128
MAX_METADATA_BYTES = 2048


@router.post("/v1/documents", status_code=202, response_model=JobCreated, responses=ERROR_RESPONSES,
             summary="Upload a document (PDF, DOCX, JPG, PNG, WEBP, TIFF) for text extraction + ID detection "
                     "(async job)")
async def upload(request: Request, response: Response, file: UploadFile = File(...),
                 callbackUrl: str | None = Form(None, description="Webhook URL (host must be allow-listed)"),
                 externalId: str | None = Form(None, description="Your reference, echoed in the job and webhook "
                                                                 f"(at most {MAX_EXTERNAL_ID} characters)"),
                 metadata: str | None = Form(None, description="JSON object echoed in the job and webhook "
                                                               f"(at most {MAX_METADATA_BYTES} bytes)"),
                 includeBoxes: bool | None = Form(None, description="Add `lines` [{text, confidence, bbox}] and "
                                                                    "the image size to every OCR page (default: the "
                                                                    "profile's, normally false)"),
                 profile: str | None = Form(None, description="Processing profile: one of your client's "
                                                              "(default: its first). Changes processing, output "
                                                              "and retention; see `processing.profile`"),
                 idempotency_key: str | None = Header(None, alias="Idempotency-Key",
                                                      description="Retry-safe uploads: the same key returns "
                                                                  "the same job (within RESULT_TTL_S)"),
                 client: Client = Depends(require_scope("documents")), settings: Settings = Depends(app_settings),
                 store: JobStore = Depends(app_store), clients: ClientRegistry = Depends(app_clients)):
    prof = use_profile(clients, client, profile)
    return await queue_job(request, response, file, client, settings, store, prof, callbackUrl, externalId, metadata,
                           includeBoxes, idempotency_key, endpoint="documents")


async def queue_job(request: Request, response: Response, file: UploadFile, client: Client, settings: Settings,
                    store: JobStore, prof: Profile, callbackUrl: str | None, externalId: str | None,
                    metadata: str | None, includeBoxes: bool | None, idempotency_key: str | None, endpoint: str,
                    task_options: dict | None = None, task_facts: dict | None = None) -> dict:
    """Check the request, read the upload and store the job (or replay an idempotent one): {"jobId"}.

    task_options: more job options for the pipeline (e.g. {"task": "extract_and_answer", "answer": {...}}); they are
    part of the idempotency fingerprint, so the same key with other options is a conflict. task_facts: metadata
    (never content) for the job.received audit event."""
    owner = client.id
    include_boxes = prof.include_boxes if includeBoxes is None else includeBoxes
    if callbackUrl:
        try:
            webhook.check_callback_url(callbackUrl, client.webhook_hosts, client.webhook_secret)
        except webhook.CallbackNotAllowed as e:
            raise ApiError(400, "callback_not_allowed", str(e)) from None
    if externalId is not None and not (0 < len(externalId) <= MAX_EXTERNAL_ID):
        raise ApiError(422, "invalid_request", f"externalId must be 1-{MAX_EXTERNAL_ID} characters")
    meta = None
    if metadata is not None:
        try:
            meta = json.loads(metadata) if len(metadata.encode()) <= MAX_METADATA_BYTES else None
        except ValueError:
            meta = None
        if not isinstance(meta, dict):
            raise ApiError(422, "invalid_request",
                           f"metadata must be a JSON object of at most {MAX_METADATA_BYTES} bytes")
    if idempotency_key is not None and not IDEMPOTENCY_KEY.fullmatch(idempotency_key):
        raise ApiError(422, "invalid_request", "Idempotency-Key must be 1-128 characters of [A-Za-z0-9._:-]")

    async def replay(job: Job, sha: str) -> dict:
        response.headers["Idempotent-Replayed"] = "true"
        log.info("job replayed job=%s client=%s req=%s", job.id, owner, request.state.request_id)
        await run_in_threadpool(audit.record_safely, store, AuditEvent(
            audit.REPLAYED, owner, job.id, request.state.request_id, {"sha256": sha}), request.app.state.metrics)
        return {"jobId": job.id}

    known = await run_in_threadpool(store.get_idempotent, owner, idempotency_key) if idempotency_key else None
    if known is None:  # a replay is always allowed, also when the queue is full
        if await run_in_threadpool(store.count_queued) >= settings.max_queued_jobs:
            raise ApiError(503, "queue_full", "queue is full, retry later", retry_after(settings))
        if (client.max_queued_jobs is not None
                and await run_in_threadpool(store.count_queued_for, owner) >= client.max_queued_jobs):
            raise ApiError(503, "queue_full", "this API key's queue is full, retry later", retry_after(settings))

    data, mime = await read_upload(file, settings)
    sha = hashlib.sha256(data).hexdigest()  # of the file (audit trail)
    idem = sha  # what an Idempotency-Key is bound to: the file, plus the task's options (e.g. other questions)
    if task_options:
        fingerprint = json.dumps(task_options, sort_keys=True, ensure_ascii=False).encode()
        idem = hashlib.sha256(sha.encode() + b"\0" + fingerprint).hexdigest()
    if known is not None:
        existing = await run_in_threadpool(store.get, known[0])
        if known[1] != idem:
            raise ApiError(409, "idempotency_conflict", "Idempotency-Key was already used for another file"
                                                        + (" or other options" if task_options else ""))
        if existing is not None:
            return await replay(existing, sha)

    rid = request.state.request_id
    options = {"include_boxes": True} if include_boxes else {}
    if not prof.is_default:
        options["profile"] = prof.to_job()  # copied into the job: processed the same way on any replica
    options.update(task_options or {})
    job = Job(id=uuid.uuid4().hex, owner=owner, filename=file.filename or "", size=len(data),
              callback_url=callbackUrl, external_id=externalId, metadata=meta, request_id=rid,
              options=options or None, result_ttl_s=prof.result_ttl_s)
    facts = {"endpoint": endpoint, **audit.file_facts(data, mime, sha), "callback": bool(callbackUrl),
             "includeBoxes": include_boxes, "idempotent": idempotency_key is not None, "profile": prof.name,
             **(task_facts or {})}
    if externalId:
        facts["externalIdHash"] = request.app.state.ref_hash(externalId)  # never the reference itself
    received = AuditEvent(audit.RECEIVED, owner, job.id, rid, facts)  # stored with the job, or not at all
    try:
        stored = await run_in_threadpool(store.add, job, data, (idempotency_key, idem) if idempotency_key else None,
                                         received)
    except IdempotencyConflict:
        raise ApiError(409, "idempotency_conflict", "Idempotency-Key was already used for another file") from None
    if stored.id != job.id:  # a concurrent request with the same key won
        return await replay(stored, sha)
    log.info("job queued job=%s client=%s bytes=%d req=%s", job.id, owner, job.size, rid)
    return {"jobId": job.id}
