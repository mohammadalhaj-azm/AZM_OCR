"""POST /v1/extract: synchronous text-only extraction (no ID detection, no LLM)."""

import logging
import time
from contextlib import ExitStack

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api import audit
from app.api.clients import Client, ClientRegistry
from app.api.deps import (REJECT_STATUS, app_clients, app_settings, app_store, read_upload, require_scope,
                          retry_after, use_profile)
from app.api.errors import ERROR_RESPONSES, ApiError
from app.api.models import ExtractResult
from app.api.store import AuditEvent, JobStore
from app.config import Settings
from app.extractors.filetype import FileRejected

log = logging.getLogger("app.api")
router = APIRouter()


def _take_slots(stack: ExitStack, request: Request, client: Client, settings: Settings) -> None:
    """The client's own EXTRACT_MAX_CONCURRENT slot (if it has a limit), then one of the global ones; else 503."""
    state = request.app.state
    for slots, message in ((state.client_extract_slots.get(client.id), "too many simultaneous requests for this "
                                                                       "API key, retry later"),
                           (state.extract_slots, "busy, retry later")):
        if slots is None:
            continue
        if not slots.acquire(blocking=False):
            raise ApiError(503, "busy", message, retry_after(settings))
        stack.callback(slots.release)


@router.post("/v1/extract", response_model=ExtractResult, responses=ERROR_RESPONSES,
             summary="Extract text only (no ID detection, no LLM) and return it in the response",
             description="Synchronous: the request waits for OCR (about 2-3 s per scanned page on CPU). "
                         "Arabic, English and mixed documents. At most `EXTRACT_MAX_PAGES` pages; "
                         "use `/v1/documents` for larger files. The file is processed in memory only.")
async def extract_text(request: Request, file: UploadFile = File(...),
                       includeBoxes: bool | None = Form(None, description="Add `lines` [{text, confidence, bbox}] "
                                                                          "and the image size to every OCR page "
                                                                          "(default: the profile's, normally false)"),
                       profile: str | None = Form(None, description="Processing profile: one of your client's "
                                                                    "(default: its first)"),
                       client: Client = Depends(require_scope("extract")),
                       settings: Settings = Depends(app_settings), store: JobStore = Depends(app_store),
                       clients: ClientRegistry = Depends(app_clients)):
    state = request.app.state
    prof = use_profile(clients, client, profile)
    if not prof.return_text:
        raise ApiError(403, "forbidden", f"profile {prof.name} returns no text: use POST /v1/documents")
    include_boxes = prof.include_boxes if includeBoxes is None else includeBoxes
    with ExitStack() as slots:  # released when the request ends
        _take_slots(slots, request, client, settings)
        data, mime = await read_upload(file, settings)
        t0 = time.perf_counter()
        status, pages = "failed", 0
        try:
            opts: dict = {"include_boxes": True} if include_boxes else {}
            if not prof.is_default:
                opts["profile"] = prof
            result = await run_in_threadpool(state.analyze_fn, data, file.filename or "", state.services,
                                             detect_id=False, max_pages=settings.extract_max_pages, **opts)
            status, pages = "done", result["file"]["pages"]
            state.metrics.observe_result(result)
            facts = {"endpoint": "extract", **audit.file_facts(data, mime),
                     "ms": round((time.perf_counter() - t0) * 1000), **audit.result_facts(result)}
            await run_in_threadpool(audit.record_safely, store, AuditEvent(
                audit.EXTRACT_DONE, client.id, None, request.state.request_id, facts), state.metrics)
            return result
        except FileRejected as e:
            msg = str(e)
            if e.code == "too_many_pages":
                msg += "; use POST /v1/documents for large documents"
            raise ApiError(REJECT_STATUS.get(e.code, 400), e.code, msg) from None
        except Exception as e:  # noqa: BLE001 - never leak details (they may contain document content)
            log.error("extract failed error=%s client=%s req=%s", type(e).__name__, client.id,
                      request.state.request_id)
            raise ApiError(500, "internal_error", "processing failed") from None
        finally:
            state.metrics.extracts.labels(status, client.id).inc()
            log.info("extract finished client=%s bytes=%d pages=%d ms=%d status=%s req=%s", client.id, len(data),
                     pages, (time.perf_counter() - t0) * 1000, status, request.state.request_id)
