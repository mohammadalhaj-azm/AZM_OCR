"""POST /v1/handwriting: experimental, synchronous handwriting OCR (Arabic first, English too).

Fully separate from /v1/extract and /v1/documents: its own service (app/handwriting), its own concurrency limit
(one request at a time per process), its own response model. GET /health/handwriting reports its status.
"""

import logging
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.api import audit
from app.api.clients import Client
from app.api.deps import REJECT_STATUS, app_settings, app_store, read_upload, require_scope, retry_after
from app.api.errors import ERROR_RESPONSES, ApiError
from app.api.models import HandwritingResult
from app.api.store import AuditEvent, JobStore
from app.config import Settings
from app.extractors.filetype import FileRejected
from app.handwriting import HandwritingBusy, HandwritingService, HandwritingUnavailable, UnknownChoice

log = logging.getLogger("app.api")
router = APIRouter()
status_router = APIRouter()


def app_handwriting(request: Request) -> HandwritingService:
    return request.app.state.handwriting


def _facts(result: dict) -> dict:
    """Metadata only: never text or file names."""
    pages = result["pages"]
    warnings = sorted({w for p in pages for w in p["warnings"]})
    return {"pages": len(pages), "skipped": sum(1 for p in pages if p.get("skipped")), "warnings": warnings,
            "languages": result["languages"], "timingsMs": result["timingsMs"], "model": result["model"]}


@router.post("/v1/handwriting", response_model=HandwritingResult, responses=ERROR_RESPONSES, tags=["experimental"],
             summary="Experimental: read handwritten text (Arabic first, English too) and return it",
             description="Synchronous and slow (on CPU about a minute or more per page). Images (JPG, PNG, WEBP, "
                         "TIFF) and PDFs, every page rendered; at most `HANDWRITING_MAX_PAGES` pages. One request at "
                         "a time (503 `busy` otherwise). Pages without ink are skipped, not sent to the model. "
                         "`confidence` is always null: the model gives no calibrated confidence. Separate from "
                         "`/v1/extract`, and may change while it is evaluated. The file is processed in memory "
                         "only. `model` picks one of the server's handwriting models (`GET /health/handwriting` "
                         "lists them; 400 `unknown_model` otherwise), `segmenter` how pages are split for it: "
                         "`whole_page` (one call per page, also for an image that is already a single line) or "
                         "`lines` (one call per detected text line).")
async def handwriting(request: Request, file: UploadFile = File(...),
                      model: str | None = Form(None, description="Handwriting model id; empty = the server's default",
                                               max_length=64),
                      segmenter: str | None = Form(None, description="whole_page or lines; empty = the model's "
                                                                     "default", max_length=32),
                      client: Client = Depends(require_scope("extract")),
                      settings: Settings = Depends(app_settings), store: JobStore = Depends(app_store),
                      service: HandwritingService = Depends(app_handwriting)):
    if not service.enabled:
        raise ApiError(503, "handwriting_unavailable", "handwriting OCR is disabled on this server")
    data, mime = await read_upload(file, settings)
    t0 = time.perf_counter()
    status, pages = "failed", 0
    try:
        result = await run_in_threadpool(service.transcribe, data, mime, model or None, segmenter or None)
        result["file"] = {"name": file.filename or "", **result["file"]}
        status, pages = "done", result["file"]["pages"]
        facts = {"endpoint": "handwriting", **audit.file_facts(data, mime),
                 "ms": round((time.perf_counter() - t0) * 1000), **_facts(result)}
        await run_in_threadpool(audit.record_safely, store, AuditEvent(
            audit.HANDWRITING_DONE, client.id, None, request.state.request_id, facts), request.app.state.metrics)
        return result
    except FileRejected as e:
        status = "rejected"
        raise ApiError(REJECT_STATUS.get(e.code, 400), e.code, str(e)) from None
    except UnknownChoice as e:  # a configured name, never document content
        status = "rejected"
        raise ApiError(400, e.code, str(e)) from None
    except HandwritingBusy:
        status = "busy"
        raise ApiError(503, "busy", "the handwriting model is busy, retry later", retry_after(settings)) from None
    except HandwritingUnavailable as e:  # its message never contains document content
        status = "unavailable"
        raise ApiError(503, "handwriting_unavailable", str(e)) from None
    except Exception as e:  # noqa: BLE001 - never leak details (they may contain document content)
        log.error("handwriting failed error=%s client=%s req=%s", type(e).__name__, client.id,
                  request.state.request_id)
        raise ApiError(500, "internal_error", "processing failed") from None
    finally:
        log.info("handwriting finished client=%s bytes=%d pages=%d ms=%d status=%s req=%s", client.id, len(data),
                 pages, (time.perf_counter() - t0) * 1000, status, request.state.request_id)


@status_router.get("/health/handwriting", tags=["experimental"],
                   summary="Experimental: status of the handwriting model (enabled, installed, downloaded, loaded "
                           "or reachable). Not part of /health or /health/ready")
def handwriting_status(service: HandwritingService = Depends(app_handwriting)):
    return service.status()
