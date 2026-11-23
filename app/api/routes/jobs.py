"""GET /v1/jobs/{job_id} (status + result, with optional long-poll) and DELETE /v1/jobs/{job_id}."""

import asyncio
import time

from fastapi import APIRouter, Depends, Query, Request
from fastapi.concurrency import run_in_threadpool

from app.api import audit
from app.api.clients import Client
from app.api.deps import app_settings, app_store, own_job, require_client
from app.api.errors import ERROR_RESPONSES
from app.api.models import JobView
from app.api.store import FINAL, AuditEvent, JobStore
from app.config import Settings

router = APIRouter()
POLL_STEP_S = 0.25


@router.get("/v1/jobs/{job_id}", response_model=JobView, responses=ERROR_RESPONSES,
            summary="Job status, and the result when done",
            description="With `wait=N` (seconds, at most MAX_WAIT_S) the request is held until the job is done or "
                        "failed, or N seconds pass (long-polling: no client-side polling loop needed).")
async def get_job(request: Request, job_id: str, wait: float = Query(0, ge=0, le=3600),
                  client: Client = Depends(require_client), settings: Settings = Depends(app_settings),
                  store: JobStore = Depends(app_store)):
    owner = client.id
    job = await run_in_threadpool(own_job, store, job_id, owner)
    end = time.monotonic() + min(wait, settings.max_wait_s)
    while job.status not in FINAL and time.monotonic() < end:
        await asyncio.sleep(min(POLL_STEP_S, max(0.0, end - time.monotonic())))
        job = await run_in_threadpool(own_job, store, job_id, owner)
    view: dict = {"jobId": job.id, "status": job.status}
    if job.external_id:
        view["externalId"] = job.external_id
    if job.metadata:
        view["metadata"] = job.metadata
    if job.status == "done" and job.result:
        view.update(job.result)
    if job.status == "failed":
        view["error"] = job.error
    if job.webhook:
        view["webhook"] = job.webhook
    if job.status in FINAL:  # the result (or error) was handed out
        await run_in_threadpool(audit.record_safely, store, AuditEvent(
            audit.FETCHED, owner, job.id, request.state.request_id, {"status": job.status}), request.app.state.metrics)
    return view


@router.delete("/v1/jobs/{job_id}", status_code=204, responses=ERROR_RESPONSES,
               summary="Delete a job and its result now")
def delete_job(request: Request, job_id: str, client: Client = Depends(require_client),
               store: JobStore = Depends(app_store)):
    job = own_job(store, job_id, client.id)
    # also its upload and idempotency key; the audit event is written with the deletion
    store.delete(job.id, AuditEvent(audit.DELETED, client.id, job.id, request.state.request_id, {"by": "client"}))
