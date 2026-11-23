"""Health probes (no API key) and Prometheus metrics."""

import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from app import __version__
from app.ocr.engine import OcrEngine

router = APIRouter()
metrics_router = APIRouter()  # included only with METRICS_ENABLED


def llm_status(request: Request) -> dict:
    settings = request.app.state.settings
    llm = getattr(getattr(request.app.state, "services", None), "llm", None)
    models = llm.available_models(max_age_s=5) if llm else set()
    return {"reachable": bool(models), "textModel": settings.llm_text_model in models,
            "visionModel": settings.llm_vision_model in models}


@router.get("/health", summary="Full status: 503 unless OCR models are present AND the text LLM is available")
def health(request: Request):
    state = request.app.state
    llm = llm_status(request)
    ocr_ok = OcrEngine(state.settings).models_present()
    ok = llm["textModel"] and ocr_ok
    body = {
        "status": "ok" if ok else "degraded",
        "version": __version__,
        "llm": llm,
        "ocr": {"modelsPresent": ocr_ok},
        "jobs": state.store.counts(),
        "time": int(time.time()),
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/health/live", summary="Liveness: the process is up (for restarts by an orchestrator)")
def health_live():
    return {"status": "ok"}


@router.get("/health/ready",
            summary="Readiness: OCR models present and the job worker running (for load balancers). "
                    "The LLM is reported but not required, because /v1/extract works without it.")
def health_ready(request: Request):
    state = request.app.state
    worker = getattr(state, "worker", None)
    dispatcher = getattr(state, "dispatcher", None)
    ocr_ok = OcrEngine(state.settings).models_present()
    worker_ok = worker is not None and worker.alive()
    db_ok = state.store.ping()
    ok = ocr_ok and worker_ok and db_ok
    body = {
        "status": "ready" if ok else "not_ready",
        "version": __version__,
        "ocr": {"modelsPresent": ocr_ok},
        "worker": {"alive": worker_ok},
        "store": {"backend": state.settings.job_store, "reachable": db_ok},
        "llm": llm_status(request),
        "jobs": state.store.counts() if db_ok else None,
        "webhooksPending": (dispatcher.pending() if dispatcher else 0) if db_ok else None,  # all replicas' deliveries
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@metrics_router.get("/metrics", include_in_schema=False)
def prometheus_metrics(request: Request):
    state = request.app.state
    db_ok = state.store.ping()
    worker = getattr(state, "worker", None)
    up = {"store": db_ok, "ocr_models": OcrEngine(state.settings).models_present(),
          "worker": worker is not None and worker.alive(), "llm": llm_status(request)["textModel"]}
    body = state.metrics.render(state.store.counts() if db_ok else {}, state.store.ages() if db_ok else None,
                                state.store.count_pending_webhooks() if db_ok else None, up)
    return Response(body, media_type="text/plain; version=0.0.4; charset=utf-8")
