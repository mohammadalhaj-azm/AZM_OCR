"""REST API: async document jobs (OCR + ID detection) and synchronous text-only extraction.

    uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000

`create_app` wires settings, the job store, the worker and the webhook dispatcher. The endpoints are in
`app/api/routes/`, shared dependencies in `app/api/deps.py`, the middleware in `app/api/middleware.py`.
With ADMIN_USERS_FILE set, the admin dashboard (`app/admin`) is mounted at /admin.
"""

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from starlette.formparsers import MultiPartParser

from app import __version__, logs
from app.api import audit, errors, middleware, webhook
from app.api.clients import load_clients
from app.api.deps import max_request_bytes
from app.api.metrics import Metrics
from app.api.routes import documents, extract, handwriting, health, jobs
from app.api.store import JobStore, make_store
from app.api.worker import AnalyzeFn, Worker
from app.config import Settings, get_settings
from app.handwriting import HandwritingService
from app.pipeline import Services, analyze
from app.privacy import apply_offline_env

log = logging.getLogger("app.api")


def keep_uploads_in_memory(max_body: int) -> None:
    """Starlette writes multipart files over 1 MB to a temp file on disk while it parses the request. Raise that
    threshold to the largest accepted request, so uploads stay in RAM until they are processed (or rejected).
    Larger bodies never get that far: the upload routes stop reading at `max_body` (see create_app)."""
    MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, max_body)


def create_app(settings: Settings | None = None, services: Any = None, analyze_fn: AnalyzeFn = analyze,
               store: JobStore | None = None, webhook_transport: httpx.BaseTransport | None = None,
               handwriting_service: HandwritingService | None = None) -> FastAPI:
    settings = settings or get_settings()
    clients = load_clients(settings)  # CLIENTS_FILE, or API_KEYS; ConfigError when invalid
    # ENVIRONMENT=production: refuse to start with an unsafe configuration
    settings.check_production(clients.production_problems())
    apply_offline_env()
    logs.configure(settings.log_format, settings.log_level)
    store = store or make_store(settings)
    # Experimental handwriting OCR: its own service, separate from the pipeline (ConfigError when misconfigured)
    hw = handwriting_service or HandwritingService(settings)
    metrics = Metrics()
    max_body = max_request_bytes(settings)
    keep_uploads_in_memory(max_body)
    admin = None
    if settings.admin_enabled:
        from app.admin import AdminRuntime

        admin = AdminRuntime(settings, store, clients, metrics)  # ConfigError for an invalid ADMIN_USERS_FILE

    def on_fallback(stage: str) -> None:
        metrics.llm_fallback(stage)  # ocr_llm_fallbacks_total
        audit.note_fallback(stage)  # recorded with the job (job.done `fallbacks`)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        enabled = sum(1 for c in clients.clients if c.enabled)
        log.info("clients loaded source=%s clients=%d enabled=%d", clients.source, len(clients.clients), enabled)
        if not enabled:
            log.warning("no API key or enabled client configured: all /v1 requests will be rejected")
        store.open()
        if admin is not None:
            admin.prepare(app)  # dashboard-managed configuration, before the services start
        svc = services or Services.create(settings)
        if hasattr(svc, "report_fallbacks"):
            svc.report_fallbacks(on_fallback)
        client = httpx.Client(timeout=settings.webhook_timeout_s, transport=webhook_transport)
        dispatcher = webhook.Dispatcher(store, client, clients, settings, metrics)  # sends due webhooks from the store
        worker = Worker(settings, store, analyze_fn, svc, dispatcher, metrics)
        app.state.services, app.state.worker, app.state.dispatcher = svc, worker, dispatcher
        dispatcher.start()
        worker.start()
        if hw.enabled and settings.handwriting_warmup:
            threading.Thread(target=hw.warmup, name="handwriting-warmup", daemon=True).start()
        if admin is not None:
            admin.start(app)
        yield
        if admin is not None:
            admin.stop()
        worker.stop()  # unfinished jobs go back to the queue
        dispatcher.stop()  # pending webhooks stay in the store
        client.close()
        hw.close()
        store.close()

    app = FastAPI(title="OCR + ID detection", version=__version__, lifespan=lifespan,
                  description="Local, privacy-first text extraction (Arabic/English) and identity-document detection. "
                              "All `/v1` endpoints need the `X-Api-Key` header. Every error has the shape "
                              '`{"error": {"code", "message", "requestId"}}`; send `X-Request-Id` to correlate.')
    app.state.settings, app.state.store, app.state.metrics = settings, store, metrics
    app.state.clients, app.state.analyze_fn, app.state.admin = clients, analyze_fn, admin
    app.state.handwriting = hw
    app.state.ref_hash = audit.ref_hasher(settings)  # externalId -> its hash in the audit trail
    app.state.extract_slots = threading.BoundedSemaphore(max(1, settings.extract_max_concurrent))
    app.state.client_extract_slots = {c.id: threading.BoundedSemaphore(c.extract_concurrent)
                                      for c in clients.clients if c.extract_concurrent}
    errors.install(app)
    middleware.install(app, settings, clients, metrics)

    # The middleware rejects a too-large Content-Length up front; the route limit also stops chunked uploads
    # (no Content-Length) once they pass max_body, with the usual 413 too_large error.
    app.include_router(documents.router, max_body_size=max_body)
    app.include_router(extract.router, max_body_size=max_body)
    app.include_router(handwriting.router, max_body_size=max_body)  # experimental
    app.include_router(jobs.router)
    app.include_router(health.router)
    app.include_router(handwriting.status_router)
    if settings.metrics_enabled:
        app.include_router(health.metrics_router)
    if admin is not None:
        from app.admin import install

        install(app, admin)
    return app


app = create_app()
