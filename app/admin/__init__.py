"""Admin dashboard and monitoring (/admin): visibility into documents, resources, AI models, integrations, logs and
alerts, and control over the configuration. On when ADMIN_USERS_FILE is set.

`AdminRuntime` is created by `create_app` and runs these background parts while the app runs:

- `ResourceSampler` (system.py): CPU, memory, disks, I/O, GPUs, dependency health, every SAMPLE_INTERVAL_S;
- `LogBuffer` (logbuffer.py): recent log records in RAM, WARNING+ stored for all replicas;
- `AlertEvaluator` (alerts.py): the alert rules, every ALERT_EVAL_S;
- `ManagedConfig` (managed_config.py): configuration changes made in the dashboard, applied every CONFIG_POLL_S.

The HTTP API and pages are in routes.py, the browser code in static/ (no build step, no external resources).
"""

import logging
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from app.admin.alerts import AlertEvaluator
from app.admin.auth import AdminAuth
from app.admin.logbuffer import LogBuffer
from app.admin.managed_config import ManagedConfig
from app.admin.store import make_admin_store
from app.admin.system import ResourceSampler
from app.api.clients import ClientRegistry
from app.api.store import JobStore
from app.config import Settings
from app.ocr.engine import OcrEngine

log = logging.getLogger("app.admin")
STATIC = Path(__file__).with_name("static")
LLM_CHECK_S = 15
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
       "font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


class AdminRuntime:
    def __init__(self, settings: Settings, jobs: JobStore, clients: ClientRegistry, metrics):
        self.s, self.jobs, self.clients, self.metrics = settings, jobs, clients, metrics
        self.auth = AdminAuth(settings)  # ConfigError at startup for an invalid users file
        self.store = make_admin_store(jobs)
        self.config = ManagedConfig(settings, self.store, clients)
        self.instance: str | None = None
        self.sampler: ResourceSampler | None = None
        self.logs: LogBuffer | None = None
        self.alerts: AlertEvaluator | None = None
        self.stopping = threading.Event()
        self._app_state = None

    def prepare(self, app: FastAPI) -> None:
        """After the job store opened, before the services are created: apply the dashboard-managed configuration
        (settings that need a restart included), so the services start with it."""
        self._app_state = app.state
        self.config.apply(startup=True)
        app.state.extract_slots = threading.BoundedSemaphore(max(1, self.s.extract_max_concurrent))
        self._rebuild_client_slots()

    def start(self, app: FastAPI) -> None:
        state = app.state
        worker = state.worker
        self.instance = worker.worker_id
        self.logs = LogBuffer(self.instance, self.s.log_buffer_size, self.store)
        self.logs.install()
        self.sampler = ResourceSampler(self.s, self.instance, self.store, self._health, self._work)
        self.alerts = AlertEvaluator(
            self.s, self.store, self.jobs, self.instance, self.sampler.latest,
            lambda: self.metrics.registry.get_sample_value("ocr_audit_write_failures_total") or 0.0,
            lambda: self.config.error, lambda: [c.id for c in self.clients.clients])
        self.config.on_change.append(self._config_changed)
        self.sampler.tick()  # a first sample right away
        self.sampler.start()
        self.alerts.start()
        self.config.start()
        log.info("admin dashboard on instance=%s users=%d store=%s", self.instance, len(self.auth.users()),
                 "postgres" if self.store.persistent else "memory")

    def stop(self) -> None:
        self.stopping.set()
        for part in (self.config, self.alerts, self.sampler):
            if part is not None:
                part.stop()
        if self.logs is not None:
            self.logs.uninstall()

    # --- callbacks ---------------------------------------------------------------------------------------------
    def _health(self) -> dict:
        state = self._app_state
        services = getattr(state, "services", None)
        worker = getattr(state, "worker", None)
        llm = getattr(services, "llm", None)
        # the model list is cached: one request to the LLM server per LLM_CHECK_S, not per sample
        models = llm.available_models(max_age_s=LLM_CHECK_S) if llm is not None else set()
        return {"store": self.jobs.ping(), "ocrModels": OcrEngine(self.s).models_present(),
                "worker": bool(worker and worker.alive()), "llmReachable": bool(models),
                "llm": self.s.llm_text_model in models, "vision": self.s.llm_vision_model in models}

    def _work(self) -> dict:
        state = self._app_state
        services = getattr(state, "services", None)
        worker = getattr(state, "worker", None)
        engine = getattr(services, "engine", None)
        llm = getattr(services, "llm", None)
        calls = llm.stats.snapshot() if hasattr(llm, "stats") else {}
        return {"activeJobs": len(worker.active()) if worker else 0,
                "ocr": engine.stats() if hasattr(engine, "stats") else None,
                "llmInFlight": sum(m["inFlight"] for m in calls.values())}

    def _rebuild_client_slots(self) -> None:
        state = self._app_state
        old = getattr(state, "client_extract_slots", {}) or {}
        new = {}
        for c in self.clients.clients:
            if not c.extract_concurrent:
                continue
            prev = old.get(c.id)
            # keep a semaphore whose limit didn't change, so requests in progress are still counted
            new[c.id] = prev if prev is not None and getattr(prev, "_initial_value", None) == c.extract_concurrent \
                else threading.BoundedSemaphore(c.extract_concurrent)
        state.client_extract_slots = new

    def _config_changed(self, kinds: set[str]) -> None:
        services = getattr(self._app_state, "services", None)
        if hasattr(services, "reset_detectors"):
            services.reset_detectors()  # thresholds, models, rule sets
        if "registry" in kinds:
            self._rebuild_client_slots()


def install(app: FastAPI, runtime: AdminRuntime) -> None:
    """The admin routes, static files and security headers."""
    from app.admin.routes import router

    app.include_router(router)
    app.mount("/admin/static", StaticFiles(directory=STATIC), name="admin-static")

    @app.middleware("http")
    async def admin_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/admin"):
            response.headers.setdefault("Content-Security-Policy", CSP)
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers["Cache-Control"] = "no-store" if "/api/" in request.url.path else "no-cache"
        return response
