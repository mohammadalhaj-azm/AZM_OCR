"""Prometheus metrics (GET /metrics). Labels are low-cardinality only: enums, route templates and client ids (bounded
by the client registry); never job ids, file names or content.

Metrics are per process: run one API process per container and scale with containers (not `uvicorn --workers`),
so each scrape target is one process. Counters and histograms count this process's work: add them up across
replicas (`sum`). The gauges read from the shared job store (`ocr_jobs`, `ocr_queue_oldest_age_seconds`,
`ocr_processing_oldest_age_seconds`, `ocr_webhooks_pending`) are the same on every replica: use `max`, not `sum`.
Example alert rules: docs/alerts.yml.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

DURATION_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200)
WAIT_BUCKETS = (0.5, 1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)
STAGES = ("extract", "ocr", "llm", "vision")
NO_CLIENT = "-"  # requests without a valid key
DEPENDENCIES = ("store", "ocr_models", "worker", "llm")


class Metrics:
    def __init__(self):
        self.registry = r = CollectorRegistry()
        self.http = Counter("ocr_http_responses", "HTTP responses", ["route", "status", "client"], registry=r)
        self.jobs = Counter("ocr_jobs_finished", "Finished jobs", ["status", "client"], registry=r)
        self.job_seconds = Histogram("ocr_job_duration_seconds", "Job processing time", buckets=DURATION_BUCKETS,
                                     registry=r)
        self.queue_wait = Histogram("ocr_job_queue_wait_seconds", "Time a job waited in the queue before a worker "
                                    "took it", buckets=WAIT_BUCKETS, registry=r)
        self.stage_seconds = Histogram("ocr_stage_duration_seconds", "Time per pipeline stage", ["stage"],
                                       buckets=DURATION_BUCKETS, registry=r)
        self.pages = Counter("ocr_pages", "Pages extracted", ["method"], registry=r)
        self.decisions = Counter("ocr_id_decisions", "ID detection results", ["doc_type", "decided_by"], registry=r)
        self.llm_fallbacks = Counter("ocr_llm_fallbacks", "ID detection went on without a model that failed or is "
                                     "missing (llm: rules only; vision: text only)", ["stage"], registry=r)
        self.extracts = Counter("ocr_extract_requests", "Synchronous /v1/extract requests", ["status", "client"],
                                registry=r)
        self.webhooks = Counter("ocr_webhooks", "Finished webhook deliveries", ["result", "client"], registry=r)
        self.audit_failures = Counter("ocr_audit_write_failures", "Audit events that could not be written",
                                      registry=r)
        # At scrape time, from the shared job store (the same on every replica: aggregate with max)
        self.queue = Gauge("ocr_jobs", "Jobs in the store by status (at scrape time)", ["status"], registry=r)
        self.oldest = Gauge("ocr_queue_oldest_age_seconds", "Age of the oldest queued job (0: none)", registry=r)
        self.oldest_processing = Gauge("ocr_processing_oldest_age_seconds", "How long the longest-running job has "
                                       "been processing (0: none)", registry=r)
        self.webhooks_pending = Gauge("ocr_webhooks_pending", "Webhook deliveries not yet delivered or given up",
                                      registry=r)
        # At scrape time, for this process
        self.up = Gauge("ocr_dependency_up", "1 if the dependency works for this process: store (database), "
                        "ocr_models, worker, llm (the text model is available)", ["dependency"], registry=r)

    def llm_fallback(self, stage: str) -> None:
        self.llm_fallbacks.labels(stage).inc()

    def observe_result(self, result: dict) -> None:
        timings = result.get("timingsMs") or {}
        for stage in STAGES:
            if stage in timings:
                self.stage_seconds.labels(stage).observe(timings[stage] / 1000)
        for page in result.get("pages") or []:
            self.pages.labels(page.get("method", "unknown")).inc()
        det = result.get("idDetection")
        if det:
            self.decisions.labels(det.get("docType", "unknown"), (det.get("decidedBy") or ["none"])[-1]).inc()

    def render(self, counts: dict[str, int], ages: dict[str, float] | None = None, webhooks_pending: int | None = None,
               up: dict[str, bool] | None = None) -> bytes:
        for status, n in counts.items():
            self.queue.labels(status).set(n)
        if ages is not None:
            self.oldest.set(ages.get("queued", 0.0))
            self.oldest_processing.set(ages.get("processing", 0.0))
        if webhooks_pending is not None:
            self.webhooks_pending.set(webhooks_pending)
        for dependency, ok in (up or {}).items():
            self.up.labels(dependency).set(1 if ok else 0)
        return generate_latest(self.registry)
