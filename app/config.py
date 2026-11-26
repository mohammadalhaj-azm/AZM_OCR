"""Service configuration. Every value comes from env vars (or a local .env file)."""

import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER = re.compile(r"change[-_]?me", re.IGNORECASE)  # the example values shipped in .env.example
OCR_DEVICE_PATTERN = r"^(|cpu|gpu(:\d+)?)$"  # "" = auto
MIN_SECRET_CHARS = 32  # e.g. python -c "import secrets; print(secrets.token_urlsafe(32))" gives 43


class ConfigError(RuntimeError):
    """The configuration is unsafe or invalid; the service refuses to start."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # development: no checks. production: the service refuses to start with an unsafe configuration
    # (see production_problems).
    environment: Literal["development", "production"] = "development"
    # json: one JSON object per line with request_id / job_id / client_id (app/logs.py); the Docker image uses json
    log_format: Literal["text", "json"] = "text"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # LLM backend: any OpenAI-compatible server (Ollama now, vLLM later).
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "ollama"  # Ollama ignores it; vLLM may require one.
    llm_text_model: str = "qwen3:8b"
    llm_vision_model: str = "qwen3-vl:8b-instruct"
    llm_vision_base_url: str = ""  # empty = same server as llm_base_url (vLLM serves one model per server)
    llm_timeout_s: float = 180.0
    llm_vision_timeout_s: float = 300.0  # CPU vision is slow, and slower still when RAM is tight
    llm_enabled: bool = True
    # Extra JSON merged into every chat request. Ollama: disable Qwen3 thinking with reasoning_effort.
    # vLLM: '{"chat_template_kwargs": {"enable_thinking": false}}'
    llm_extra_body: str = '{"reasoning_effort": "none"}'
    llm_max_chars: int = 5000  # longer text is truncated (head + tail)
    vision_enabled: bool = True
    vision_threshold: float = 0.7  # use vision when combined confidence is below this
    vision_max_side: int = 1024  # downscale images sent to the vision model (CPU speed)

    # Combining: final P(id) = (1 - rules_weight) * P_llm + rules_weight * rule_score;
    # with vision: (1 - vision_weight) * P_text + vision_weight * P_vision.
    rules_weight: float = 0.4
    vision_weight: float = 0.6

    # Skip the LLM (and vision) for documents that are clearly not IDs: no rule signal at all (no MRZ, no ID
    # keyword, no ID-number pattern) AND long (at least this many pages OR non-space characters). ID documents are
    # 1-2 short pages, so this saves a 30-60 s LLM call on contracts, reports, statements... 0 disables a criterion.
    llm_skip_min_pages: int = 5
    llm_skip_min_chars: int = 8000
    llm_skip_confidence: float = 0.9

    # Uploads and extraction
    max_upload_mb: int = 20
    max_pages: int = 100  # PDF pages / TIFF frames / DOCX images
    min_text_chars_per_page: int = 30  # below this (non-space chars), a PDF page is rendered and OCR'd
    ocr_dpi: int = 300

    # OCR (PaddleOCR PP-OCRv5). Detection runs once; both recognizers read every box.
    ocr_det_model: str = "PP-OCRv5_mobile_det"  # "PP-OCRv5_server_det" is ~14x slower on CPU
    ocr_rec_model_ar: str = "arabic_PP-OCRv5_mobile_rec"
    ocr_rec_model_en: str = "en_PP-OCRv5_mobile_rec"  # empty = Arabic model only
    ocr_det_limit_side_len: int = 64
    ocr_det_limit_type: str = "min"  # "min" = never downscale (same as the PaddleOCR pipeline)
    ocr_min_score: float = 0.5  # drop recognized boxes below this confidence
    ocr_max_side: int = 4000  # downscale larger images before OCR
    ocr_cpu_threads: int | None = None  # None = PaddleOCR default
    # "" = GPU 0 when PaddlePaddle is the CUDA build (GPU image) and sees a GPU, else the CPU. Or "cpu", "gpu", "gpu:1".
    ocr_device: str = Field("", pattern=OCR_DEVICE_PATTERN)
    ocr_preprocess: bool = False  # OpenCV deskew + CLAHE
    # Independent OCR engines (each loads its own models, ~0.4 GB RAM): this many images are OCR'd at once, e.g. the
    # scanned pages of one PDF in parallel. With OCR_CPU_THREADS unset, the cores are split evenly between engines.
    # Measured (scripts/bench_ocr.py, 16 threads): 2 engines ~2.2x pages/s, 3-4 engines no further gain.
    ocr_engines: int = 2
    # Run the Arabic and English recognizers at the same time (per engine): ~1.4x faster per image, same text.
    ocr_parallel_rec: bool = True
    ocr_rec_batch_size: int = 8  # text boxes per recognizer batch
    ocr_allow_download: bool = False  # runtime stays offline; use `app.cli download-models`

    # Handwriting OCR (experimental, POST /v1/handwriting; see app/handwriting). Separate from the OCR above: it never
    # changes /v1/extract or /v1/documents. Model: a Qwen2.5-VL-3B fine-tune for Arabic/English handwriting.
    handwriting_enabled: bool = True  # off = the endpoint answers 503 handwriting_unavailable
    # transformers: the model runs in this process (pip extra `handwriting`; CPU or CUDA).
    # openai: any OpenAI-compatible server that serves the model (e.g. vLLM on a GPU machine).
    handwriting_backend: Literal["transformers", "openai"] = "transformers"
    handwriting_model: str = "sherif1313/Arabic-English-handwritten-OCR-v3"  # HF id or local folder / served name
    handwriting_base_url: str = ""  # openai backend, e.g. http://gpu-server:8002/v1
    handwriting_api_key: str = ""
    handwriting_timeout_s: float = 600.0  # per page (openai backend)
    # Pages at the model at once, across all requests and jobs. openai backend (vLLM batches them on the GPU): e.g. 8.
    # transformers backend: 1 (the only allowed value; the in-process model reads one page at a time).
    handwriting_max_concurrent: int = 1
    handwriting_device: Literal["auto", "cpu", "cuda"] = "auto"  # transformers backend; auto = cuda if available
    handwriting_dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"  # auto = bf16 on cuda, fp32 on cpu
    # Cap on the image size the model sees (pixels; 28x28 = one visual token). The main speed lever on CPU. The
    # model's own default (12.8 MP, ~16k tokens) is far too slow there.
    handwriting_max_pixels: int = 1280 * 28 * 28
    handwriting_min_pixels: int = 4 * 28 * 28
    handwriting_max_side: int = 1600  # openai backend: images are downscaled to this before upload
    handwriting_max_new_tokens: int = 1024  # per page
    handwriting_repetition_penalty: float = 1.1  # the model card's value
    handwriting_max_pages: int = 5  # synchronous endpoint: keep it small (CPU: about a minute or more per page)
    handwriting_dpi: int = 200  # PDF pages are rendered at this DPI (the model card asks for >= 200)
    # Pages with no ink are not sent to the model, which could otherwise invent text for them. Kept very low on
    # purpose: a small signature or a single short word must still reach the model (app/handwriting/guards.py). An
    # empty page with visible dirt is NOT caught; the evaluation measures what the model invents for such pages.
    handwriting_blank_check: bool = True
    handwriting_blank_ink_ratio: float = 0.00001  # ink share (margins and specks ignored) below which a page is blank
    handwriting_segmenter: Literal["whole_page"] = "whole_page"  # V1: the whole page in one model call
    handwriting_warmup: bool = False  # load the model at startup instead of on the first request
    handwriting_prompt: str = ""  # empty = the prompt the model was fine-tuned with (app/handwriting/backends.py)

    # Extract-and-answer (experimental, POST /v1/extract-and-answer; see app/answer): pages are routed to PaddleOCR
    # or the handwriting model, then the text LLM corrects them, lists what a person should review, and answers the
    # caller's questions from the document only.
    answer_enabled: bool = True  # off = the endpoint answers 503 answer_unavailable
    answer_max_pages: int = 20  # handwriting is slow: about a minute or more per page on CPU
    answer_max_questions: int = 10
    answer_max_question_chars: int = 500
    answer_llm_model: str = ""  # empty = LLM_TEXT_MODEL
    answer_timeout_s: float = 300.0  # per LLM call (one per page chunk, one for the questions)
    answer_max_tokens: int = 4096  # per LLM call; an output cut off at this length is not used
    # Page text is corrected in chunks of at most this many characters (Ollama's context window is set on the server,
    # OLLAMA_CONTEXT_LENGTH, and silently truncates longer input)
    answer_chunk_chars: int = 3000
    # Correction calls of one job sent at once (page chunks). 1 = one after another. With vLLM (continuous batching)
    # e.g. 4; with Ollama it only helps if OLLAMA_NUM_PARALLEL allows it. The result is the same in any case.
    answer_llm_concurrency: int = Field(1, ge=1, le=32)
    answer_context_chars: int = 20000  # questions are answered from at most this much (corrected) text
    answer_keep_max_side: int = 2000  # page images kept for the handwriting model / vision are downscaled to this
    # Routing of scanned pages and images. STARTING VALUES, NOT CALIBRATED: measure them on your own documents with
    # eval/run_routing_eval.py. Confidences are PaddleOCR's recognition scores (0-1).
    routing_low_line_conf: float = 0.80  # a recognized text box below this counts as low-confidence
    routing_printed_min_conf: float = 0.90  # printed: mean confidence at least this ...
    routing_printed_max_low_share: float = 0.15  # ... and at most this share of low-confidence boxes
    routing_handwritten_max_conf: float = 0.70  # handwritten: mean confidence below this ...
    routing_min_kept_share: float = 0.50  # ... or fewer than this share of detected boxes could be read at all
    routing_vision: bool = True  # unclear pages are classified by LLM_VISION_MODEL (printed/handwritten/mixed)
    # Unclear page and no vision verdict: both = run both engines and keep both readings (handwriting first);
    # handwriting / printed = that engine only. The page is marked `uncertain` in every case.
    routing_uncertain_fallback: Literal["both", "handwriting", "printed"] = "both"
    route_printed_arabic: Literal["paddle", "handwriting"] = "paddle"  # engine for printed Arabic scans

    # Detection
    id_patterns_path: Path = Path("config/id_patterns.yaml")

    # API
    api_keys: str = ""  # comma-separated; empty = every request is rejected (fail closed)
    # Client registry (YAML, see config/clients.example.yaml): key hashes, scopes, webhook settings and limits per
    # client system. Empty = each API_KEYS entry is a client with all scopes and the global WEBHOOK_* settings.
    clients_file: str = ""
    upload_dir: Path = Path(tempfile.gettempdir()) / "ocr-id-service"
    result_ttl_s: int = 3600  # finished jobs (and their results) are deleted after this
    max_queued_jobs: int = 100
    worker_threads: int = 1  # OCR + LLM are CPU-bound; more threads rarely help on CPU
    # > 0: split jobs into an OCR stage (WORKER_THREADS threads) and an LLM stage (this many threads), so the next
    # job's OCR runs while the LLM works on the previous one. Pays off when the LLM runs on another machine / GPU.
    llm_worker_threads: int = 0
    ocr_warmup: bool = True  # load OCR models at startup
    # Synchronous text-only endpoint (POST /v1/extract): the HTTP request waits for OCR, so keep it small.
    extract_max_pages: int = 20  # larger documents: use the async /v1/documents
    extract_max_concurrent: int = 2  # more simultaneous /v1/extract requests get 503
    retry_after_s: int = 5  # Retry-After header on 503 (queue full / busy)
    max_wait_s: int = 30  # longest long-poll allowed by GET /v1/jobs/{id}?wait=
    metrics_enabled: bool = True  # Prometheus text format at GET /metrics (no API key: keep it internal)

    # Job store. memory = single process, lost on restart. postgres = durable, shared by all replicas.
    job_store: Literal["memory", "postgres"] = "memory"
    database_url: str = ""  # e.g. postgresql://ocr:secret@127.0.0.1:5432/ocr
    db_pool_size: int = 10
    db_poll_s: float = 0.5  # idle workers check the queue this often
    job_stale_s: int = 120  # a processing job without a worker heartbeat for this long is re-queued
    job_max_attempts: int = 2  # after this many claims a job fails with worker_lost (protects against crashing files)
    # base64 of 32 random bytes: encrypts uploads, results, file names and metadata stored by the postgres store
    # (and keys the externalId hash in the audit trail)
    result_encryption_key: str = ""
    audit_retention_days: int = 365  # audit events older than this are deleted; 0 = keep them forever

    # Webhooks: POST {jobId, status} on completion, signed with HMAC-SHA256. Deliveries live in the job store (an
    # outbox), so retries survive restarts and any replica sends them.
    webhook_secret: str = ""
    webhook_allowed_hosts: str = ""  # comma-separated host names; empty = callbacks disabled
    webhook_timeout_s: float = 10.0
    # Retries: the first after WEBHOOK_BACKOFF_S, doubling up to WEBHOOK_MAX_BACKOFF_S between attempts, until
    # WEBHOOK_RETRY_WINDOW_S after the job finished (a client can have its own window in CLIENTS_FILE).
    webhook_retries: int = 0  # most attempts in total; 0 = no limit (only the window)
    webhook_backoff_s: float = 2.0
    webhook_max_backoff_s: float = 3600.0
    webhook_retry_window_s: int = 86400  # receivers in a maintenance window still get their callback
    webhook_threads: int = 4  # simultaneous deliveries per process, so one slow receiver can't hold up the others

    # Admin dashboard (/admin, see app/admin). Empty ADMIN_USERS_FILE = the dashboard is off (404).
    admin_users_file: str = ""  # YAML with admin users and scrypt password hashes (scripts/admin_user.py)
    admin_session_secret: str = ""  # signs the session cookie; empty (development only) = random per process
    admin_session_hours: float = 8.0
    sample_interval_s: float = 2.0  # resource sampling (CPU, RAM, GPU, ...) for the live view
    sample_persist_s: float = 10.0  # every this often a sample is stored (history, other replicas)
    sample_retention_h: int = 24
    log_buffer_size: int = 5000  # recent log records kept in RAM per process (all levels)
    log_retention_days: int = 30  # stored WARNING+ log records
    alert_eval_s: float = 15.0  # how often each replica evaluates the alert rules
    alert_for_s: int = 300  # resource thresholds must be exceeded this long before an alert fires
    alert_cpu_pct: float = 90.0
    alert_mem_pct: float = 90.0
    alert_disk_pct: float = 90.0
    alert_gpu_mem_pct: float = 95.0
    alert_queue_age_s: int = 600  # the oldest queued job waited longer than this
    alert_job_stuck_s: int = 1800  # a job has been processing longer than this
    alert_failure_rate: float = 0.1  # share of failed jobs in the last 15 minutes (at least 5 jobs)
    config_poll_s: float = 5.0  # replicas pick up configuration changes made in the dashboard this often
    eval_out_dir: Path = Path("eval/out")  # where eval.run_eval writes summary.json (accuracy in AI monitoring)

    @property
    def admin_enabled(self) -> bool:
        return bool(self.admin_users_file)

    @property
    def webhook_host_set(self) -> set[str]:
        return {h.strip().lower() for h in self.webhook_allowed_hosts.split(",") if h.strip()}

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def production_problems(self) -> list[str]:
        """Why this configuration is unsafe for production; empty if it is fine. Never contains secret values."""
        problems = []
        if self.job_store != "postgres":
            problems.append("JOB_STORE must be postgres (the memory store loses every job on restart)")
        if not self.result_encryption_key:
            problems.append("RESULT_ENCRYPTION_KEY must be set (stored documents and results would be readable)")
        if not self.clients_file:  # with CLIENTS_FILE the registry is checked (app.api.clients)
            if not self.api_key_set:
                problems.append("API_KEYS is empty (or set CLIENTS_FILE)")
            weak = sum(1 for k in self.api_key_set if len(k) < MIN_SECRET_CHARS or PLACEHOLDER.search(k))
            if weak:
                problems.append(f"API_KEYS: {weak} key(s) shorter than {MIN_SECRET_CHARS} characters or a placeholder")
        if self.webhook_secret and (len(self.webhook_secret) < MIN_SECRET_CHARS
                                    or PLACEHOLDER.search(self.webhook_secret)):
            problems.append(f"WEBHOOK_SECRET is shorter than {MIN_SECRET_CHARS} characters or a placeholder")
        if self.webhook_host_set and not self.webhook_secret:
            problems.append("WEBHOOK_ALLOWED_HOSTS is set but WEBHOOK_SECRET is empty")
        if PLACEHOLDER.search(self.database_url):
            problems.append("DATABASE_URL contains a placeholder password")
        if self.ocr_allow_download:
            problems.append("OCR_ALLOW_DOWNLOAD must be false (the runtime stays offline)")
        if self.admin_enabled and (len(self.admin_session_secret) < MIN_SECRET_CHARS
                                   or PLACEHOLDER.search(self.admin_session_secret)):
            problems.append(f"ADMIN_SESSION_SECRET must be at least {MIN_SECRET_CHARS} characters (and not a "
                            "placeholder) when the admin dashboard is on")
        return problems

    def check_production(self, more_problems: list[str] | None = None) -> None:
        """Raise ConfigError when ENVIRONMENT=production and the configuration is unsafe.
        more_problems: from other checks, e.g. the client registry."""
        if self.environment == "production" and (problems := self.production_problems() + (more_problems or [])):
            raise ConfigError("ENVIRONMENT=production, refusing to start: " + "; ".join(problems))


@lru_cache
def get_settings() -> Settings:
    return Settings()
