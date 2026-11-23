# OCR + ID-detection service

Local, privacy-first REST service that extracts text (Arabic first, English second) from
PDF, DOCX, and images, and decides whether a document is an identity document.
All AI runs locally; no document data leaves the machine.

**Quick start** (after [Setup from zero](#setup-from-zero)):
```powershell
ollama serve                                            # in its own terminal
$env:API_KEYS = 'change-me'
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
curl -H "X-Api-Key: change-me" -F "file=@scan.pdf" http://127.0.0.1:8000/v1/documents   # text + ID detection (async)
curl -H "X-Api-Key: change-me" -F "file=@scan.pdf" http://127.0.0.1:8000/v1/extract     # text only (sync)
```

Pipeline: detect type (magic bytes) → extract (PDF text layer / OCR per page, DOCX + embedded images,
images) → Arabic normalization → rules (MRZ, keywords, ID-number patterns) → LLM (`qwen3:8b`) →
optional vision fallback (`qwen3-vl:8b-instruct`) → result.

| Module | Purpose |
|---|---|
| `app/extractors/` | file-type detection, PDF (pypdfium2), DOCX (python-docx), images |
| `app/ocr/` | PaddleOCR PP-OCRv5 engine, Arabic+English merge, OpenCV preprocessing |
| `app/normalize/` | Arabic normalization |
| `app/detection/` | MRZ, rules, LLM client, combiner |
| `app/api/` | FastAPI app (`main.py` wiring, `routes/` endpoints, `deps.py`, `middleware.py`), job store + `migrations/`, worker, webhooks |
| `app/admin/` | [admin dashboard](#admin-dashboard): monitoring, alerts, configuration, its API and its pages (`static/`) |
| `eval/` | evaluation over your real documents |

## Setup from zero

Tested on Windows 11 Pro (x64), Intel Core 7 240H, 32 GB RAM, **no NVIDIA GPU** (CPU only).
Run in PowerShell. These are the exact commands used.

### 1. Install uv and Ollama (winget)
```powershell
winget install --id astral-sh.uv -e --accept-source-agreements --accept-package-agreements
winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
```
Open a **new** terminal afterwards so `uv` and `ollama` are on `PATH`.
Installed versions: uv 0.13.9, Ollama 0.40.16.

### 2. Python 3.12 (managed by uv)
PaddlePaddle's Windows install guide lists Python up to 3.12, and wheels for 3.12 exist.
```powershell
uv python install 3.12
```

### 3. Project and dependencies
The project was created with:
```powershell
uv init --bare --python 3.12 --name ocr-id-service --vcs git
uv python pin 3.12
uv add fastapi "uvicorn[standard]" httpx pydantic-settings pyyaml python-multipart
uv add "paddlepaddle==3.2.0" "paddleocr>=3.7,<3.8"
uv add --dev pytest arabic-reshaper python-bidi pillow
```
On a fresh clone, one command does it all:
```powershell
uv sync
```
Notes:
- `paddlepaddle==3.2.0` is the CPU version recommended by the PaddleOCR install guide.
- `paddlex` (pulled in by PaddleOCR) pins `pyyaml==6.0.2`, so our pyyaml constraint is just `>=6.0`.
  Without the `paddleocr>=3.7` bound, uv silently picks the old PaddleOCR 2.10 (no PP-OCRv5).
- `arabic-reshaper` + `python-bidi` are dev-only: they render Arabic into synthetic test images,
  because Pillow on Windows has no built-in Arabic shaping (raqm).

### 4. Pull the models
```powershell
ollama pull qwen3:8b               # 5.2 GB, Q4_K_M: text classification
ollama pull qwen3-vl:8b-instruct   # 6.2 GB, Q4_K_M: vision fallback (instruct = no thinking step)
ollama list
```
`qwen3-vl:8b` (the default tag) is the *thinking* variant, which is very slow on CPU. That's why
`8b-instruct` is used instead.

PaddleOCR models are downloaded automatically on first use into
`%USERPROFILE%\.paddlex\official_models` (`PP-OCRv5_server_det`, `arabic_PP-OCRv5_mobile_rec`, ...).

### 5. Limit Ollama to one loaded model (saves RAM)
```powershell
[Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS', '1', 'User')
# then quit Ollama from the tray icon and start it again
```

### 6. Verify
```powershell
uv run python -c "import paddle; paddle.utils.run_check()"
$env:PYTHONIOENCODING = 'utf-8'
uv run python scripts/check_env.py     # Ollama text + vision + PaddleOCR on a synthetic image
uv run pytest
```

### Privacy / offline notes
- Ollama listens on `127.0.0.1:11434` only by default. Don't set `OLLAMA_HOST=0.0.0.0`.
- The Ollama desktop app checks for updates online. To avoid any outbound traffic at runtime,
  turn off automatic updates in the Ollama settings, or quit the tray app and run `ollama serve` headless.
- After all models are downloaded, run with Paddle and Hugging Face in offline mode:
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, `HF_HUB_OFFLINE=1` (see `.env.example`).
  Phase 4 makes the service enforce this.

### Starting Ollama
Either start the **Ollama** app from the Start menu, or run the server headless (preferred for
privacy, since there is no tray app and no update checker):
```powershell
$env:OLLAMA_MAX_LOADED_MODELS = '1'
ollama serve
```
Note: during setup, restarting the tray app from a non-interactive shell left it running *without*
its server (nothing listening on port 11434). `ollama serve` worked every time.

### Measured speed on this CPU-only machine (Phase 0, `scripts/check_env.py`)
| Step | Measured |
|---|---|
| `qwen3:8b` cold load + tiny reply | 15.4 s |
| `qwen3:8b` classification (89 prompt tokens, 105 output tokens, thinking off) | 12.5 s, ~8.4 tok/s |
| `qwen3-vl:8b-instruct` on a 1200×420 image (includes swapping models in RAM) | 80.4 s |
| PaddleOCR first init (includes model download) | 103 s, one-time |
| PaddleOCR warm, 1200×420 image, `PP-OCRv5_server_det` + `arabic_PP-OCRv5_mobile_rec` | 6.0 s |

Both LLM calls returned schema-valid JSON through `/v1/chat/completions` with
`response_format: json_schema` and `reasoning_effort: "none"`, with no thinking output.
Real document prompts will be about 1–2K tokens, so expect roughly 20–40 s per LLM classification.
OCR speed is tuned in Phase 1.

## Text extraction (Phase 1)

```powershell
uv run python -m app.cli download-models   # once, needs internet; runtime never downloads
uv run python -m app.cli extract <file>    # prints JSON: file, pages[{page, method, text, confidence}], fullText, timingsMs
uv run python -m app.cli extract <file> --boxes   # + lines with bounding boxes on OCR pages
uv run python -m tests.synth tests/fixtures/synthetic   # write synthetic sample documents
```

- **Type detection** uses magic bytes only; the file extension is ignored. Supported: PDF, DOCX, JPG, PNG, WEBP, TIFF (multi-page).
  Rejected: other types, files over `MAX_UPLOAD_MB` (20), more than `MAX_PAGES` (100), corrupt or encrypted files.
- **PDF** (pypdfium2): decided per page. A page with fewer than `MIN_TEXT_CHARS_PER_PAGE` non-space characters
  is rendered at `OCR_DPI` (300) and OCR'd; otherwise the text layer is used.
- **DOCX** (python-docx): headers/footers, paragraphs and tables in document order go in page 1 (`text_layer`).
  Each embedded image in `word/media` becomes another page (`ocr`).
- **OCR** (PaddleOCR 3.7, PP-OCRv5): detection runs once (`PP-OCRv5_mobile_det`), then every text box is read by
  **both** `arabic_PP-OCRv5_mobile_rec` and `en_PP-OCRv5_mobile_rec`, and the results are merged per box:
  - The Arabic model's dictionary contains Latin letters and digits, and it reads pure-English lines well.
  - On a **mixed** line, though, each model reads only its own script. So instead of picking the most confident result,
    the merge combines the Arabic part (Arabic model) with the Latin part (English model). See `app/ocr/merge.py`.
  - Boxes are grouped into lines; Arabic lines are ordered right-to-left.
- `OCR_PREPROCESS=true` turns on OpenCV deskew (line-blob angle, ±15°) and CLAHE contrast.
- Runtime is offline: models load from `%USERPROFILE%\.paddlex\official_models` (or `PADDLE_PDX_CACHE_HOME`),
  and the model-hoster connectivity check is disabled.

Measured on this CPU: about 3 s per A4 page at 300 DPI (warm), and about 5 s to load the models once.
The server detection model (`OCR_DET_MODEL=PP-OCRv5_server_det`) was about 14× slower here (27.6 s per A4 page).

### OCR throughput (engine pool)
- **`OCR_ENGINES` (2):** independent engines, so this many images are OCR'd at once. The scanned pages of one PDF
  are OCR'd in parallel (rendering stays sequential, and at most `OCR_ENGINES` rendered pages wait in memory),
  and a synchronous `/v1/extract` no longer waits behind a job's page. Each engine costs about 0.4 GB of RAM.
  With `OCR_CPU_THREADS` unset the cores are split evenly between the engines. In a container with a CPU quota,
  set `OCR_CPU_THREADS` = CPUs / `OCR_ENGINES` yourself.
- **`OCR_PARALLEL_REC` (true):** the Arabic and English recognizers read the boxes at the same time.
- **Benchmark:** `uv run python scripts/bench_ocr.py [your-file]` compares the settings on your machine and
  shows text similarity to the baseline (it prints timings only, never text). On this laptop (16 threads,
  synthetic 8-page dense scanned A4 PDF):

  | Configuration | s/page | Speed-up | Text vs baseline |
  |---|---|---|---|
  | 1 engine, sequential AR/EN (old default) | 8.5 | 1.0× | identical |
  | 1 engine, parallel AR/EN | 6.0 | 1.4× | identical |
  | 2 engines × 8 threads | 3.8 | 2.2× | identical |
  | 3 engines × 5 threads | 3.7 | 2.3× | identical |
  | 4 engines × 4 threads | 3.8 | 2.2× | identical |
  | rec batch size 16 | 8.8 | 0.96× | 0.997 |
  | 200 DPI | 5.1 | 1.7× | **0.925**: loses text, not used |

  Laptop timings vary by ±25 % between runs (turbo/thermals). Two engines with parallel recognition measured
  about 8 % faster than two engines alone in a back-to-back run, so both are on by default.

## Arabic normalization (Phase 2)

`app/normalize/arabic.py` holds pure, unit-tested functions. Every page's `text` is cleaned; the result
also has `normalizedText` (for matching) and `languages` (`ar` / `en`).

| Step | `text` (original, cleaned) | `normalizedText` (matching only) |
|---|---|---|
| Unicode NFKC (presentation forms such as ﺍﻟﺮﻗﻢ → الرقم, ligature ﻻ → لا) | ✓ | ✓ |
| Arabic-Indic / Persian digits ٠-٩ ۰-۹ → 0-9 (also ٫ ٬ ٪) | ✓ | ✓ |
| Remove tatweel, bidi control marks (LRM/RLM/…), zero-width space | ✓ | ✓ |
| Fix reversed (visual-order) lines from PDF text layers | ✓ | ✓ |
| Strip tashkeel | kept | ✓ |
| أ/إ/آ/ٱ → ا, ى → ي, ة → ه, Persian ک/ی → ك/ي, ، ؛ ؟ → , ; ? | kept | ✓ |

**Reversed-text detection:**
- It uses positional cues. ة/ى end words and ال starts them, so a reversed word shows ة/ى at its start and لا at its end.
  When a PDF stores presentation forms, their initial/final shapes are also used.
- A line with clear cues is fixed on its own. A short line with no cues follows the page: if the page
  as a whole is clearly reversed, its Arabic lines are flipped; otherwise the line is left unchanged.
- Latin words, numbers and dates keep their left-to-right order, and brackets are mirrored.
- It runs on text-layer pages only, because OCR output is already in reading order.

## ID detection (Phase 3)

```powershell
uv run python -m app.cli analyze <file>   # extraction + idDetection (Ollama must be running)
```

`idDetection: { isId, docType, confidence, decidedBy: ["rules"|"llm"|"vision"], reasons, mrz? }`.
`docType` is one of `national_id`, `passport`, `driver_license`, `residence_permit`, `other_id`, `not_id`.

1. **Rules** (`app/detection/rules.py`, `config/id_patterns.yaml`): every signal has a weight and a reason.
   - **MRZ** (`app/detection/mrz.py`): TD1, TD2 and TD3, with ICAO check digits. It tolerates OCR noise
     (case, spaces, `«`, O/0 in numeric fields, lost or extra `<`). The names line carries no check
     digits, so it is accepted at any plausible length.
   - **Keywords** (Arabic and English) are matched on `normalizedText` with spaces removed, because OCR drops spaces.
     Keywords of 4 characters or fewer must match whole words.
   - **Negative keywords** (invoice/letter words) dampen the score.
   - **ID-number regexes** are configured per country. The ones shipped are **examples**; replace them with your countries.
   - Positive weights are combined with noisy-OR: `1 - Π(1 - w)`.
2. **A fully valid MRZ decides** on its own (`decidedBy: ["rules"]`, confidence 0.99), with no LLM call.
   **Parsed MRZ fields** are returned in `idDetection.mrz` whenever an MRZ with at most one failed check digit is
   found (see [MRZ fields](#mrz-fields)).
3. **Text LLM** (`qwen3:8b`) via `LLM_BASE_URL/chat/completions`:
   - The prompt is in English and the model reads Arabic and English.
   - `response_format: json_schema` forces the output shape; thinking is disabled through `LLM_EXTRA_BODY`.
   - Long text keeps its head and tail (`LLM_MAX_CHARS`).
   - Invalid JSON is retried once.
   - The rules and LLM are merged as `P(id) = 0.6·P_llm + 0.4·rule_score` (`RULES_WEIGHT`).
   - If the LLM is unreachable, the result uses rules only, with confidence capped at 0.65.
4. **LLM skip:** a long document with **no rule signal at all** (no MRZ, no ID keyword, no ID-number pattern) is
   `not_id` by rules alone (confidence `LLM_SKIP_CONFIDENCE`, 0.9), with no LLM or vision call. "Long" means at least
   `LLM_SKIP_MIN_PAGES` (5) pages or `LLM_SKIP_MIN_CHARS` (8000) non-space characters; 0 disables a criterion.
   ID documents are one or two short pages, so this saves 30–60 s per contract, report or statement.
   Check the thresholds on your documents with `eval.run_eval` before relying on them.
5. **Vision fallback** (`qwen3-vl:8b-instruct`) runs only when confidence < `VISION_THRESHOLD` (0.7), the input
   has an OCR'd image, and the model is installed. The image is downscaled to `VISION_MAX_SIDE` (1024).
   The results merge as `P = 0.4·P_text + 0.6·P_vision`, or `P_vision` alone if the text gave no evidence.

Nothing from documents is logged: the LLM client logs only model, duration and status.

Measured on this CPU (with only about 5 GB of RAM free): **text LLM 30–57 s per document**; **vision 27–180 s**,
highly variable because the 6.2 GB model competes for RAM. More free RAM, `qwen3-vl:4b-instruct`, or a
GPU server (vLLM) are the ways to make it faster.

### MRZ fields
```json
"mrz": {
  "format": "TD3", "documentCode": "P", "issuingCountry": "UTO", "surname": "TESTER", "givenNames": "SAMPLE",
  "documentNumber": "AB1234567", "nationality": "UTO", "birthDate": "1990-02-01", "sex": "M",
  "expiryDate": "2030-01-01", "optionalData": null, "expired": false, "valid": true, "namesSource": "mrz+text",
  "checks": {"documentNumber": true, "birthDate": true, "expiryDate": true, "personalNumber": true, "composite": true}
}
```
- TD1 (ID cards), TD2 and TD3 (passports). Dates are ISO; birth years in the future are moved to the 1900s.
  `expired` compares `expiryDate` with today's date.
- **`checks`** are the ICAO check digits. A field whose check failed is still returned, because it often holds a
  single OCR error; rely on a field only when its check is `true` (`valid` = all checks passed).
- **Names carry no check digit**, and OCR tends to drop letters in the dense names line (`TESTER` → `TEER`).
  Each name word is therefore compared with the printed text: if exactly one printed word contains it (in order,
  at most 3 letters longer), that word is used and `namesSource` is `mrz+text`; otherwise it is `mrz`.
  Treat names as unverified either way.
- Letter-only fields (countries, nationality, names) undo OCR digit look-alikes (`UT0` → `UTO`).
- These are personal data: they are returned only to the client that uploaded the document, and never logged.

## REST API (Phase 4)

```powershell
$env:API_KEYS = 'change-me'          # comma-separated; with no keys, every /v1 request gets 401
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```
OpenAPI docs: http://127.0.0.1:8000/docs. Click **Authorize** and enter your `X-Api-Key`.

**The API contract** is `docs/openapi.json`: every response field is typed, so client teams can generate clients from
it. `tests/test_api.py` fails when the API no longer matches the file, so every API change is deliberate.
- **Changing the API:** run `uv run python scripts/export_openapi.py`, check that the diff is additive (the rules for
  `/v1` are in `docs/INTEGRATION.md` §14), and add a line to `docs/CHANGELOG.md`.
- **Releases:** bump `__version__` in `app/__init__.py` and `version` in `pyproject.toml` together.
For more than one client system, use the [client registry](#clients-client-registry) instead of `API_KEYS`.

| Method | Path | |
|---|---|---|
| `POST` | `/v1/documents` | Text + ID detection, **async**: multipart `file` (+ optional `callbackUrl`, `externalId`, `metadata`; header `Idempotency-Key`) → `202 {"jobId"}`. 413 too large, 415 unsupported type, 409 idempotency conflict |
| `POST` | `/v1/extract` | **Text only, synchronous**: multipart `file` → `200 {file, pages, fullText, normalizedText, languages, timingsMs}`. No ID detection, no LLM. See [Text only](#text-only-post-v1extract) |
| `GET` | `/v1/jobs/{jobId}[?wait=N]` | `{jobId, status}`; with `wait=N` the request waits up to N s (max `MAX_WAIT_S`, 30) for the job to finish (long-poll). `externalId`/`metadata` are echoed when given. `status` is `queued`, `processing`, `done` or `failed`. When done, it also has `file, pages, fullText, normalizedText, languages, idDetection, timingsMs, processing` (see [Audit trail](#audit-trail)); when failed, `error: {code, message}` |
| `DELETE` | `/v1/jobs/{jobId}` | delete the job and its result now (204) |
| `GET` | `/health` | Full status. 200 `ok` / 503 `degraded`: LLM reachable, models installed, OCR models present, job counts. No key needed |
| `GET` | `/health/ready` | **Readiness probe** for load balancers: 200 when OCR models are present and the worker runs. The LLM is reported but not required (`/v1/extract` works without it) |
| `GET` | `/health/live` | **Liveness probe**: 200 while the process is up |
| `GET` | `/metrics` | Prometheus metrics (no key; disable with `METRICS_ENABLED=false`). See [Metrics](#metrics) |

```bash
curl -s http://127.0.0.1:8000/health
curl -s -H "X-Api-Key: change-me" -F "file=@scan.pdf" http://127.0.0.1:8000/v1/documents
# -> {"jobId":"bbb75af4..."}
curl -s -H "X-Api-Key: change-me" http://127.0.0.1:8000/v1/jobs/bbb75af4...
curl -s -X DELETE -H "X-Api-Key: change-me" http://127.0.0.1:8000/v1/jobs/bbb75af4...
# with a webhook:
curl -s -H "X-Api-Key: change-me" -F "file=@id.jpg" -F "callbackUrl=https://hooks.internal/ocr-done" \
     http://127.0.0.1:8000/v1/documents
```

- **Two stages (`LLM_WORKER_THREADS`, default 0 = off):** with N > 0 a job's text extraction runs on a worker
  thread and its ID detection on one of N LLM threads, so the next job's OCR runs while the LLM works on the
  previous one. At most N extracted jobs wait for the LLM. `timingsMs.llmQueue` shows the wait. Turn it on when the
  LLM runs on another machine or a GPU (vLLM); on one CPU shared with Ollama the two stages compete for cores.
- **Jobs:** a background worker thread (`WORKER_THREADS`, default 1) runs them from a queue (`MAX_QUEUED_JOBS`).
  The store and queue sit behind `app/api/store.py::JobStore`: `JOB_STORE=memory` (default, single process, lost
  on restart) or `JOB_STORE=postgres` (durable and shared, see [PostgreSQL job store](#postgresql-job-store)).
- **Jobs are private to the client that created them** (with `API_KEYS`, each key is a client). Another client gets 404.
- **Webhooks:**
  - The callback is `POST {"event": "job.done"|"job.failed", "deliveryId", "jobId", "status"}`, plus `error.code` on
    failure. **No document data is sent**; the receiver fetches the result with its API key.
  - **Durable delivery (outbox):** the delivery is written to the job store with the result, and a dispatcher in
    every process sends due deliveries from there (`WEBHOOK_THREADS`, 4, at once). Nothing lives only in memory, so
    retries survive restarts and deploys, and with PostgreSQL any replica sends any job's callback. Job processing
    never waits for a receiver.
  - **Retries:** the first after `WEBHOOK_BACKOFF_S` (2 s), doubling up to `WEBHOOK_MAX_BACKOFF_S` (1 h) between
    attempts, for `WEBHOOK_RETRY_WINDOW_S` (24 h) after the job finished, so a receiver in a maintenance window still
    gets its callback. A client can have its own window (`webhook.retry_window_s` in the registry);
    `WEBHOOK_RETRIES` (0 = no limit) caps the attempts.
  - Every attempt, also after a restart, reuses the same `deliveryId` (also sent as `X-Webhook-Id`): receivers
    should de-duplicate on it.
  - **The result waits for the receiver:** a job with a pending webhook isn't purged, and its result is kept
    `RESULT_TTL_S` after the webhook was delivered or given up, so the receiver always has time to fetch it.
  - `GET /v1/jobs/{jobId}` shows the delivery: `webhook: {"state": "pending"|"delivered"|"failed", "attempts"}`.
    `/health/ready` reports `webhooksPending` (all replicas).
  - Signature: `X-Webhook-Signature: sha256=HMAC_SHA256(secret, "<X-Webhook-Timestamp>." + raw_body)`, where the
    secret is the client's own (client registry), or `WEBHOOK_SECRET` with `API_KEYS`.
  - The callback host must be allowed for the client: its `webhook.allowed_hosts` in the registry, or
    `WEBHOOK_ALLOWED_HOSTS` with `API_KEYS` (empty = callbacks disabled, and the secret must be set).
    This is checked again before every attempt: a client that was disabled or lost the host gets no callback.
  - Verifying the signature on the receiver:
    ```python
    import hashlib, hmac, time
    def verify(secret: bytes, ts: str, body: bytes, sig: str) -> bool:
        good = "sha256=" + hmac.new(secret, ts.encode() + b"." + body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(good, sig) and abs(time.time() - int(ts)) < 300
    ```

### Clients (client registry)
Each system that calls the API is a **client** with a stable id. List them in a YAML file and set
`CLIENTS_FILE=config/clients.yaml` (start from `config/clients.example.yaml`; then leave `API_KEYS`, `WEBHOOK_SECRET`
and `WEBHOOK_ALLOWED_HOSTS` empty, or the service refuses to start):
```yaml
clients:
  civil-registry:                     # the id: owns the client's jobs, appears in the logs (client=civil-registry)
    name: "Civil Registry System"
    key_sha256: [<hash of key 1>, <hash of key 2>]  # rotation: list the new key, switch, then drop the old one
    scopes: [documents, extract]      # documents = POST /v1/documents + its jobs, extract = POST /v1/extract
    webhook: {secret_env: WEBHOOK_SECRET_CIVIL_REGISTRY, allowed_hosts: [hooks.civil-registry.internal]}
    limits: {max_queued_jobs: 30, extract_concurrent: 1}   # optional, on top of the global limits
    enabled: true                     # false: its keys get 401, its webhooks stop
```
- **The file holds no secrets**, so it can be reviewed and versioned like code. API keys are stored as SHA-256
  hashes, and each client's webhook signing secret is read from the environment variable it names (or `.env`).
  `uv run python scripts/client_key.py new` prints a new key and its hash; `... hash` hashes an existing key (read
  from the terminal, never an argument); `... secret` prints a new webhook secret.
- **Key rotation keeps the jobs:** they belong to the client id, not to a key.
- **Isolation:** each client has its own webhook secret and callback hosts, so no client can forge or redirect
  another one's callbacks. A missing scope gets `403 forbidden`; the client's own queue quota or concurrency limit
  gets `503 queue_full` / `503 busy`, while the other clients keep working.
- **Invalid files fail the start:** unknown fields (a typo never drops a limit silently), malformed hashes, a hash
  listed twice, or a missing secret variable. Changes take effect on restart.
- **Without `CLIENTS_FILE`**, each `API_KEYS` entry is a client with all scopes and the global `WEBHOOK_*` settings,
  and its id is a hash of the key: the job owner from before the registry, so existing jobs stay visible.
  Moving to the registry changes the owners, so switch when no unfinished jobs are left (or wait `RESULT_TTL_S`).
- Rejected keys are logged as `auth rejected reason=no_key|unknown_key|disabled client=<id> path=... req=...`,
  never the key itself.

### Processing profiles (per entity, no code changes)
Profiles live in the same file (`profiles:`) and are assigned per client (`profiles: [first-is-default, ...]`). A
request may pick another of its client's profiles with the `profile` form field; any other name gets `403 forbidden`.
```yaml
profiles:
  civil-ids:     {id_patterns: config/patterns/civil.yaml, max_pages: 20, result_ttl_s: 86400}
  id-check-only: {return_text: false, return_mrz: false, llm_enabled: false, vision_enabled: false}
  archive-text:  {detect_id: false, include_boxes: true, ocr_dpi: 400}
```

| Setting | Effect |
|---|---|
| `id_patterns` | the entity's own rules file (keywords, ID-number formats) instead of `ID_PATTERNS_PATH` |
| `detect_id` | `false`: `/v1/documents` extracts text only |
| `llm_enabled`, `vision_enabled` | `false`: that model is never called (e.g. the text must not leave for an LLM server) |
| `max_pages` (≤ `MAX_PAGES`), `ocr_dpi`, `min_text_chars_per_page` | extraction limits and quality |
| `include_boxes` | default of the `includeBoxes` request field |
| `return_text`, `return_mrz` | `false`: **data minimization**, see below |
| `result_ttl_s` | this profile's retention instead of `RESULT_TTL_S` |

- **Validated at startup:** unknown keys are errors, infrastructure settings (database, LLM server, keys) can't be
  set, the rules file must load, and `max_pages` can't exceed `MAX_PAGES`. `default` is built in (the global
  settings) and can't be redefined.
- **Data minimization:** `return_text: false` empties `pages[].text`, `fullText` and `normalizedText`, drops `lines`
  and the AI models' free-text reasons (they can quote the document); `return_mrz: false` drops `idDetection.mrz`.
  This happens in the pipeline, so withheld data is **never stored**, only the decision. `processing.withheld` lists
  it. A profile without text can't be used on `/v1/extract` (403).
- **A job keeps its profile:** the profile is copied into the job at upload, so every replica processes it the same
  way, and a later change to the file doesn't touch queued jobs. `processing.profile` and the audit trail
  (`job.received`) name it.

### Integrating from another system
- **Retry-safe uploads:** send `Idempotency-Key: <your id>` (1–128 of `[A-Za-z0-9._:-]`). Repeating the request
  with the same key and the same file returns the **same `jobId`** (header `Idempotent-Replayed: true`), also while the
  queue is full. The same key with a different file gets `409 idempotency_conflict`. Keys are per client and expire
  with their job (`RESULT_TTL_S`).
- **Your references:** `externalId` (≤ 128 characters) and `metadata` (a JSON object, ≤ 2 KB) are stored with the
  job and echoed in `GET /v1/jobs/{id}` and in the webhook, so you don't need your own jobId mapping. They are never
  logged, but they are sent in webhooks: don't put personal data in them.
- **No polling loop:** `GET /v1/jobs/{id}?wait=30` returns as soon as the job is done or failed (or after 30 s).
- **Recommended flow:** `POST /v1/documents` with `Idempotency-Key` → webhook **or** `GET ?wait=30` in a loop →
  read the result → `DELETE /v1/jobs/{id}` when you've stored what you need.

### Errors, request ids, retries (for client systems)
- **Every error** has one shape: `{"error": {"code": "...", "message": "...", "requestId": "..."}}`.
  Program against `code`; `message` is for humans and may change. Validation errors never echo submitted values.

  | HTTP | `code` |
  |---|---|
  | 400 | `empty`, `corrupt`, `callback_not_allowed` |
  | 401 | `unauthorized` (also a disabled client) |
  | 403 | `forbidden` (the client lacks the endpoint's scope) |
  | 404 | `not_found` (also a job of another client) |
  | 409 | `idempotency_conflict` |
  | 413 | `too_large`, `too_many_pages` |
  | 415 | `unsupported_type` |
  | 422 | `invalid_request` (missing/malformed fields) |
  | 500 | `internal_error` (no details, by design) |
  | 503 | `queue_full`, `busy`, both with a `Retry-After` header (`RETRY_AFTER_S`, 5); `handwriting_unavailable` (experimental `/v1/handwriting` only) |

  A failed job reports the same codes in `error.code`, plus `internal_error` and `worker_lost` (its worker died
  `JOB_MAX_ATTEMPTS` times, e.g. a file that crashes the OCR).
- **`X-Request-Id`:** send one (`[A-Za-z0-9._:-]`, at most 64 characters) and it is echoed back and written to the
  log lines of that request and its job (`req=...`). Without one, the service generates it.
- **Retry policy for clients:** retry on 503 after `Retry-After`, and on network errors; don't retry 4xx.

### Text only: `POST /v1/extract`
Use this when you only need the text. The request waits and the text comes back in the same response.
- **What it does:** the same extraction as `/v1/documents`, minus ID detection.
  - PDF text layer, or OCR for scanned pages. DOCX text, tables, headers and footers, plus OCR of embedded images. JPG/PNG/WEBP/TIFF by OCR.
  - Arabic, English and mixed documents: each text box is read by both the Arabic and the English model and merged.
    `normalizedText` holds the Arabic-normalized text, and `languages` lists `ar`/`en` in order of amount.
- **Line positions:** add the form field `includeBoxes=true` (also on `/v1/documents`) and every OCR page gets
  `width`, `height` (pixels of the OCR'd image) and `lines: [{text, confidence, bbox: [x0, y0, x1, y1]}]` in reading
  order. `bbox` is normalized to 0–1 of the page (origin top-left), so it maps onto the page at any resolution;
  it already accounts for OCR downscaling and deskewing. Text-layer pages have no `lines`.
  ```json
  {"page": 2, "method": "ocr", "text": "...", "confidence": 0.93, "width": 2480, "height": 3508,
   "lines": [{"text": "فاتورة ضريبية", "confidence": 0.97, "bbox": [0.512, 0.061, 0.884, 0.083]}, ...]}
  ```
- **It never calls the LLM,** so it works even when Ollama is stopped. `/health` reports `degraded` in that case,
  because it checks the LLM for `/v1/documents`.
- **Limits:**
  - `EXTRACT_MAX_PAGES` (default 20): a larger document gets `413`; send it to `/v1/documents` instead.
  - `EXTRACT_MAX_CONCURRENT` (default 2): more simultaneous requests get `503 busy, retry later`.
  - Speed is about 2–3 s per scanned page on this CPU, and text-layer pages are instant. Give your client a timeout of about 120 s.
- **Privacy:** the file is processed in memory only and never written to disk. The log has one line per request with bytes, pages, ms and status.
- **Errors** (envelope and codes: see [Errors](#errors-request-ids-retries-for-client-systems)): 401 (key), 400 (empty or corrupt), 413 (too large or too many pages), 415 (unsupported type),
  503 (busy), 500 `processing failed` (no details).

```bash
curl -s --max-time 120 -H "X-Api-Key: change-me" -F "file=@letter.pdf" http://127.0.0.1:8000/v1/extract
```
```json
{
  "file": {"name": "letter.pdf", "mimeType": "application/pdf", "pages": 2},
  "pages": [
    {"page": 1, "method": "text_layer", "text": "Dear customer, ...", "confidence": null},
    {"page": 2, "method": "ocr", "text": "فاتورة ضريبية رقم 2024/115 ...", "confidence": 0.93}
  ],
  "fullText": "Dear customer, ...\n\nفاتورة ضريبية رقم 2024/115 ...",
  "normalizedText": "...",
  "languages": ["ar", "en"],
  "timingsMs": {"extract": 3120, "ocr": 3050, "ocrPages": 1, "normalize": 2, "total": 3125},
  "processing": {"version": "0.3.0", "ocr": {"det": "PP-OCRv5_mobile_det", "recAr": "arabic_PP-OCRv5_mobile_rec",
                 "recEn": "en_PP-OCRv5_mobile_rec", "preprocess": false, "dpi": 300}}
}
```

### Privacy measures
- **Logs** contain only jobId, byte sizes, timings, status and error *types*. They never contain file names, text,
  file contents or exception messages (those could contain document text). The tests check this.
- **While a request is read**, the upload stays in RAM. The web framework would normally spool files over 1 MB to a
  temp file; the service raises that threshold to the largest accepted request, and stops reading any upload request
  larger than `MAX_UPLOAD_MB` + 64 KB with `413 too_large` (also chunked uploads that send no `Content-Length`).
- **Uploads** (memory store) are written to `UPLOAD_DIR` (default `%TEMP%\ocr-id-service`) under a random name with mode 0600.
  They are deleted right after processing, also on failure, and leftovers are removed at startup.
  Results live in memory (memory store) or in PostgreSQL, optionally encrypted (see [PostgreSQL job store](#postgresql-job-store)), and are deleted `RESULT_TTL_S` (3600) after the job finishes, or on `DELETE`.
- **Outbound network:** the service only calls `LLM_BASE_URL`, `DATABASE_URL` (postgres store) and the allow-listed webhook hosts. At startup it sets
  `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`, `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`.
  OCR models load from local folders only, and the service never downloads them.
- **Ollama itself:**
  - Run it headless (`ollama serve`) so the tray app's update checker doesn't run.
  - Set `OLLAMA_NO_CLOUD=1` to turn off cloud features.
  - The server also fetches "model recommendations" online, and there is no documented switch for that. To block
    all of Ollama's internet access, add an outbound firewall rule (as admin):
    ```powershell
    New-NetFirewallRule -DisplayName "Block Ollama outbound" -Direction Outbound -Action Block `
      -Program "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
    ```
    Localhost (127.0.0.1) traffic isn't affected. Remove the rule temporarily when you `ollama pull` a new model.

## PostgreSQL job store

`JOB_STORE=postgres` + `DATABASE_URL=postgresql://user:pass@host:5432/db` (PostgreSQL 13+). The tables
(`ocr_jobs`, `ocr_job_uploads`, `ocr_idempotency_keys`) are created at startup.

- **Schema migrations:** numbered SQL files in `app/api/migrations/` (`001_init.sql`, ...). At startup the missing ones
  are applied in order, in one transaction, under an advisory lock, and recorded in `ocr_schema_migrations`.
  A database created before migrations existed is adopted as is (`001` is idempotent). Never edit a deployed
  migration, and keep each one backward compatible so older replicas keep working during a rolling deploy
  (see `app/api/migrations/README.md`).

- **Durable:** jobs and results survive restarts and deploys; clients keep their job ids.
- **Scales out:** upload bytes are stored in the database, so any replica's worker can process any job. Workers
  claim jobs with `FOR UPDATE SKIP LOCKED`, so a job is never processed twice at the same time. Run more containers
  (each loads its own OCR models); prefer containers over `uvicorn --workers` (metrics are per process).
- **Crash recovery:** workers send heartbeats. A job whose worker stopped for `JOB_STALE_S` (120) goes back to the
  queue; after `JOB_MAX_ATTEMPTS` (2) claims it fails with `worker_lost` (and its webhook is scheduled).
- **Graceful shutdown:** on a normal stop (deploy, restart), jobs still being processed after a few seconds go back
  to the queue at once, without counting the interrupted attempt, so deploys never push a job towards `worker_lost`.
  Pending webhooks stay in the database for the next process.
- **Privacy:**
  - Uploads are deleted right after processing, and finished jobs after `RESULT_TTL_S`, as with the memory store.
  - Set `RESULT_ENCRYPTION_KEY` (base64 of 32 bytes): uploads, results, file names and metadata are then stored
    AES-256-GCM encrypted (bound to their job id), so dumps, backups and WAL contain no readable document data.
    Plain columns hold only ids, sizes, states, timestamps and error codes. Keep the key out of the backups.
  - Without the key, deleted rows still exist in WAL and backups until those rotate. Use the key in production.
  - Use a dedicated database user that owns only these tables, keep the server on an internal network, and keep
    `log_statement=none` (the compose file does this).
- **docker-compose** runs `postgres:17` on the internal network and sets `JOB_STORE=postgres`; set `POSTGRES_PASSWORD`
  (and `RESULT_ENCRYPTION_KEY`) in `.env`.
- **Tests:** with `TEST_DATABASE_URL` set (e.g. `postgresql://postgres:dev@127.0.0.1:5432/postgres`), `uv run pytest`
  also runs the PostgreSQL tests. They drop and recreate the `ocr_*` tables (including `ocr_schema_migrations`), so
  use a throwaway database.

## Audit trail

Every request and job leaves a record of **who sent which document when, what happened to it, and who read or
deleted the result**, also after the job itself is gone. Records hold metadata only: never document text,
file names, MRZ fields, the LLM's free-text reasons (they can quote the document), caller `metadata`, or
`externalId` in clear.

- **Where:** table `ocr_audit_events` (migration `003`) with the PostgreSQL store; the latest 10 000 events in RAM with
  the memory store (development). Columns: `at`, `event`, `client_id`, `job_id`, `request_id` (the `X-Request-Id`),
  `data` (jsonb).
- **Events:**

  | event | when | `data` |
  |---|---|---|
  | `job.received` | `POST /v1/documents` stored a job | endpoint, size, sha256, mimeType, callback, includeBoxes, idempotent, externalIdHash |
  | `job.replayed` | the same `Idempotency-Key` and file came again | sha256 |
  | `job.started` | a worker claimed the job | attempt, worker |
  | `job.done` / `job.failed` | processing ended | ms, attempt, result facts / error code |
  | `job.requeued` | its worker died, the job is queued again | reason, attempts |
  | `job.fetched` | `GET /v1/jobs/{id}` returned the finished job | status |
  | `job.deleted` | `DELETE /v1/jobs/{id}`, or the `RESULT_TTL_S` purge | by (`client` / `ttl`) |
  | `webhook.delivered` / `webhook.failed` | the callback ended | attempts, reason |
  | `extract.done` | `POST /v1/extract` returned text | endpoint, size, sha256, mimeType, ms, result facts |
  | `request.error` | any `/v1` request answered with an error (also 401s) | method, route, status, code |
  | `admin.*` | an action in the [admin dashboard](#admin-dashboard) (login, configuration, keys, ...) | actor, and what changed (never passwords, keys or secrets) |

  Result facts: pages, pageMethods (`{"ocr": 2, "text_layer": 1}`), languages, timingsMs, `processing` (below),
  ocrConfidence (mean OCR confidence), fallbacks (models ID detection went on without), and for ID detection isId,
  docType, confidence, decidedBy, mrzFound, mrzValid.
- **Can't be lost:** job lifecycle events are written in the same transaction as the state change. The other events
  are best effort: a failed write is logged (`audit write failed event=...`) and counted in
  `ocr_audit_write_failures_total`, and never fails the request.
- **Retention:** `AUDIT_RETENTION_DAYS` (365; 0 = forever), purged by the worker's maintenance sweep.
- **Finding records** (SQL on the job store database):
  ```sql
  SELECT at, event, client_id, data FROM ocr_audit_events WHERE job_id = '<jobId>' ORDER BY id;     -- a job's history
  SELECT at, event, job_id, data FROM ocr_audit_events WHERE request_id = '<X-Request-Id>' ORDER BY id;
  SELECT at, event, job_id FROM ocr_audit_events WHERE data->>'externalIdHash' = '<hash>' ORDER BY id;
  SELECT at, job_id FROM ocr_audit_events WHERE data->>'sha256' = '<sha256 of a file>';  -- was this file processed?
  ```
  `externalIdHash` is an HMAC keyed from `RESULT_ENCRYPTION_KEY`; `uv run python -m app.cli audit-ref <externalId>`
  prints the value to search for.
- **Provenance:** every result has `processing`: the service `version` (`app/__init__.py`), the OCR models used
  (`ocr`: det, recAr, recEn, preprocess, dpi), the rules file version (`rules`: start of its SHA-256), and the LLM
  and vision models if they were called. A part that wasn't used is left out. It is also in `job.done`, so "which
  model and which rules decided this?" can be answered after the result is deleted.

## Logging

`LOG_FORMAT=text` (default) or `json` (the Docker image's default): one JSON object per line, ready for a log
collector, e.g.
```json
{"ts": "2026-11-21T09:15:02.481+00:00", "level": "INFO", "logger": "app.api.worker",
 "msg": "job finished job=4f1c... client=civil-registry status=done bytes=48211 ms=3120 req=order-77-1",
 "request_id": "order-77-1", "job_id": "4f1c...", "client_id": "civil-registry"}
```
- **Every line carries its ids:** the HTTP middleware sets `request_id` and `client_id` for a request; the worker
  and the webhook dispatcher set `job_id`, `request_id` (the upload's `X-Request-Id`) and `client_id` for the job they
  work on. So every line written meanwhile has them, also from the LLM client or the store. Search by the
  `X-Request-Id` a client quotes to see the whole story: upload, processing, webhook.
- uvicorn's own lines (startup, access log) use the same format in JSON mode. `LOG_LEVEL` (INFO) sets the level.
- **Privacy:** messages never contain document text, file names, keys or exception messages (the tests check this);
  for an exception a JSON line has only its type (`exc_type`), never the message or the traceback.

## Metrics

`GET /metrics` (Prometheus text format, no API key; serve it on an internal network only). Labels are enums, route
templates and client ids only (bounded by the client registry; `-` = no valid key), never job ids, file names or text.

| Metric | Meaning |
|---|---|
| `ocr_http_responses_total{route,status,client}` | responses per route template, HTTP status and client |
| `ocr_jobs_finished_total{status,client}`, `ocr_job_duration_seconds` | finished jobs and their processing time |
| `ocr_job_queue_wait_seconds` | how long jobs waited in the queue before a worker took them |
| `ocr_stage_duration_seconds{stage}` | `extract`, `ocr`, `llm`, `vision` time per document |
| `ocr_pages_total{method}` | pages by `text_layer` / `ocr` |
| `ocr_id_decisions_total{doc_type,decided_by}` | detection results and the last stage that decided |
| `ocr_llm_fallbacks_total{stage}` | detection went on without the `llm` (rules only) or `vision` model: failed or missing |
| `ocr_extract_requests_total{status,client}`, `ocr_webhooks_total{result,client}` | `/v1/extract` calls, finished webhook deliveries |
| `ocr_audit_write_failures_total` | audit events that could not be written (alert on any increase) |
| `ocr_jobs{status}` | jobs in the store by status (queue depth = `status="queued"`) |
| `ocr_queue_oldest_age_seconds`, `ocr_processing_oldest_age_seconds` | wait of the oldest queued job; run time of the longest-running job (a hung job) |
| `ocr_webhooks_pending` | deliveries not yet delivered or given up |
| `ocr_dependency_up{dependency}` | 1/0 for `store` (database), `ocr_models`, `worker`, `llm` (the text model), for this replica |

- **Aggregating replicas:** counters and histograms count each replica's own work: `sum` them. The gauges read from
  the shared job store (`ocr_jobs`, `ocr_queue_oldest_age_seconds`, `ocr_processing_oldest_age_seconds`,
  `ocr_webhooks_pending`) are the same on every replica: use `max`, never `sum`.
- **Alerts:** `docs/alerts.yml` has example Prometheus rules for these: replica down or not ready, LLM unavailable
  or falling back, queue backlog, a stuck job, job failure rate, 500s, a client being shed with 503, webhooks given
  up or piling up, and audit write failures. A test checks that every metric they use exists.

## Admin dashboard

A web dashboard at `/admin` for the people who run the service. It gives visibility into documents, resources, AI
models, client systems, logs and alerts, and control over the configuration. Everything shown is measured, not mocked.
It is built in (`app/admin/`) with no build step and no external resources (no CDN), so it works offline like the
rest of the service. The pages work in light and dark mode, and every chart has a table view.

### Turning it on
```powershell
copy config\admins.example.yaml config\admins.yaml
uv run python scripts/admin_user.py hash      # prints a scrypt hash for a password (typed twice): put it in admins.yaml
uv run python scripts/admin_user.py secret    # prints a value for ADMIN_SESSION_SECRET
$env:ADMIN_USERS_FILE = 'config/admins.yaml'; $env:ADMIN_SESSION_SECRET = '<the secret>'
uv run uvicorn app.api.main:app --host 127.0.0.1 --port 8000   # then open http://127.0.0.1:8000/admin
```
- **Users** are listed in `ADMIN_USERS_FILE` with a role: `admin` (may change configuration and keys, delete jobs,
  acknowledge alerts) or `viewer` (read-only). The file holds only password hashes. It is re-read when it changes:
  removing a user or changing a role takes effect on their next request, and a new password hash ends their sessions.
- **Sessions** are cookies signed with `ADMIN_SESSION_SECRET` (shared by all replicas), valid `ADMIN_SESSION_HOURS` (8).
  With `ENVIRONMENT=production` the secret must be at least 32 characters. The cookie is then `Secure`, so serve
  `/admin` over HTTPS (a TLS-terminating proxy).
- **Protection:** failed logins slow down after 5 attempts per user and address. Every change needs the
  `X-Admin-Request` header and a same-origin request (CSRF). Pages are sent with a strict Content-Security-Policy.
  Keep `/admin` on an internal network, like `/metrics`.
- **Every admin action is audited** (`admin.login`, `admin.login_failed`, `admin.config_saved`, `admin.key_created`,
  `admin.key_revoked`, `admin.secret_created`, `admin.job_deleted`, `admin.alert_acked`) with `data.actor`. Passwords,
  keys and secrets are never recorded. For a failed login, the user name is recorded only if that user exists, since a
  mistyped name can be a password.
- The admin API (`/admin/api/*`) is not part of the client API contract and is left out of `docs/openapi.json`.

### What it shows
| Section | Content | Source |
|---|---|---|
| **Overview** | Documents processed, success rate, failures, queue and processing (live), p50/p95 processing time, pages, IDs found, throughput, dependency health, replicas, open alerts, recent activity | audit trail, job store, samples |
| **Documents** | Every submitted document: async jobs, sync extracts and rejected uploads. Filter by status, source, client, type and period; search by job id, request id, SHA-256 or `externalId` (hashed before the search). Each document has its lifecycle timeline (received → started → done → fetched → webhook → deleted), step timings (queue wait, extraction, OCR, normalization, rules, LLM, vision), result metadata (pages, methods, languages, OCR confidence, decision, confidence, decided by, MRZ found/valid, model fallbacks, models and rules versions), attempts, webhook state and error code. Admins can delete a job | audit trail + job store |
| **System** | Per replica: CPU (total and per core, load), memory, swap, the service's RSS and threads, disks of the app and upload folders, disk and network I/O, NVIDIA GPUs (utilization, VRAM, temperature, power), dependency health, jobs in progress, OCR engines busy, LLM calls in flight. History charts; PostgreSQL size and connections | `psutil`, NVML (`nvidia-ml-py`) |
| **AI models** | OCR models, languages and engines (loaded / busy); text and vision LLM: reachable, installed, held in RAM/VRAM and the share on the GPU (Ollama `/api/ps`), calls, errors, response time; inference times p50/p95 per stage, throughput, who decided, document types, confidence distribution, fallbacks; accuracy from the latest `eval.run_eval` run | LLM client, OCR pool, audit trail, `EVAL_OUT_DIR` |
| **Integrations** | Every client system: health, authentication (enabled, keys, scopes), last request, uploads, sync extracts, errors by code, 503 shedding, average job time, webhooks delivered / given up / pending, queued jobs; requests with unknown keys. Per client: usage chart and request history | client registry, audit trail |
| **Logs & audit** | Log records of this replica (all levels, live tail) or the warnings and errors of all replicas; the audit trail with filters and CSV export; **trace**: everything about one `X-Request-Id` or job id (audit events and log lines) on one page | log buffer, `ocr_log_records`, audit trail |
| **Configuration** | Settings, clients and keys, processing profiles, ID rules and document types, version history | see below |
| **Alerts** | Firing, acknowledged and resolved alerts; the rules | `ocr_alerts` |

**Privacy:** the dashboard shows metadata only, never document text, file names, MRZ fields, caller `metadata` or
`externalId`. Log messages already exclude document content. Secrets (`DATABASE_URL`, keys, `RESULT_ENCRYPTION_KEY`,
...) are shown only as "set" or "not set". A test calls every dashboard endpoint after processing documents and checks
that none of this appears.

### Real time
- The live view updates every `SAMPLE_INTERVAL_S` (2 s) over Server-Sent Events (`/admin/api/stream`). It falls back
  to polling if a proxy buffers the stream. Each stream ends after 30 s and the browser reconnects, so a deploy never
  waits for open dashboards.
- Each replica stores a sample every `SAMPLE_PERSIST_S` (10 s) for `SAMPLE_RETENTION_H` (24 h). The dashboard uses them
  to show every replica, whichever one serves the page, and the history. A replica that stops sending samples
  raises `replica_missing`; one that shuts down cleanly doesn't.
- Logs: each process keeps its latest `LOG_BUFFER_SIZE` records in RAM. WARNING and above are also stored for
  `LOG_RETENTION_DAYS` (30). A failed log write is dropped, never retried in a loop.

### Configuration from the dashboard
Changes are validated with the same code the service uses at startup. They are stored as **versions** (table
`ocr_config_versions`: who, when, comment), and every replica applies them within `CONFIG_POLL_S` (5 s). The history
shows the diff of every version and can restore any of them; a restore is saved as a new version. Two admins editing
at once don't overwrite each other: the second save gets `409` and reloads.
- **Settings:** environment variables stay the base, and the dashboard stores overrides. Each setting shows its value
  and where it comes from (default, env or dashboard).
  - **Live** settings change at once: thresholds and weights, LLM and vision on/off, models, timeouts, page and
    queue limits, DPI, retention, webhook retries, alert thresholds, and `ID_PATTERNS_PATH`.
  - **Restart** settings are applied when a replica starts, and are shown as pending until then: OCR models,
    engines, threads, worker threads, sync concurrency.
  - Infrastructure can't be changed here: database, job store, LLM server URLs, keys and secrets, environment, upload
    size.
- **Clients and keys:** create, edit, disable or delete client systems (scopes, profiles, limits, webhook hosts and
  retry window).
  - **API keys:** generate a key (shown once; only its hash is stored) or revoke one. A client's last key can't be
    revoked: disable the client instead.
  - **Webhook secrets:** generate one (shown once, stored encrypted with `RESULT_ENCRYPTION_KEY` in
    `ocr_client_secrets`), or keep `secret_env`.
  - **Where the registry lives:** the first change moves it from `CLIENTS_FILE` into the database. From then on the
    dashboard version is used and `CLIENTS_FILE` is no longer read. **Export YAML** downloads it in the
    `CLIENTS_FILE` format, for version control.
  - **Not with `API_KEYS`:** move to a client registry first, as for `CLIENTS_FILE`.
- **Profiles:** create, edit or delete them (a profile still used by a client can't be deleted).
- **ID rules and document types:** see the active rules (keywords and patterns per document type), and create rule
  sets as YAML. Each set is validated, stored as `db:<name>`, and can be used by `ID_PATTERNS_PATH` or a profile's
  `id_patterns`. The document types themselves are fixed (`national_id`, `passport`, `driver_license`,
  `residence_permit`, `other_id`, `not_id`). Rules give the detector evidence for them.
- A replica that can't apply a new version (e.g. a webhook `secret_env` missing in its environment) keeps the previous
  configuration and raises `config_apply_failed`. With `ENVIRONMENT=production`, a replica refuses to start with a
  stored configuration it can't apply.
- With `JOB_STORE=memory` (development) changes apply but are lost on restart; the dashboard says so.

### Alerts
Every replica evaluates the rules every `ALERT_EVAL_S` (15 s). An alert is identified by its rule and scope (the
platform, a client, a replica), and there is at most one open alert per rule and scope, also with many replicas.
Alerts are shown in the dashboard: a badge, a toast when one fires, and the Alerts page, where admins acknowledge
them. They resolve on their own when the condition clears. There are no outbound notifications; for those, use the
Prometheus rules in `docs/alerts.yml`.

| Rule | Fires when |
|---|---|
| `ocr_models_missing`, `worker_down` (critical) | a replica has no OCR models / no running job worker |
| `llm_unavailable`, `vision_unavailable` | the text / vision model is enabled but not available |
| `queue_backlog`, `job_stuck` (critical) | the oldest queued job waited over `ALERT_QUEUE_AGE_S` / a job runs over `ALERT_JOB_STUCK_S` |
| `job_failure_rate`, `server_errors`, `model_fallbacks` | over `ALERT_FAILURE_RATE` of jobs failed (≥ 5 jobs), 500s, or LLM/vision fallbacks in the last 15 min |
| `load_shedding`, `webhooks_failed` | a client got ≥ 20 × 503 in 10 min / had webhooks given up in the last hour |
| `webhooks_piling` | more than 100 webhook deliveries pending |
| `cpu_high`, `memory_high`, `disk_high`, `gpu_memory_high` | over `ALERT_CPU_PCT` / `ALERT_MEM_PCT` / `ALERT_DISK_PCT` / `ALERT_GPU_MEM_PCT` for `ALERT_FOR_S` |
| `audit_write_failures`, `config_apply_failed` (critical) | audit events couldn't be written / a configuration version couldn't be applied |
| `replica_missing` | a replica sent no sample for 60 s without shutting down |

Thresholds are settings, editable live in the dashboard (Configuration → Monitoring & alerts).

### Storage
With `JOB_STORE=postgres`, migration `006_admin.sql` adds `ocr_instance_samples`, `ocr_alerts`, `ocr_log_records`,
`ocr_config_versions` and `ocr_client_secrets`, plus an index on the audit trail (`event, at`). None of them holds
document content. Old samples and log records are purged automatically. Document history comes from the audit trail,
so it covers `AUDIT_RETENTION_DAYS`, not just `RESULT_TTL_S`. Two facts were added to `job.done` for this:
`fallbacks` (models that failed or were missing) and `ocrConfidence` (mean OCR confidence).

## Evaluation on your real documents (Phase 5)

Keep the documents **outside** this repository; the script refuses to run on a folder inside it.
Write a `labels.csv` (UTF-8, header required). `file` is relative to the folder, and `label` is one of
`national_id, passport, driver_license, residence_permit, other_id, not_id`:
```csv
file,label
scans/id_001.jpg,national_id
letters/offer_2024.pdf,not_id
```
```powershell
uv run python -m eval.run_eval D:\private\docs --labels D:\private\docs\labels.csv
uv run python -m eval.run_eval D:\private\docs --labels D:\private\docs\labels.csv --no-llm --limit 20  # quick, rules only
```
- **Report contents:** OCR time per page (mean/median/p95/max), LLM, vision and total time per document,
  ID-vs-not-ID accuracy, precision, recall and F1, document-type accuracy, which stage decided, a confusion
  matrix, and the misclassified files.
- **Outputs** go to `eval/out/<timestamp>/` (git-ignored): `report.md`, `results.csv`, `summary.json`.
  They contain your **file names** and predictions, but no document text.
- **The console** shows only counters and predictions, never file names.
- Files without a label are timed but not scored. Files that fail to process count as `error` in the matrix.

## Handwriting OCR (experimental)

`POST /v1/handwriting` reads **handwritten** Arabic (the priority) and English with
[`sherif1313/Arabic-English-handwritten-OCR-v3`](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-v3)
(Apache-2.0), a full fine-tune of Qwen2.5-VL-3B-Instruct. It is a separate, experimental endpoint for testing the model
on real documents before deciding whether to merge it into the pipeline:
- `/v1/extract`, `/v1/documents` and their results are unchanged and never use this model;
- its own code (`app/handwriting/`), its own limit (one request at a time per process), its own response model;
- off by default (`HANDWRITING_ENABLED=false`: the endpoint answers `503 handwriting_unavailable`).

The model card's accuracy figures are self-reported and contradict each other; judge the model with the evaluation
below, on your own hand-checked samples.

### Set up and call it
```powershell
uv sync --extra handwriting                          # torch (CPU build) + transformers (GPU: --extra handwriting-cuda)
uv run python -m app.cli download-handwriting-model  # ~7.5 GB into the Hugging Face cache (needs internet, once)
```
Note: a plain `uv sync` (without `--extra handwriting`) removes torch and transformers again. On a Linux NVIDIA
server use `--extra handwriting-cuda` instead (CUDA 12.6 torch; the two extras can't be installed together). In
Docker, only the GPU setup (`docker-compose.gpu.yml`, see [Docker deployment](#docker-deployment)) includes it.
Then set `HANDWRITING_ENABLED=true` in `.env`, start the service, and check `GET /health/handwriting`
(`depsInstalled`, `modelDownloaded`, `loaded`). The model loads on the first request (or at startup with
`HANDWRITING_WARMUP=true`); on CPU it needs about 15 GB of RAM (float32).
```powershell
curl.exe -H "X-Api-Key: $env:KEY" -F "file=@D:\private\note.jpg" http://127.0.0.1:8000/v1/handwriting
```
The response (shape only; the numbers are placeholders, not measurements):
```json
{"file": {"name": "note.jpg", "mimeType": "image/jpeg", "pages": 1},
 "pages": [{"page": 1, "text": "...", "confidence": null, "ms": 0, "warnings": []}],
 "fullText": "...", "languages": ["ar"], "timingsMs": {"decode": 0, "model": 0, "total": 0},
 "model": {"backend": "transformers", "name": "sherif1313/Arabic-English-handwritten-OCR-v3", "device": "cpu",
           "maxPixels": 1003520, "segmenter": "whole_page"}}
```
- **Inputs:** JPG, PNG, WEBP, TIFF (multi-page) and PDF, at most `HANDWRITING_MAX_PAGES` (5) pages. Every PDF page is
  rendered at `HANDWRITING_DPI` (200), text layer or not. DOCX gets `415`. The `extract` scope is required.
- **Speed:** not measured on this machine yet. Expect a minute or more per page on CPU (about 1,300 input tokens per
  page with the default `HANDWRITING_MAX_PIXELS`, plus up to `HANDWRITING_MAX_NEW_TOKENS` generated). Lower
  `HANDWRITING_MAX_PIXELS` for speed, raise it for small writing; the eval shows the effect.
- **`text` is the model's output as read:** only line endings and surrounding whitespace are tidied; no Arabic or
  Unicode normalization.
- **`confidence` is always `null`:** the model gives no calibrated confidence, and none is made up.
- **No invented text:** decoding is always greedy (the model repo's sampling defaults are overridden), and the card's
  `min_new_tokens=50` is not used (it forces text out of empty pages). Pages without ink are not sent to the model
  (`"skipped": "blank"`); the check is deliberately lenient, so a small signature or a single word still goes to the
  model, and an empty page with visible dirt is not caught either. `warnings` flags pages to review: `truncated`
  (the output hit `HANDWRITING_MAX_NEW_TOKENS`) and `repetition_loop` (the output repeats itself).
- **Whole page per call** (`HANDWRITING_SEGMENTER=whole_page`). Line or region cropping can be added later as another
  segmenter (`app/handwriting/regions.py`) without changing the API.
- Errors: `503 handwriting_unavailable` (disabled, extra not installed, model not downloaded, server unreachable; the
  message says which), `503 busy` (another request is being read; retry after `Retry-After`).

### Evaluate it on your documents
Collect real samples **outside** the repository, with a `manifest.csv` and a hand-typed transcript per file:
```csv
file,category,gt_file,verified_by,verified_at,notes
letters/a01.jpg,ar_handwritten,letters/a01.txt,reviewer-1,2026-11-20,
misc/blank1.png,blank,,reviewer-1,2026-11-20,empty page
misc/sig1.jpg,signature_only,misc/sig1.txt,reviewer-1,2026-11-21,only the name is readable
```
- **Ground truth must be typed and checked by a person,** never copied from a model's output. Only rows with
  `verified_by` filled in are run and scored; the others are listed as unverified.
- Categories: `ar_handwritten`, `en_handwritten`, `mixed`, `blank`, `signature_only`, `stamp_only`, `sparse` (a few
  words), `printed_control`. Include the "no text" kinds: they show whether the model invents text.
```powershell
uv run python -m eval.run_handwriting_eval D:\private\hw-samples                    # model settings from .env
uv run python -m eval.run_handwriting_eval D:\private\hw-samples --no-blank-check   # the model alone on empty pages
uv run python -m eval.run_handwriting_eval D:\private\hw-samples --save-text        # + each output, for review
```
The report (`eval/out/handwriting-<timestamp>/`: `report.md`, `results.csv`, `summary.json`) gives per category:
- **raw CER/WER:** only Unicode NFC and whitespace are evened out; every other difference counts;
- **normalized CER/WER, separately:** tatweel and tashkeel removed, Arabic-Indic digits and Arabic punctuation unified,
  and nothing else. Letters are never folded (أ/إ/آ/ا, ى/ي, ة/ه and the hamza carriers stay distinct), because
  confusing them is a real recognition error;
- **the gap** between the two, with samples over 5 points flagged (the normalized score forgives a lot there);
- **invented text** on `blank`, `signature_only` and `stamp_only` pages (how many produced output, excess
  characters, how many the blank check skipped), and the share of output characters not in the transcript;
- time per page and the warnings.

### OpenAI-compatible backend (vLLM): implemented, not verified
`HANDWRITING_BACKEND=openai` sends each page to `HANDWRITING_BASE_URL/chat/completions` with the same prompt and
explicit greedy settings (`temperature: 0`, `max_tokens`, `repetition_penalty`). The client side is tested; serving
**this** model with vLLM is not verified yet (vLLM needs Linux / WSL2 with an NVIDIA GPU). The repository looks
compatible (native `Qwen2_5_VLForConditionalGeneration`, standard chat template and processor, no custom code), but
don't rely on it until these steps pass:
1. On the GPU machine, start it, checking each flag against `vllm serve --help` of the installed version:
   `vllm serve sherif1313/Arabic-English-handwritten-OCR-v3 --port 8002 --max-model-len 8192
   --limit-mm-per-prompt '{"image": 1}' --mm-processor-kwargs '{"max_pixels": 1003520}' --generation-config vllm`
   (`--generation-config vllm` stops the repo's sampling defaults from being applied).
2. Run the evaluation on the same samples with `HANDWRITING_BACKEND=transformers` and with
   `HANDWRITING_BACKEND=openai`, `HANDWRITING_BASE_URL=http://gpu-server:8002/v1`, and compare: the texts should be
   the same or the CER within noise.
3. Then record the exact vLLM version and command here.

## Docker deployment

Docker is **not** needed for local development. For a server:
```bash
cp .env.example .env                                  # set API_KEYS, POSTGRES_PASSWORD, RESULT_ENCRYPTION_KEY, WEBHOOK_* ...
docker compose --profile setup run --rm ollama-pull   # one-time: pull qwen3:8b + qwen3-vl:8b-instruct
docker compose up -d --build
curl http://127.0.0.1:8000/health
```
- **Production mode:** the image sets `ENVIRONMENT=production`, and the service then **refuses to start** with an
  unsafe configuration. The error lists every problem (never the secret values):
  - the memory job store, or no `RESULT_ENCRYPTION_KEY`;
  - an API key or webhook secret (`WEBHOOK_SECRET`, or a client's in `CLIENTS_FILE`) shorter than 32 characters or
    still `change-me`, or a `CLIENTS_FILE` without an enabled client;
  - `WEBHOOK_ALLOWED_HOSTS` without `WEBHOOK_SECRET`, a `change-me` password in `DATABASE_URL`, or `OCR_ALLOW_DOWNLOAD=true`.

  Generate keys and secrets with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Set
  `ENVIRONMENT=development` in `.env` only for a throwaway test deployment.
- The image (`Dockerfile`) is python:3.12-slim plus uv 0.13.9, with the PaddleOCR models baked in at build time.
  It runs as a non-root user and is fully offline at runtime. It is **x86_64 only**: PaddlePaddle has no Linux arm64 wheels on PyPI.
- **Ollama** (`ollama/ollama:0.40.16`) sits on an `internal: true` network, so at runtime it has **no internet access**.
  That also solves the "model recommendations" call. Only the one-time `ollama-pull` helper is on a network with internet.
- **`/tmp` is a `tmpfs` (RAM)**, so temp files and `UPLOAD_DIR` never touch persistent disk. The API is published on
  `127.0.0.1:8000`; change the host IP in `docker-compose.yml` to expose it.
- The container `HEALTHCHECK` uses `/health/ready`, so an Ollama outage doesn't mark the app unhealthy.
- **NVIDIA GPU server:** add `COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml` to `.env`. Ollama and the
  handwriting model then run on the GPU (PaddleOCR stays on the CPU), the image is built with
  `UV_EXTRA=handwriting-cuda`, and a one-time `docker compose --profile setup run --rm handwriting-pull` puts the
  handwriting model into the `hf-models` volume. Step by step for AWS EC2 g6 on Ubuntu 22.04, including host setup
  (`scripts/ec2_setup.sh`) and `.env` generation (`scripts/make_env.py`): [docs/DEPLOY-EC2.md](docs/DEPLOY-EC2.md).
- These files were written without Docker available on the development machine, so **run the build once** on the target host.

## Switching the LLM to vLLM on a GPU server

The service only talks to an OpenAI-compatible API, so switching is configuration only. vLLM serves one
model per server, so run one for text and (optionally) one for vision:
```bash
# on the GPU server (Linux, NVIDIA)
pip install vllm
vllm serve Qwen/Qwen3-8B --port 8000 --max-model-len 8192 --api-key "$VLLM_KEY"
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 8192 --api-key "$VLLM_KEY"
```
```dotenv
LLM_BASE_URL=http://gpu-server:8000/v1
LLM_TEXT_MODEL=Qwen/Qwen3-8B
LLM_VISION_BASE_URL=http://gpu-server:8001/v1
LLM_VISION_MODEL=Qwen/Qwen3-VL-8B-Instruct
LLM_API_KEY=<VLLM_KEY>
LLM_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}   # vLLM way to disable Qwen3 thinking
LLM_TIMEOUT_S=60
LLM_VISION_TIMEOUT_S=60
```
- **What stays the same:** `response_format: json_schema` is supported by vLLM's structured outputs.
  Both models are Apache-2.0 (`Qwen/Qwen3-8B`, `Qwen/Qwen3-VL-8B-Instruct`).
- **Privacy:** document text is then sent over your internal network to the GPU server. Keep it on a trusted network
  (or behind TLS), and keep the vLLM server itself offline once its models are downloaded (`HF_HUB_OFFLINE=1`).
- **Check after switching:** `GET /health` should report `textModel: true`, and the live tests should pass
  (`uv run pytest tests/test_live_llm.py`).

## Upgrading OCR to PaddleOCR-VL

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) (Apache-2.0, about 0.9B parameters) is a document VLM.
It supports 109 languages, **Arabic included**, and handles layout, tables and reading order much better than
line-by-line OCR. The installed PaddleOCR 3.7 already ships it (`from paddleocr import PaddleOCRVL`, default
pipeline `v1.6`). It is far too slow on this CPU, so use it with a GPU:

1. On the GPU server, start the VLM inference server (vLLM backend; vLLM doesn't run natively on Windows):
   ```bash
   pip install "paddleocr[doc-parser]"
   paddleocr genai_server --model_name PaddleOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8118 --backend vllm
   ```
2. Add an engine next to `app/ocr/engine.py` with the same interface, `recognize(img) -> OcrOutput(text, confidence, boxes)`:
   ```python
   from paddleocr import PaddleOCRVL
   pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://gpu-server:8118/v1")
   res = pipeline.predict(img)[0]          # layout detection runs locally, the VLM runs on the server
   text = res.markdown["markdown_texts"]   # or res.json for blocks with positions
   ```
   Then choose the engine with a setting (e.g. `OCR_ENGINE=pp-ocrv5|paddleocr-vl`) in `app/pipeline.py`.
   Extraction, normalization and detection stay unchanged.
3. **Compare before switching.** Run `eval.run_eval` on the same folder with both engines, and compare text quality
   (especially mixed Arabic/English lines and Arabic-Indic digits, the known PP-OCRv5 weaknesses) and time per page.
4. **Offline:** download the PaddleOCR-VL models once, then keep the server offline. The pipeline also downloads its
   layout model on first use; run it once while online, and keep `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`.

## Configuration
All settings come from env vars or a local `.env` (copy `.env.example`, which lists every setting with
its default). They are defined in `app/config.py`. The main groups:

| Group | Settings |
|---|---|
| Service | `ENVIRONMENT` (`development` / `production`: startup safety checks, see [Docker deployment](#docker-deployment)) |
| Logging | `LOG_FORMAT` (`text` / `json`), `LOG_LEVEL` (see [Logging](#logging)) |
| LLM backend | `LLM_BASE_URL`, `LLM_TEXT_MODEL`, `LLM_VISION_BASE_URL`, `LLM_VISION_MODEL`, `LLM_API_KEY`, `LLM_EXTRA_BODY`, `LLM_TIMEOUT_S`, `LLM_VISION_TIMEOUT_S`, `LLM_ENABLED`, `LLM_MAX_CHARS` |
| Extraction | `MAX_UPLOAD_MB` (20), `MAX_PAGES` (100), `MIN_TEXT_CHARS_PER_PAGE` (30), `OCR_DPI` (300) |
| OCR | `OCR_DET_MODEL`, `OCR_REC_MODEL_AR`, `OCR_REC_MODEL_EN`, `OCR_MIN_SCORE`, `OCR_MAX_SIDE`, `OCR_CPU_THREADS`, `OCR_PREPROCESS`, `OCR_ENGINES`, `OCR_PARALLEL_REC`, `OCR_REC_BATCH_SIZE` |
| Detection | `ID_PATTERNS_PATH` (keywords + country ID regexes), `RULES_WEIGHT`, `VISION_ENABLED`, `VISION_THRESHOLD`, `VISION_WEIGHT`, `VISION_MAX_SIDE`, `LLM_SKIP_MIN_PAGES`, `LLM_SKIP_MIN_CHARS`, `LLM_SKIP_CONFIDENCE` |
| API | `API_KEYS` or `CLIENTS_FILE` ([client registry](#clients-client-registry)), `UPLOAD_DIR`, `RESULT_TTL_S`, `MAX_QUEUED_JOBS`, `WORKER_THREADS`, `LLM_WORKER_THREADS`, `OCR_WARMUP`, `EXTRACT_MAX_PAGES`, `EXTRACT_MAX_CONCURRENT`, `RETRY_AFTER_S`, `MAX_WAIT_S`, `METRICS_ENABLED` |
| Job store | `JOB_STORE` (`memory`/`postgres`), `DATABASE_URL`, `DB_POOL_SIZE`, `DB_POLL_S`, `JOB_STALE_S`, `JOB_MAX_ATTEMPTS`, `RESULT_ENCRYPTION_KEY`, `AUDIT_RETENTION_DAYS` |
| Admin dashboard | `ADMIN_USERS_FILE`, `ADMIN_SESSION_SECRET`, `ADMIN_SESSION_HOURS`, `SAMPLE_INTERVAL_S`, `SAMPLE_PERSIST_S`, `SAMPLE_RETENTION_H`, `LOG_BUFFER_SIZE`, `LOG_RETENTION_DAYS`, `ALERT_*`, `CONFIG_POLL_S`, `EVAL_OUT_DIR` (see [Admin dashboard](#admin-dashboard)) |
| Handwriting (experimental) | `HANDWRITING_ENABLED`, `HANDWRITING_BACKEND`, `HANDWRITING_MODEL`, `HANDWRITING_BASE_URL`, `HANDWRITING_MAX_PIXELS`, `HANDWRITING_MAX_PAGES`, `HANDWRITING_BLANK_CHECK`, ... (see [Handwriting OCR](#handwriting-ocr-experimental)) |
| Webhooks | `WEBHOOK_SECRET`, `WEBHOOK_ALLOWED_HOSTS` (with `API_KEYS` only; per client in `CLIENTS_FILE`), `WEBHOOK_TIMEOUT_S`, `WEBHOOK_RETRY_WINDOW_S`, `WEBHOOK_BACKOFF_S`, `WEBHOOK_MAX_BACKOFF_S`, `WEBHOOK_RETRIES`, `WEBHOOK_THREADS` |

## Tests
```powershell
uv run pytest                          # everything; live LLM tests run if Ollama is up (~2.5 min on CPU)
$env:RUN_VISION_TESTS = '1'; uv run pytest tests/test_live_llm.py   # + the slow vision test
$env:RUN_HANDWRITING_TESTS = '1'; uv run pytest tests/test_live_handwriting.py   # the handwriting model (slow; needs it downloaded)
```
All test documents are synthetic, generated by `tests/synth.py`: fake names and numbers, the fictional ICAO
state "UTO", and invoices and letters as non-ID samples. No real documents are in the repository.

## Project layout
```
app/            config.py, cli.py, extractors/, ocr/, normalize/, detection/, api/ (routes/, migrations/),
                admin/ (dashboard: monitoring, alerts, configuration; static/ = its pages),
                handwriting/ (experimental handwriting OCR: backends, guards, regions)
config/         id_patterns.yaml (per-country ID regex, Phase 3), clients.example.yaml (client registry template),
                admins.example.yaml (admin dashboard users)
docs/           INTEGRATION*.md (for client teams), openapi.json (the API contract), CHANGELOG.md
scripts/        check_env.py (environment smoke test), bench_ocr.py (OCR settings benchmark), export_openapi.py,
                client_key.py (API keys / hashes / webhook secrets for the client registry),
                admin_user.py (admin password hashes, session secret)
tests/          pytest; fixtures/synthetic/ holds generated fake documents only
eval/           evaluation over real documents kept OUTSIDE the repo (Phase 5; run_handwriting_eval.py: handwriting CER/WER)
```
