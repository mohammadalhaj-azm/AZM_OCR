# Integration Guide: OCR + ID-Detection Service

This guide is for **teams whose system needs to connect to the OCR + ID-detection service**.
It covers what you get from us, what we need from you, how to connect step by step, and the full API contract.

> Values in `<ANGLE_BRACKETS>` are placeholders. The service operator fills them in before handing over this guide.

---

## 1. What the service does

You upload a document (**PDF, DOCX, JPG, PNG, WEBP or TIFF**). The service returns:

- **The extracted text.** Arabic first, English second, and mixed documents. It uses the PDF text layer when the page has one and OCR for scanned pages and images.
- **Optionally, ID detection.** It reports whether the document is an identity document and which kind (national ID, passport, driver's license, residence permit…). When a passport or ID card has a machine-readable zone (MRZ), it also returns the parsed MRZ fields.

All processing runs on our own servers. No document data is sent to any external or cloud service.

### Which endpoint should I use?

| You need | Use | Style | Typical time |
|---|---|---|---|
| **Text only** | `POST /v1/extract` | Synchronous: the result comes back in the same response | ~2–3 s per scanned page, instant for text PDFs |
| **Text + "is this an ID?"** | `POST /v1/documents` | Asynchronous: you get a `jobId`, then fetch the result | ~30–90 s per document (AI model on CPU) |
| Text only, but the document has **more than 20 pages** | `POST /v1/documents` | Asynchronous | depends on page count |
| **Handwritten or mixed** documents, corrected text, review items, **answers to your questions** (experimental) | `POST /v1/extract-and-answer` | Asynchronous | not measured yet: a minute or more per handwritten page on CPU |

---

## 2. What we (the operator) provide you

You will receive the following from us. Keep the secrets in a secret store, not in source code.

| # | Item | Value | Notes |
|---|---|---|---|
| 1 | **Base URL** | `<BASE_URL>` (e.g. `https://ocr.internal.example:8000`) | Reachable only from our internal network / VPN / allow-listed IPs |
| 2 | **API key** | `<API_KEY>` (sent separately over a secure channel) | Your system is registered as one client, with its own key. Send it in the `X-Api-Key` header |
| 3 | **Webhook secret** | `<WEBHOOK_SECRET>` (sent separately) | Only if you use webhooks. Yours alone: used to verify our callbacks to you |
| 4 | **API spec (OpenAPI)** | `<BASE_URL>/docs` (interactive), `<BASE_URL>/openapi.json`, and the file `docs/openapi.json` | Every response field is typed: generate your client from it. Read [§14 Versioning](#14-api-versioning-and-compatibility) and the [changelog](CHANGELOG.md) |
| 5 | **Limits in effect** | see [§10 Limits](#10-limits) | Tell us if you need different values |
| 6 | **Support contact** | `<SUPPORT_CONTACT>` | Quote the `X-Request-Id` when you report a problem |

You may get separate keys for **test** and **production**. Results are private to your system: a job you created is invisible (`404`) to every other client.

**Key rotation without downtime:** ask us for a new key. For a while both keys work, and both see the same jobs (jobs belong to your system, not to a key). Switch to the new key, then tell us to remove the old one. The same applies if a key leaks: tell us at once and we disable it.

Your access may be limited to the endpoints you need (`403 forbidden` for the others), and to your own share of the queue (`503 queue_full`) and of concurrent `/v1/extract` requests (`503 busy`), so other systems can't slow you down.

---

## 3. What we need from you (the client team)

Please send us the following before we enable your access:

| # | Item | Why we need it |
|---|---|---|
| 1 | **Name of your system and a technical contact** | So we can issue a dedicated API key and reach you |
| 2 | **Source IP addresses / network** your calls come from | To open the firewall / VPN for you |
| 3 | **Which mode** you will use: `/v1/extract`, `/v1/documents`, or both | To size the service |
| 4 | **Expected volume**: documents per day, peak documents per minute, typical pages per document | To set queue size and worker count |
| 5 | **Webhook callback URL(s)**, if you want callbacks instead of polling (e.g. `https://yourapp.internal/hooks/ocr`) | We must allow-list the **host name** on our side, or callbacks are rejected with `400 callback_not_allowed` |
| 6 | **Whether you need line positions** (`includeBoxes`) | For highlighting text on the page image |
| 7 | **Which data you need back** (the text, the ID decision, MRZ fields), **which ID documents and countries** you handle, and **how long** we should keep results | We set up a [processing profile](#processing-profiles), so you receive (and we keep) only what you need |

---

## 4. Step-by-step connection

### Step 1: Check that you can reach the service (no key needed)
```bash
curl -s <BASE_URL>/health/ready
# 200 {"status":"ready", ...}
```
If this times out, the network or firewall is not open yet. Contact us with your source IP.

### Step 2: Make an authenticated test call (text only)
```bash
curl -s --max-time 120 -H "X-Api-Key: <API_KEY>" -F "file=@sample.pdf" <BASE_URL>/v1/extract
```
A `401 unauthorized` means the key is wrong or missing.

### Step 3: Make an asynchronous call (text + ID detection)
```bash
# 1. upload -> jobId
curl -s -H "X-Api-Key: <API_KEY>" -H "Idempotency-Key: order-123-doc-1" \
     -F "file=@id_card.jpg" -F "externalId=order-123" \
     <BASE_URL>/v1/documents
# -> {"jobId":"bbb75af4c0d2..."}

# 2. wait for the result (holds up to 30 s, returns as soon as the job is finished)
curl -s -H "X-Api-Key: <API_KEY>" "<BASE_URL>/v1/jobs/bbb75af4c0d2...?wait=30"

# 3. delete once you have stored the result
curl -s -X DELETE -H "X-Api-Key: <API_KEY>" <BASE_URL>/v1/jobs/bbb75af4c0d2...
```

### Step 4 (optional): Use webhooks instead of waiting
Add `-F "callbackUrl=https://yourapp.internal/hooks/ocr"` to the upload, then implement the receiver as described in [§8 Webhooks](#8-webhooks).

### Step 5: Make it robust
Add an `Idempotency-Key`, retries on `503`, `DELETE` after use, and `X-Request-Id` logging. See [§9](#9-errors-and-retry-policy) and the [full code samples](#12-code-samples).

### Step 6: Go live
Work through the [go-live checklist](#13-go-live-checklist).

---

## 5. Authentication and headers

| Header | Direction | Required | Description |
|---|---|---|---|
| `X-Api-Key` | request | **yes**, on every `/v1/*` call | Your API key. Missing or wrong → `401 unauthorized` |
| `Idempotency-Key` | request | recommended on `POST /v1/documents` | 1–128 characters of `A-Z a-z 0-9 . _ : -`. Repeating an upload with the same key and the **same file** returns the **same `jobId`** (response header `Idempotent-Replayed: true`). The same key with a **different file** → `409 idempotency_conflict`. Keys expire together with their job (1 h after the job finishes) |
| `X-Request-Id` | request + response | optional | Your correlation id (1–64 characters of `A-Z a-z 0-9 . _ : -`). It is echoed back and written to our logs. If you don't send one, we generate one. Always log it on your side |

The `/health*` endpoints need no key.

---

## 6. Endpoint reference

### `POST /v1/extract`: text only, synchronous

`multipart/form-data` fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | PDF, DOCX, JPG, PNG, WEBP or TIFF, at most 20 MB, at most 20 pages |
| `includeBoxes` | `true`/`false` | no (default: your profile's, normally `false`) | Adds line positions to each OCR'd page (see [§7](#7-result-format)) |
| `profile` | string | no (default: your first profile) | One of the [processing profiles](#processing-profiles) we set up for your system |

Responses: `200` with the [result](#7-result-format), without `idDetection`. Errors: `400`, `401`, `413` (too large, or more than 20 pages: use `/v1/documents`), `415`, `503 busy`.
Set your HTTP client timeout to about **120 s**.

### `POST /v1/documents`: text + ID detection, asynchronous

`multipart/form-data` fields, plus the optional `Idempotency-Key` header:

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | yes | PDF, DOCX, JPG, PNG, WEBP or TIFF, at most 20 MB, at most 100 pages |
| `callbackUrl` | string | no | Webhook URL. Its host must be allow-listed by us |
| `externalId` | string | no | Your reference (1–128 characters). Echoed in the job and in the webhook |
| `metadata` | string (a JSON **object**) | no | At most 2 KB, e.g. `{"customerRef":"C-77","channel":"web"}`. Echoed in the job and in the webhook |
| `includeBoxes` | `true`/`false` | no | Same as in `/v1/extract` |
| `profile` | string | no | Same as in `/v1/extract` |

Responses: `202 {"jobId": "<32 hex chars>"}`. Errors: `400`, `401`, `409`, `413`, `415`, `422`, `503 queue_full`.

### `POST /v1/extract-and-answer` (experimental): printed + handwritten text, correction, answers, asynchronous

Reads every page with the right engine (printed text: OCR; handwritten or mixed pages: the handwriting model). The
local AI model then corrects and formats the text, lists what a person should check, and answers your questions
from the document only. **Experimental:** it may change while it is evaluated (see [§14](#14-api-versioning-and-compatibility)).

Same fields as `/v1/documents` (`file`, `callbackUrl`, `externalId`, `metadata`, `includeBoxes`, `profile`,
`Idempotency-Key`), plus:

| Field | Type | Required | Description |
|---|---|---|---|
| `questions` | string, repeatable | no | A question about the document; send the field once per question (at most 10, 500 characters each). Without questions, the job only extracts and corrects |
| `mode` | `auto`/`printed`/`handwritten` | no (default `auto`) | `auto` routes each page. `printed` / `handwritten` read every page with that engine (`handwritten` also for PDF pages that have a text layer) |
| `correct` | `true`/`false` | no (default `true`) | `false`: no AI correction (extraction, routing and answers only) |

Responses: `202 {"jobId": ...}`, then the result from `GET /v1/jobs/{jobId}` (below). Errors as for `/v1/documents`,
plus `403` for a profile that returns no text, `409` when an `Idempotency-Key` is reused with other questions or
options, and `503 answer_unavailable` when we have turned it off. Needs the `documents` scope. At most 20 pages.
It is slow: about a minute or more per handwritten page on CPU, plus the AI calls.

The result has the usual fields; `pages[].text` and `fullText` are the text **as read**. In addition:
- `pages[].engine`, `pages[].route` (`kind`: printed / handwritten / mixed / no_text / unknown; `decidedBy`;
  `uncertain`), `pages[].readings` (each engine's reading: mixed and uncertain pages have two),
  `pages[].correctedText` and `pages[].changes` (every word the correction changed);
- `correctedText`: all pages corrected, or `null` when the text was not corrected (`correction.status`:
  `done` / `partial` / `skipped` / `unavailable`);
- `review[]`: what a person should check (`page`, `quote`, `suggestion`, `reason`, `category`, `source`). **Show
  these to a reviewer**, especially `protected_value`, `inconsistent` and `routing` items;
- `answers[]` (only with questions): `status` `answered` / `not_found` / `unavailable`, `answer`, `citations[]`
  (`page`, `quote`, `verified`: the quote was found on that page) and `needsReview`. Treat an answer with
  `needsReview: true` as a hint, not a fact.

Numbers, dates and IDs are never changed by the correction. Handwriting has no confidence score (`confidence:
null`), and none is made up.

### `GET /v1/jobs/{jobId}[?wait=N]`: status and result

- `wait=N` (seconds, at most 30) holds the request until the job is `done` or `failed`, or until N seconds pass. **Use `wait=30` in a loop** instead of fast polling. Set your client timeout to about **35 s**.
- `status` is one of `queued`, `processing`, `done` or `failed`.

```json
{"jobId": "bbb75af4...", "status": "processing", "externalId": "order-123"}
```
When `done`, the response also contains the [result fields](#7-result-format), including `idDetection`.
When `failed`, it contains `"error": {"code": "...", "message": "..."}`.
If you gave a `callbackUrl`, it also contains `"webhook": {"state": "pending"|"delivered"|"failed", "attempts": N}`.

Errors: `401`, `404 not_found` (unknown or expired job, or a job of another client).

### `DELETE /v1/jobs/{jobId}`: delete a job and its result now
`204 No Content`. Please call this **as soon as you have stored the result**. Otherwise the result is deleted automatically 1 hour after the job finishes.

### Health endpoints (no key)
| Endpoint | Use |
|---|---|
| `GET /health/live` | The process is up |
| `GET /health/ready` | Ready to accept work (OCR models loaded, worker running). Use this for your monitoring |
| `GET /health/handwriting` | Status of the experimental handwriting model (see the README); not part of the other checks |
| `GET /health` | Full status, including the AI model. `503 degraded` means `/v1/documents` ID detection may be limited, while `/v1/extract` still works |

---

## 7. Result format

```json
{
  "jobId": "bbb75af4...",
  "status": "done",
  "externalId": "order-123",
  "metadata": {"customerRef": "C-77"},
  "file": {"name": "id_card.jpg", "mimeType": "image/jpeg", "pages": 1},
  "pages": [
    {"page": 1, "method": "ocr", "text": "...", "confidence": 0.93}
  ],
  "fullText": "...all pages joined...",
  "normalizedText": "...Arabic-normalized text, best for searching/matching...",
  "languages": ["ar", "en"],
  "idDetection": {
    "isId": true,
    "docType": "national_id",
    "confidence": 0.94,
    "decidedBy": ["rules", "llm"],
    "reasons": ["..."],
    "mrz": { "...": "only when an MRZ was found, see below" }
  },
  "timingsMs": {"extract": 3120, "ocr": 3050, "llm": 41000, "total": 44500},
  "processing": {"version": "0.3.0", "ocr": {"det": "PP-OCRv5_mobile_det", "...": "..."}, "rules": "3f9a1c2b7d4e",
                 "llm": "qwen3:8b"}
}
```

| Field | Meaning |
|---|---|
| `pages[].method` | `text_layer` (text taken from the PDF directly) or `ocr` (text recognized from the image) |
| `pages[].confidence` | OCR confidence, 0–1 (`null` for `text_layer`) |
| `fullText` | The original extracted text |
| `normalizedText` | Arabic-normalized text (unified letter forms, no diacritics). Use it for search and matching |
| `languages` | `ar` / `en`, ordered by amount of text |
| `idDetection.isId` | `true` if this is an identity document |
| `idDetection.docType` | One of `national_id`, `passport`, `driver_license`, `residence_permit`, `other_id`, `not_id` |
| `idDetection.confidence` | 0–1. Pick a threshold for your use case (e.g. send results below 0.8 to manual review) |
| `idDetection.decidedBy` | Which stages decided: `rules`, `llm`, `vision` |
| `idDetection.reasons` | Human-readable reasons, for audit and review screens |
| `processing` | What produced the result: service `version`, the `profile`, the OCR models, the ID rules version and the AI models that were used (parts that weren't used are left out), and `withheld` if your profile removed data. Store it with the result if you need to explain a decision later |

`idDetection` is present only on `/v1/documents` results.

### Line positions (`includeBoxes=true`)
Each OCR page then also has its image size and the lines in reading order:
```json
{"page": 2, "method": "ocr", "text": "...", "confidence": 0.93, "width": 2480, "height": 3508,
 "lines": [{"text": "فاتورة ضريبية", "confidence": 0.97, "bbox": [0.512, 0.061, 0.884, 0.083]}]}
```
`bbox` is `[x0, y0, x1, y1]` as **fractions (0–1) of the page**, measured from the top-left corner. Multiply by your image width and height to draw it. `text_layer` pages have no `lines`.

### MRZ fields (passports / ID cards)
```json
"mrz": {
  "format": "TD3", "documentCode": "P", "issuingCountry": "UTO", "surname": "TESTER", "givenNames": "SAMPLE",
  "documentNumber": "AB1234567", "nationality": "UTO", "birthDate": "1990-02-01", "sex": "M",
  "expiryDate": "2030-01-01", "optionalData": null, "expired": false, "valid": true, "namesSource": "mrz+text",
  "checks": {"documentNumber": true, "birthDate": true, "expiryDate": true, "personalNumber": true, "composite": true}
}
```
- **Use a field only if its `checks` entry is `true`.** `valid: true` means all checks passed. A field whose check failed is still returned, because it often holds a single OCR error.
- **Names have no check digit.** Treat `surname` and `givenNames` as unverified and confirm them against your own data.
- Dates are ISO (`YYYY-MM-DD`). `expired` compares `expiryDate` with today's date.

### Processing profiles
We can set up one or more **profiles** for your system, to match what it needs. A profile can:
- **change the processing:** your own ID keywords and ID-number formats, ID detection off (text only), no AI models (rules only), a page limit, a higher scan resolution;
- **withhold data you don't need:** with `return_text: false` results carry no document text (`pages[].text`, `fullText`, `normalizedText` are empty, no `lines`, and the AI models' free-text reasons are removed); with `return_mrz: false` they carry no MRZ fields. Withheld data is **not stored** by us either. `processing.withheld` lists what was left out;
- **keep results longer or shorter** than the standard 1 hour.

Your first profile applies by default. Send `profile=<name>` to use another one of yours. `processing.profile` names the profile that produced a result.

---

## 8. Webhooks

If you send a `callbackUrl`, we `POST` to it when the job finishes:

```http
POST /hooks/ocr HTTP/1.1
Content-Type: application/json
X-Webhook-Id: 5f0c...            (same as deliveryId)
X-Webhook-Timestamp: 1763560000  (unix seconds)
X-Webhook-Signature: sha256=9a1b...

{"event": "job.done", "deliveryId": "5f0c...", "jobId": "bbb75af4...", "status": "done",
 "externalId": "order-123", "metadata": {"customerRef": "C-77"}}
```
On failure, `event` is `job.failed` and the body also contains `"error": {"code": "..."}`.

**The webhook carries no document data.** After you receive it, fetch the result with `GET /v1/jobs/{jobId}` using your API key.

Your receiver must:
1. **Verify the signature:** `HMAC_SHA256(<WEBHOOK_SECRET>, "<X-Webhook-Timestamp>." + raw_body)`. Compare it in constant time and reject timestamps older than 5 minutes.
2. **Answer `2xx` quickly** (within 10 s). Do the real work asynchronously. Any non-2xx answer or timeout is retried with growing delays (2 s, doubling up to 1 h between attempts) for up to 24 hours, also across restarts of our service. A short outage or maintenance window on your side loses nothing.
3. **De-duplicate on `deliveryId`.** Retries reuse the same id.
4. The result is kept until the webhook was delivered (or given up) **plus 1 hour**, so you always have time to fetch it. If the webhook never arrives, you can still fetch the job by `jobId`.

**Python (verify):**
```python
import hashlib, hmac, time

def verify(secret: bytes, ts: str, raw_body: bytes, sig: str) -> bool:
    good = "sha256=" + hmac.new(secret, ts.encode() + b"." + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(good, sig) and abs(time.time() - int(ts)) < 300
```

**Node.js (verify, Express with the raw body):**
```js
const crypto = require("crypto");
function verify(secret, ts, rawBody, sig) {
  const good = "sha256=" + crypto.createHmac("sha256", secret).update(`${ts}.`).update(rawBody).digest("hex");
  const a = Buffer.from(good), b = Buffer.from(sig || "");
  return a.length === b.length && crypto.timingSafeEqual(a, b) && Math.abs(Date.now() / 1000 - Number(ts)) < 300;
}
// app.post("/hooks/ocr", express.raw({ type: "application/json" }), (req, res) => { ... verify(SECRET, req.get("X-Webhook-Timestamp"), req.body, req.get("X-Webhook-Signature")) ... })
```
Sign the **raw bytes** you received. Don't re-serialize parsed JSON, or the signature won't match.

---

## 9. Errors and retry policy

Every error has the same shape:
```json
{"error": {"code": "unsupported_type", "message": "...", "requestId": "3f2a..."}}
```
**Write your code against `code`.** `message` is for humans and may change.

| HTTP | `code` | What to do |
|---|---|---|
| 400 | `empty`, `corrupt` | The file is empty, damaged or encrypted (password-protected PDF). Don't retry |
| 400 | `callback_not_allowed` | The callback host isn't allow-listed. Ask us to add it |
| 401 | `unauthorized` | Wrong, missing or disabled `X-Api-Key` |
| 403 | `forbidden` | Your key may not use this endpoint or the requested `profile` (or the profile returns no text and you called `/v1/extract`). Ask us if you need it |
| 404 | `not_found` | Unknown or expired job, or a job of another client |
| 409 | `idempotency_conflict` | The `Idempotency-Key` was reused for a different file. Use a new key |
| 413 | `too_large`, `too_many_pages` | Over 20 MB or over the page limit. For `/v1/extract`, send it to `/v1/documents` instead |
| 415 | `unsupported_type` | Not PDF, DOCX, JPG, PNG, WEBP or TIFF |
| 422 | `invalid_request` | A field is missing or malformed (e.g. `metadata` isn't a JSON object) |
| 500 | `internal_error` | Processing failed (no details, by design). Report it with the `requestId` |
| 503 | `queue_full`, `busy` | Temporary overload. **Retry after the `Retry-After` header** (seconds) |

A **failed job** (`status: "failed"`) uses the same codes in `error.code`, plus `worker_lost` (the file repeatedly crashed processing).

**Retry rules:**
- Retry **`503`** (after `Retry-After`) and **network errors / timeouts**, using exponential backoff with a cap (e.g. 5 attempts).
- **Don't retry other `4xx`.** Fix the request instead.
- Always send an **`Idempotency-Key`** on `POST /v1/documents` (e.g. `<your-record-id>-<document-no>`), so a retry after a timeout never creates a duplicate job.

---

## 10. Limits

These are the defaults. We may change them for your integration, so ask us if you need different ones.

| Limit | Default |
|---|---|
| Max file size | 20 MB |
| Max pages, `/v1/documents` | 100 |
| Max pages, `/v1/extract` | 20 |
| Concurrent `/v1/extract` requests (whole service) | 2 (more → `503 busy`) |
| Queued jobs (whole service) | 100 (more → `503 queue_full`) |
| Longest `wait=` on `GET /v1/jobs` | 30 s |
| Result retention after the job finishes | 1 hour (then `404`) |
| Webhook timeout / retries | 10 s per attempt / retried for up to 24 h |
| `externalId` / `metadata` | 128 characters / 2 KB JSON object |

---

## 11. Data-protection rules for clients

- **Don't put personal data in `externalId` or `metadata`.** They are sent in webhooks. Use opaque references (`order-123`), not names or ID numbers.
- **Call `DELETE /v1/jobs/{jobId}`** as soon as you have stored the result.
- MRZ fields and extracted text are **personal data**. Store and log them on your side according to your data-protection policy, and don't write them to plain application logs.
- Keep the API key and webhook secret in a secret store, and rotate them if they leak (tell us).
- Connect over HTTPS or our internal network only.

---

## 12. Code samples

### Python (httpx): full async flow with idempotency, retries, long-poll and delete
```python
import time
import httpx

BASE_URL = "<BASE_URL>"
API_KEY = "<API_KEY>"


def call(client: httpx.Client, method: str, url: str, attempts: int = 5, **kw) -> httpx.Response:
    """Retry on 503 (honouring Retry-After) and on network errors; never on other 4xx."""
    for i in range(attempts):
        try:
            r = client.request(method, url, **kw)
        except httpx.TransportError:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)
            continue
        if r.status_code == 503 and i < attempts - 1:
            time.sleep(int(r.headers.get("Retry-After", 5)))
            continue
        return r
    return r


def analyze(path: str, record_id: str) -> dict:
    with httpx.Client(base_url=BASE_URL, headers={"X-Api-Key": API_KEY}, timeout=60) as c:
        with open(path, "rb") as f:
            data = f.read()
        r = call(c, "POST", "/v1/documents",
                 files={"file": (path.rsplit("/", 1)[-1], data)},
                 data={"externalId": record_id},
                 headers={"Idempotency-Key": f"{record_id}-1", "X-Request-Id": f"{record_id}-1"})
        if r.status_code != 202:
            raise RuntimeError(r.json()["error"])
        job_id = r.json()["jobId"]

        while True:
            job = call(c, "GET", f"/v1/jobs/{job_id}", params={"wait": 30}, timeout=35).json()
            if job["status"] in ("done", "failed"):
                break

        call(c, "DELETE", f"/v1/jobs/{job_id}")
        if job["status"] == "failed":
            raise RuntimeError(job["error"])
        return job


job = analyze("scan.pdf", "order-123")
det = job["idDetection"]
print(det["isId"], det["docType"], det["confidence"])
```

### Python: text only
```python
import httpx

with open("letter.pdf", "rb") as f:
    r = httpx.post("<BASE_URL>/v1/extract", headers={"X-Api-Key": "<API_KEY>"},
                   files={"file": ("letter.pdf", f)}, timeout=120)
r.raise_for_status()
print(r.json()["fullText"])
```

### Node.js / TypeScript (Node 18+, built-in fetch)
```ts
import { readFile } from "node:fs/promises";

const BASE_URL = "<BASE_URL>";
const API_KEY = "<API_KEY>";
const headers = { "X-Api-Key": API_KEY };

async function analyze(path: string, recordId: string) {
  const form = new FormData();
  form.append("file", new Blob([await readFile(path)]), path.split("/").pop()!);
  form.append("externalId", recordId);

  const up = await fetch(`${BASE_URL}/v1/documents`, {
    method: "POST", body: form,
    headers: { ...headers, "Idempotency-Key": `${recordId}-1` },
  });
  if (up.status !== 202) throw new Error(JSON.stringify(await up.json()));
  const { jobId } = await up.json();

  let job: any;
  do {
    job = await (await fetch(`${BASE_URL}/v1/jobs/${jobId}?wait=30`, { headers })).json();
  } while (job.status !== "done" && job.status !== "failed");

  await fetch(`${BASE_URL}/v1/jobs/${jobId}`, { method: "DELETE", headers });
  if (job.status === "failed") throw new Error(job.error.code);
  return job; // job.idDetection.isId, job.idDetection.docType, job.fullText ...
}
```
Add the same 503 / `Retry-After` handling as in the Python sample for production.

### C# (.NET 6+)
> For a complete ASP.NET Core + React/Angular setup (typed client, upload API, webhook receiver, background poller
> and frontend code), see **[INTEGRATION-ASPNET.md](INTEGRATION-ASPNET.md)**.

```csharp
using var http = new HttpClient { BaseAddress = new Uri("<BASE_URL>"), Timeout = TimeSpan.FromSeconds(120) };
http.DefaultRequestHeaders.Add("X-Api-Key", "<API_KEY>");

using var form = new MultipartFormDataContent();
form.Add(new ByteArrayContent(await File.ReadAllBytesAsync("letter.pdf")), "file", "letter.pdf");
var resp = await http.PostAsync("/v1/extract", form);
var json = await resp.Content.ReadAsStringAsync();   // { "fullText": "...", "pages": [...], ... }
```

---

## 13. Go-live checklist

**Client team:**
- [ ] A production API key is received and stored in a secret store (not in code or config files in git)
- [ ] Your calls reach `<BASE_URL>/health/ready` from the production network
- [ ] Every `POST /v1/documents` sends an `Idempotency-Key`
- [ ] Retries on `503` (with `Retry-After`) and network errors only; no retries on other `4xx`
- [ ] Timeouts: about 120 s for `/v1/extract`, about 35 s for `GET /v1/jobs?wait=30`
- [ ] `DELETE /v1/jobs/{id}` is called after you store the result
- [ ] `X-Request-Id` is logged on your side for every call
- [ ] Webhooks (if used): signature verified, timestamp checked, de-duplicated on `deliveryId`, fast `2xx`
- [ ] No personal data in `externalId` / `metadata`; no document text or MRZ data in plain logs
- [ ] Business rule for low `idDetection.confidence` decided (e.g. manual review below a threshold)

**Operator (us):**
- [ ] The client is added to `CLIENTS_FILE` with its own id, a new key's hash (`scripts/client_key.py new`), only the scopes it needs, and limits sized for its volume
- [ ] The firewall / VPN is open for the client's source IPs
- [ ] If it uses webhooks: its callback host is in its `webhook.allowed_hosts`, and its own secret (`scripts/client_key.py secret`) is in the environment variable named by `webhook.secret_env` and shared with the client
- [ ] TLS is in front of the service (reverse proxy) if it is reached outside a trusted network
- [ ] Limits are sized for the client's expected volume (`MAX_QUEUED_JOBS`, `WORKER_THREADS`, `EXTRACT_MAX_CONCURRENT`)
- [ ] The PostgreSQL job store and `RESULT_ENCRYPTION_KEY` are configured for production
- [ ] The service runs with `ENVIRONMENT=production` (the Docker image default) and starts, i.e. its safety checks pass

---

## 14. API versioning and compatibility

Many systems depend on this API, so it changes by fixed rules.

**What the version numbers mean**
- `/v1` in the path is the **contract**. It only changes for breaking changes (see below).
- The **service version** (`info.version` in the spec, `version` in `/health`, `processing.version` in every result) is the release that served you. It changes with every release; quote it when you report a problem.

**Experimental endpoints** (tagged `experimental` in the spec, today `POST /v1/handwriting` and `POST /v1/extract-and-answer`, with the result fields only the latter produces) are excluded: they may change or disappear while they are evaluated. Don't build production flows on them yet.

**Within `/v1`, changes are only additive:**
- new endpoints, new optional request fields and headers;
- new fields in responses and webhooks;
- new error codes, and new values in lists of values (e.g. a new `docType`).

**So your client must:**
- **Ignore response fields it doesn't know.** Most generated clients do this by default; don't enable strict "fail on unknown property" settings.
- **Tolerate unknown values:** map value lists such as `docType`, `status` or `method` as strings (or configure your enum deserializer to accept unknown values), and send an unknown `docType` to manual review.
- **Handle unknown error codes by the HTTP status:** don't retry other 4xx; retry 503 after `Retry-After` (see [§9](#9-errors-and-retry-policy)).

**Breaking changes** come only with a new path, `/v2`:
- Breaking means removing or renaming a field, changing its type or meaning, a new required request field, or removing an endpoint or a value.
- `/v1` keeps working **for at least 6 months** after `/v2` is available. We announce it to your technical contact.
- During that time, `/v1` responses carry the `Deprecation` and `Sunset` headers (RFC 9745, RFC 8594) with the dates.

**The contract file:** `docs/openapi.json` (the same document as `<BASE_URL>/openapi.json`).
- Our build fails whenever the API differs from it, so every change to the API is a reviewed change of this file.
- Every change is listed in the [changelog](CHANGELOG.md).
- To see exactly what changed, compare two versions of the file.
