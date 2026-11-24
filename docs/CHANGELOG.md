# Changelog

Changes that client systems can see. The API contract is `/v1`, and within it changes are only additive (see
[INTEGRATION.md §14](INTEGRATION.md#14-api-versioning-and-compatibility)). The full contract is
[`openapi.json`](openapi.json).

## Unreleased

### Added (experimental)
- **`POST /v1/handwriting`** reads handwritten Arabic and English. It is **experimental**: it may change or be
  removed while it is evaluated, and the additive-only rule of §14 does not apply to it yet. It is off unless the
  operator turns it on (`503 handwriting_unavailable` otherwise), needs the `extract` scope, and has its own result
  model (`HandwritingResult`: `pages[].confidence` is always `null`). `GET /health/handwriting` reports its status.
- New error code `handwriting_unavailable` (503), returned by this endpoint only.
- Nothing changes in `/v1/extract`, `/v1/documents`, `/v1/jobs` or their results.
- `/v1/handwriting` (still experimental): optional `model` and `segmenter` form fields (several models can be offered
  side by side; without the fields the default model answers as before), `model.id`, `model.adapter` and
  `model.quantization` in the result (when they apply), page warnings `no_lines` and `too_many_lines`, error code
  `unknown_model` (400). `GET /health/handwriting` lists the models under `models`.

## 0.4.0 (2026-11-21)

### Added
- Nothing changes for client systems: the `/v1` API and the webhooks are as in 0.3.0. This release adds an admin
  dashboard for the service's operators (README, "Admin dashboard"). Your system's id, request counts, errors and
  webhook outcomes are visible to them, never the content of your documents.

## 0.3.0 (2026-11-21)

### Added
- **`processing` in every result:** the service version, and the OCR models, ID-rules version and AI models that
  produced it. Store it with the result if you need to explain a decision later.
- **Typed response schemas** in the OpenAPI document: `file`, `pages`, `lines`, `idDetection`, `mrz`, `error`, `webhook`,
  `processing` and their value lists. The JSON itself is unchanged. Regenerate your client to get the types.
- **`403 forbidden`:** your key may not use that endpoint (each client system gets only the endpoints it needs).
- **Per-client limits:** `503 queue_full` and `503 busy` can also mean that your own share is used up. Retry after
  `Retry-After`, as before.
- **`version`** in `GET /health` and `GET /health/ready`.
- **Processing profiles:** the optional `profile` request field (on `/v1/documents` and `/v1/extract`) picks one of
  the profiles set up for your system. `processing.profile` names the profile used, and `processing.withheld` lists
  data a profile left out (`text`, `mrz`). A profile you may not use gets `403 forbidden`.
- `includeBoxes` may be left out: the default is then your profile's (normally `false`, as before).

### Changed
- **Webhooks are retried for up to 24 hours** (2 s, doubling up to 1 h between attempts), also across restarts of the
  service. Before: 3 attempts within seconds. The `deliveryId` stays the same for every attempt: keep de-duplicating
  on it.
- **With a webhook, the result is kept until 1 hour after the webhook was delivered** (or given up). Before: 1 hour after
  the job finished.
- **Webhook secrets and callback hosts can be per client system.** If you get your own secret, verify our callbacks
  with it.
- **Request bodies over the upload limit are rejected while they stream in** (`413 too_large`), also without a
  `Content-Length` header.
- The `message` of `callback_not_allowed` errors changed. The code did not.

## 0.2.0

The first versioned API: `POST /v1/documents` (async, with ID detection), `POST /v1/extract` (sync, text only),
`GET/DELETE /v1/jobs/{jobId}` with long-poll, signed webhooks, `Idempotency-Key`, `externalId` / `metadata`, and one
error envelope with stable codes.
