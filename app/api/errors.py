"""One error shape for every response: {"error": {"code", "message", "requestId"}}.

`code` is stable and meant for programs; `message` is for humans and may change. Messages never contain document
content, and validation errors never echo request values.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("app.api")

# Every code the API can return (documented in the README).
CODES = {
    "unauthorized": "missing or invalid X-Api-Key (also a disabled client)",
    "forbidden": "this API key may not use the endpoint (the client's scopes in CLIENTS_FILE)",
    "not_found": "unknown job, or a job of another API key",
    "empty": "the file is empty",
    "corrupt": "the file could not be opened or decoded (also encrypted PDFs)",
    "too_large": "the file is over MAX_UPLOAD_MB (or a DOCX expands too much)",
    "too_many_pages": "more pages than allowed (MAX_PAGES, or EXTRACT_MAX_PAGES for /v1/extract)",
    "unsupported_type": "not PDF, DOCX, JPG, PNG, WEBP or TIFF",
    "callback_not_allowed": "callbackUrl is not allowed: callbacks are off for this API key, or its host isn't listed",
    "invalid_request": "missing or malformed request fields",
    "idempotency_conflict": "the Idempotency-Key was already used by this API key for a different file",
    "worker_lost": "(failed jobs only) the worker processing the job died JOB_MAX_ATTEMPTS times",
    "queue_full": "MAX_QUEUED_JOBS reached; retry after Retry-After seconds",
    "busy": "EXTRACT_MAX_CONCURRENT reached (/v1/handwriting: the model is busy); retry after Retry-After seconds",
    "handwriting_unavailable": "(experimental /v1/handwriting) the handwriting model is disabled, not installed, not "
                               "downloaded or its server is unreachable",
    "unknown_model": "(experimental /v1/handwriting) the requested handwriting model is not configured on this server",
    "internal_error": "processing failed (no details are returned)",
}
_DEFAULT_CODE = {400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
                 405: "invalid_request", 409: "idempotency_conflict", 413: "too_large", 415: "unsupported_type",
                 422: "invalid_request", 503: "busy"}


class ErrorDetail(BaseModel):
    code: str
    message: str
    requestId: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ApiError(HTTPException):
    def __init__(self, status_code: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(status_code, message, headers)
        self.code = code


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(status: int, code: str, message: str, request_id: str | None,
                   headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "requestId": request_id}}
    headers = dict(headers or {})
    if request_id:
        headers["X-Request-Id"] = request_id
    return JSONResponse(body, status_code=status, headers=headers)


# For OpenAPI: lets client generators type the error responses.
ERROR_RESPONSES = {s: {"model": ErrorResponse} for s in (400, 401, 403, 404, 409, 413, 415, 422, 500, 503)}


def install(app: FastAPI) -> None:
    # Each handler notes its code on request.state.error_code: the middleware writes it to the audit trail.
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        code = getattr(exc, "code", None) or _DEFAULT_CODE.get(exc.status_code, "internal_error")
        message = exc.detail if isinstance(exc.detail, str) else code
        request.state.error_code = code
        return error_response(exc.status_code, code, message, request_id_of(request), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Field locations and error types only: FastAPI's default would echo the submitted values.
        fields = sorted({".".join(str(p) for p in e.get("loc", ())) + f" ({e.get('type', 'invalid')})"
                         for e in exc.errors()})
        request.state.error_code = "invalid_request"
        return error_response(422, "invalid_request", "invalid request: " + ", ".join(fields),
                              request_id_of(request))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.error("unhandled error error=%s req=%s", type(exc).__name__, request_id_of(request) or "-")
        # Unhandled errors bypass the HTTP middleware, so the audit event is written here.
        from fastapi.concurrency import run_in_threadpool

        from app.api.audit import record_safely, request_error

        if request.url.path.startswith("/v1/") and hasattr(request.app.state, "store"):
            await run_in_threadpool(record_safely, request.app.state.store,
                                    request_error(request, 500, "internal_error"), request.app.state.metrics)
        return error_response(500, "internal_error", "processing failed", request_id_of(request))
