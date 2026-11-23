"""HTTP middleware: request ids, early rejection of unauthenticated or oversized uploads, response metrics, and the
`request.error` audit event of every /v1 request answered with an error."""

import logging
import re
import uuid

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import Headers

from app import logs
from app.api import audit
from app.api.clients import ClientRegistry
from app.api.deps import max_request_bytes
from app.api.errors import error_response
from app.api.metrics import NO_CLIENT, Metrics
from app.config import Settings

log = logging.getLogger("app.api")
REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")  # a client-sent X-Request-Id is used only if it matches


class LogContext:
    """Pure ASGI, outermost: picks the request id (a valid client-sent X-Request-Id, or a new one) and the client, and
    keeps them in the log context for the whole request, including sending the response (uvicorn's access log line)."""

    def __init__(self, app, clients: ClientRegistry):
        self.app, self.clients = app, clients

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        rid = headers.get("x-request-id", "")
        rid = rid if REQUEST_ID.fullmatch(rid) else uuid.uuid4().hex
        client = self.clients.lookup(headers.get("x-api-key")) if scope["path"].startswith("/v1/") else None
        state = scope.setdefault("state", {})  # = request.state further in
        state["request_id"], state["key_client"] = rid, client
        with logs.context(request_id=rid, client_id=client.id if client is not None and client.enabled else None):
            await self.app(scope, receive, send)


def install(app: FastAPI, settings: Settings, clients: ClientRegistry, metrics: Metrics) -> None:
    max_body = max_request_bytes(settings)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Echo X-Request-Id; reject unauthenticated or oversized uploads before the body is read."""
        rid = request.state.request_id  # chosen by LogContext
        return await _handle(request, call_next, rid, request.state.key_client)

    app.add_middleware(LogContext, clients=clients)  # added last: runs first

    async def _handle(request: Request, call_next, rid: str, client):
        response, code = None, None
        api = request.url.path.startswith("/v1/")
        if api:
            key = request.headers.get("x-api-key")
            if client is None or not client.enabled:
                reason = "disabled" if client else ("unknown_key" if key else "no_key")
                log.warning("auth rejected reason=%s client=%s path=%s req=%s", reason,
                            client.id if client else "-", request.url.path, rid)  # never the key itself
                code = "unauthorized"
                response = error_response(401, code, "invalid or missing X-Api-Key", rid)
            else:
                length = request.headers.get("content-length")
                if length and length.isdigit() and int(length) > max_body:
                    code = "too_large"
                    response = error_response(413, code, "file too large", rid)
        if response is None:
            response = await call_next(request)
            code = getattr(request.state, "error_code", None)
        response.headers["X-Request-Id"] = rid
        route = request.scope.get("route")
        label = client.id if client is not None and client.enabled else NO_CLIENT
        metrics.http.labels(getattr(route, "path", "unmatched"), str(response.status_code), label).inc()
        if api and response.status_code >= 400:
            event = audit.request_error(request, response.status_code, code, client.id if client else None)
            await run_in_threadpool(audit.record_safely, request.app.state.store, event, metrics)
        return response
