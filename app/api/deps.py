"""Shared FastAPI dependencies and helpers: authentication, scopes, job ownership, upload reading.

Everything per-app (settings, store, clients, metrics, services) lives on `request.app.state`, set by `create_app`.
"""

import hmac
from collections.abc import Callable

from fastapi import Depends, Request, UploadFile
from fastapi.security import APIKeyHeader

from app.api.clients import Client, ClientRegistry, Scope
from app.api.errors import ApiError
from app.api.store import Job, JobStore
from app.config import Settings
from app.extractors.filetype import FileRejected, validate_upload
from app.profiles import Profile

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)
CHUNK = 1024 * 1024
MULTIPART_OVERHEAD = 64 * 1024  # boundaries, part headers and the small form fields next to the file
REJECT_STATUS = {"too_large": 413, "too_many_pages": 413, "unsupported_type": 415}  # other FileRejected -> 400


def max_request_bytes(settings: Settings) -> int:
    """Largest request body the upload endpoints accept: the file plus multipart overhead."""
    return settings.max_upload_bytes + MULTIPART_OVERHEAD


def retry_after(settings: Settings) -> dict[str, str]:
    return {"Retry-After": str(max(1, settings.retry_after_s))}


def app_settings(request: Request) -> Settings:
    return request.app.state.settings


def app_store(request: Request) -> JobStore:
    return request.app.state.store


def app_clients(request: Request) -> ClientRegistry:
    return request.app.state.clients


def require_client(request: Request, key: str | None = Depends(api_key_header),
                   clients: ClientRegistry = Depends(app_clients)) -> Client:
    client = clients.authenticate(key)
    if client is None:  # unknown key, or a disabled client: the same answer
        raise ApiError(401, "unauthorized", "invalid or missing X-Api-Key")
    request.state.client = client  # for the audit trail of errors (middleware)
    return client


def require_scope(scope: Scope) -> Callable[..., Client]:
    """Dependency: the authenticated client, which must have `scope` (403 otherwise)."""
    def check(client: Client = Depends(require_client)) -> Client:
        if scope not in client.scopes:
            raise ApiError(403, "forbidden", f"this API key may not use the {scope} endpoint")
        return client
    return check


def use_profile(clients: ClientRegistry, client: Client, requested: str | None) -> Profile:
    """The processing profile of a request; 403 if the client may not use the requested one (unknown names get the
    same answer, so other clients' profiles stay private)."""
    profile = clients.profile_for(client, requested)
    if profile is None:
        raise ApiError(403, "forbidden", "this API key may not use the requested profile")  # never echo values
    return profile


def own_job(store: JobStore, job_id: str, owner: str) -> Job:
    job = store.get(job_id)
    if job is None or not hmac.compare_digest(job.owner, owner):
        raise ApiError(404, "not_found", "job not found")  # also for other clients' jobs: don't reveal existence
    return job


async def read_upload(file: UploadFile, settings: Settings) -> tuple[bytes, str]:
    """Read the upload in chunks (size-limited) and check its type: (bytes, MIME type by magic bytes).
    HTTP errors for unusable files."""
    chunks, size = [], 0
    while chunk := await file.read(CHUNK):
        size += len(chunk)
        if size > settings.max_upload_bytes:
            raise ApiError(413, "too_large", "file too large")
        chunks.append(chunk)
    data = b"".join(chunks)
    try:
        mime = validate_upload(data, settings.max_upload_bytes)
    except FileRejected as e:
        raise ApiError(REJECT_STATUS.get(e.code, 400), e.code, str(e)) from None
    return data, mime
