"""Logging setup: text (default) or JSON lines, with the current request, job and client on every JSON line.

    LOG_FORMAT=json   {"ts": "...", "level": "INFO", "logger": "app.api", "msg": "job finished ...",
                       "request_id": "...", "job_id": "...", "client_id": "..."}

`request_id`, `job_id` and `client_id` come from context variables. The HTTP middleware sets them for a request, the
worker and the webhook dispatcher for the job they work on, so every line written meanwhile carries them, also from
modules that know nothing about requests (LLM client, store, webhooks). Fields without a value are left out.

Privacy: log messages never hold document content (see the tests). For an exception, JSON lines carry only its type
(`exc_type`): never its message or traceback, which can quote document text.
"""

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
client_id: ContextVar[str | None] = ContextVar("client_id", default=None)
_VARS = {"request_id": request_id, "job_id": job_id, "client_id": client_id}
TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


@contextmanager
def context(**values: str | None) -> Iterator[None]:
    """Set request_id / job_id / client_id for the code inside (None values are skipped)."""
    tokens = [(_VARS[name], _VARS[name].set(value)) for name, value in values.items() if value is not None]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def current() -> dict[str, str]:
    return {name: v for name, var in _VARS.items() if (v := var.get())}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line = {"ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
                "level": record.levelname, "logger": record.name, "msg": record.getMessage(), **current()}
        if record.exc_info and record.exc_info[0] is not None:
            line["exc_type"] = record.exc_info[0].__name__  # never the message or traceback
        return json.dumps(line, ensure_ascii=False)


def configure(log_format: str = "text", level: str = "INFO") -> None:
    """Set up the root logger once (if nothing else did), and in JSON mode give uvicorn's loggers the same format."""
    root = logging.getLogger()
    formatter = JsonFormatter() if log_format == "json" else logging.Formatter(TEXT_FORMAT)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(level.upper())
    if log_format == "json":  # uvicorn (CLI) has set up its own handlers before importing the app
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            for handler in logging.getLogger(name).handlers:
                handler.setFormatter(formatter)
