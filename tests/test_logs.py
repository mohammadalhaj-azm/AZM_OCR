"""JSON logs: format, context ids, and no exception details."""

import json
import logging

from app import logs


def record(msg="hello %s", args=("world",), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("app.test", logging.WARNING, __file__, 1, msg, args, exc_info)


def test_json_line_has_the_fields_and_the_context_ids():
    with logs.context(request_id="req-1", client_id="civil"), logs.context(job_id="job-9"):
        line = json.loads(logs.JsonFormatter().format(record()))
    assert line["msg"] == "hello world" and line["level"] == "WARNING" and line["logger"] == "app.test"
    assert line["ts"].endswith("+00:00") and "T" in line["ts"]
    assert (line["request_id"], line["job_id"], line["client_id"]) == ("req-1", "job-9", "civil")


def test_context_is_restored_and_empty_values_are_left_out():
    with logs.context(request_id="outer"):
        with logs.context(request_id="inner", job_id=None):
            assert logs.current() == {"request_id": "inner"}
        assert logs.current() == {"request_id": "outer"}
    assert logs.current() == {}
    assert set(json.loads(logs.JsonFormatter().format(record()))) == {"ts", "level", "logger", "msg"}


def test_exceptions_give_their_type_only():
    try:
        raise ValueError("SECRET_TEXT from a document")
    except ValueError:
        import sys

        line = logs.JsonFormatter().format(record("failed", (), sys.exc_info()))
    assert json.loads(line)["exc_type"] == "ValueError" and "SECRET_TEXT" not in line and "Traceback" not in line


def test_json_mode_reformats_uvicorn_loggers():
    access = logging.getLogger("uvicorn.access")
    handler = logging.StreamHandler()
    access.addHandler(handler)
    try:
        logs.configure("json")
        assert isinstance(handler.formatter, logs.JsonFormatter)
    finally:
        access.removeHandler(handler)


def test_log_context_covers_sending_the_response():
    """uvicorn writes its access line while the response is sent: the request id must still be set then."""
    import anyio

    from app.api.clients import ClientRegistry
    from app.api.middleware import LogContext

    seen = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message):
        seen.append((message["type"], logs.current()))

    scope = {"type": "http", "path": "/health/live", "headers": [(b"x-request-id", b"abc-1")]}
    anyio.run(LogContext(app, ClientRegistry([], {}, "api_keys")), scope, None, send)
    assert seen == [("http.response.start", {"request_id": "abc-1"})] and scope["state"]["request_id"] == "abc-1"
    assert logs.current() == {}  # nothing leaks into the next request
