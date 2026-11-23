"""Log records for the admin dashboard.

A logging handler on the root logger keeps this process's latest LOG_BUFFER_SIZE records (all levels that reach
it) in RAM, and hands WARNING+ to a background writer that stores them in batches, so warnings and errors of every
replica are searchable in one place.

Privacy: the service's log messages never hold document content (tests check this). Like the JSON log format, a
record keeps only an exception's type, never its message or traceback.
"""

import itertools
import logging
import queue
import threading
import time
from collections import deque

from app import logs
from app.admin.store import filter_logs

WRITER_THREAD = "admin-log-writer"
PERSIST_LEVEL = logging.WARNING
BATCH = 200


class LogBuffer(logging.Handler):
    def __init__(self, instance: str, size: int, store=None):
        super().__init__(level=logging.DEBUG)
        self.instance, self.store = instance, store
        self.records: deque[dict] = deque(maxlen=max(100, size))
        self._ids = itertools.count(1)
        self._lock_rec = threading.Lock()
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=5000)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - a bad format string must not break logging
            msg = str(record.msg)
        r = {"at": record.created, "instance": self.instance, "level": record.levelname, "logger": record.name,
             "msg": msg[:4000], **logs.current()}
        if record.exc_info and record.exc_info[0] is not None:
            r["exc_type"] = record.exc_info[0].__name__
        with self._lock_rec:
            r["id"] = next(self._ids)
            self.records.append(r)
        # never persist the writer's own records (a database error would loop)
        if (self.store is not None and record.levelno >= PERSIST_LEVEL
                and threading.current_thread().name != WRITER_THREAD):
            try:
                self._queue.put_nowait(r)
            except queue.Full:
                self.dropped += 1

    def recent(self, **filters) -> list[dict]:
        with self._lock_rec:
            rows = list(self.records)
        return filter_logs(reversed(rows), **filters)

    # --- writer ------------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self.store is None:
            return
        self._thread = threading.Thread(target=self._loop, name=WRITER_THREAD, daemon=True)
        self._thread.start()

    def _flush(self) -> None:
        batch = []
        while len(batch) < BATCH:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            try:
                self.store.write_logs([{k: v for k, v in r.items() if k != "id"} for r in batch])
            except Exception:  # noqa: BLE001 - dropped: logging about it would only add more records
                self.dropped += len(batch)

    def _loop(self) -> None:
        while not self._stop.wait(2.0):
            while not self._queue.empty():
                self._flush()
        self._flush()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)

    def install(self) -> None:
        logging.getLogger().addHandler(self)
        self.start()

    def uninstall(self) -> None:
        logging.getLogger().removeHandler(self)
        self.stop()


def wait_flushed(buffer: LogBuffer, timeout: float = 5) -> None:
    """For tests: wait until the writer has stored everything queued."""
    end = time.monotonic() + timeout
    while not buffer._queue.empty() and time.monotonic() < end:
        time.sleep(0.05)
