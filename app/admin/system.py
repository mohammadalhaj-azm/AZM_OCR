"""Resource sampling for the admin dashboard: CPU, memory, disks, disk and network I/O, NVIDIA GPUs (NVML), the
service's own dependencies and work, every SAMPLE_INTERVAL_S.

Recent samples stay in RAM (the live view of this replica); every SAMPLE_PERSIST_S one is stored, so the dashboard
can show every replica and the history, and so a replica that stops sampling is noticed (alert `replica_missing`).
Real measurements only: a value that can't be measured is None, never made up.
"""

import logging
import os
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import psutil

from app import __version__
from app.config import Settings

log = logging.getLogger("app.admin")
HEALTH_EVERY_S = 5.0
LIVE_WINDOW_S = 900  # samples kept in RAM


class Gpus:
    """NVIDIA GPUs through NVML (nvidia-ml-py). Without a driver / GPU: status 'none' and no devices."""

    def __init__(self):
        self.status, self.detail = "none", "no NVIDIA driver (CPU only)"
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.count = pynvml.nvmlDeviceGetCount()
            self.status = "ok" if self.count else "none"
            self.detail = pynvml.nvmlSystemGetDriverVersion() if self.count else "no NVIDIA GPU"
            if isinstance(self.detail, bytes):
                self.detail = self.detail.decode()
        except Exception as e:  # noqa: BLE001 - NVMLError_LibraryNotFound, DriverNotLoaded, ...
            self.nvml, self.count = None, 0
            if type(e).__name__ not in ("NVMLError_LibraryNotFound", "NVMLError_DriverNotLoaded", "ImportError"):
                self.status, self.detail = "error", type(e).__name__

    def read(self) -> list[dict]:
        if self.nvml is None:
            return []
        n, out = self.nvml, []
        for i in range(self.count):
            try:
                h = n.nvmlDeviceGetHandleByIndex(i)
                util = n.nvmlDeviceGetUtilizationRates(h)
                mem = n.nvmlDeviceGetMemoryInfo(h)
                name = n.nvmlDeviceGetName(h)
                gpu = {"index": i, "name": name.decode() if isinstance(name, bytes) else name, "util": util.gpu,
                       "memUtil": util.memory, "memTotal": mem.total, "memUsed": mem.used,
                       "memPercent": round(mem.used / mem.total * 100, 1) if mem.total else None,
                       "temperature": None, "powerW": None, "powerLimitW": None}
                for key, fn in (("temperature", lambda: n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)),
                                ("powerW", lambda: round(n.nvmlDeviceGetPowerUsage(h) / 1000, 1)),
                                ("powerLimitW", lambda: round(n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000, 1))):
                    try:
                        gpu[key] = fn()
                    except Exception:  # noqa: BLE001, S110 - not every GPU reports every value
                        pass
                out.append(gpu)
            except Exception as e:  # noqa: BLE001
                out.append({"index": i, "error": type(e).__name__})
        return out

    def close(self) -> None:
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001, S110
                pass


class ResourceSampler:
    def __init__(self, settings: Settings, instance: str, store, health: Callable[[], dict],
                 work: Callable[[], dict]):
        self.s, self.instance, self.store = settings, instance, store
        self._health_fn, self._work_fn = health, work
        self.host, self.pid = socket.gethostname(), os.getpid()
        self.started_at = time.time()
        self.proc = psutil.Process(self.pid)
        self.gpus = Gpus()
        self.buffer: deque[dict] = deque(maxlen=max(10, int(LIVE_WINDOW_S / max(0.5, settings.sample_interval_s))))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_io: tuple[float, object, object] | None = None
        self._health: tuple[float, dict] = (0.0, {})
        self._last_persist = 0.0
        self._last_purge = 0.0
        self._lock = threading.Lock()
        psutil.cpu_percent(None)  # the first call only starts the measurement
        psutil.cpu_percent(None, percpu=True)
        self.proc.cpu_percent(None)

    # --- measuring ---------------------------------------------------------------------------------------------
    def _disks(self) -> list[dict]:
        out, seen = [], {}
        for label, path in (("app", Path.cwd()), ("uploads", Path(self.s.upload_dir))):
            p = path if path.exists() else path.parent
            try:
                u = psutil.disk_usage(str(p))
            except OSError:
                continue
            key = (u.total, round(u.free / 2 ** 20))
            if key in seen:  # the same volume
                seen[key]["label"] += f" + {label}"
                continue
            seen[key] = d = {"label": label, "path": str(p), "total": u.total, "used": u.used, "free": u.free,
                             "percent": u.percent}
            out.append(d)
        return out

    def _io(self, now: float) -> dict:
        try:
            disk, net = psutil.disk_io_counters(), psutil.net_io_counters()
        except Exception:  # noqa: BLE001 - e.g. no disk counters in some containers
            disk, net = None, None
        rates = {"diskReadBps": None, "diskWriteBps": None, "netSentBps": None, "netRecvBps": None}
        if self._last_io is not None:
            t0, d0, n0 = self._last_io
            dt = max(1e-3, now - t0)
            if disk is not None and d0 is not None:
                rates["diskReadBps"] = max(0.0, (disk.read_bytes - d0.read_bytes) / dt)
                rates["diskWriteBps"] = max(0.0, (disk.write_bytes - d0.write_bytes) / dt)
            if net is not None and n0 is not None:
                rates["netSentBps"] = max(0.0, (net.bytes_sent - n0.bytes_sent) / dt)
                rates["netRecvBps"] = max(0.0, (net.bytes_recv - n0.bytes_recv) / dt)
        self._last_io = (now, disk, net)
        return {k: round(v) if v is not None else None for k, v in rates.items()}

    def health(self) -> dict:
        """Dependency health, re-checked at most every HEALTH_EVERY_S."""
        now = time.monotonic()
        if now - self._health[0] >= HEALTH_EVERY_S or not self._health[1]:
            try:
                self._health = (now, self._health_fn())
            except Exception as e:  # noqa: BLE001
                log.error("health check failed error=%s", type(e).__name__)
        return self._health[1]

    def collect(self) -> dict:
        now = time.time()
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        try:
            load = [round(x, 2) for x in psutil.getloadavg()]
        except (AttributeError, OSError):
            load = None
        with self.proc.oneshot():
            rss, threads = self.proc.memory_info().rss, self.proc.num_threads()
            proc_cpu = self.proc.cpu_percent(None)
        try:
            work = self._work_fn()
        except Exception:  # noqa: BLE001
            work = {}
        return {
            "instance": self.instance, "host": self.host, "pid": self.pid, "version": __version__, "at": now,
            "startedAt": self.started_at, "uptimeS": round(now - self.started_at),
            "cpu": {"percent": psutil.cpu_percent(None), "perCore": psutil.cpu_percent(None, percpu=True),
                    "count": psutil.cpu_count(), "load": load, "process": round(proc_cpu, 1)},
            "memory": {"total": vm.total, "used": vm.total - vm.available, "available": vm.available,
                       "percent": vm.percent, "swapTotal": sw.total, "swapUsed": sw.used, "swapPercent": sw.percent,
                       "processRss": rss, "processThreads": threads},
            "disks": self._disks(), "io": self._io(now),
            "gpus": self.gpus.read(), "gpuStatus": self.gpus.status, "gpuDetail": self.gpus.detail,
            "health": self.health(), "work": work,
        }

    def latest(self) -> dict | None:
        with self._lock:
            return self.buffer[-1] if self.buffer else None

    def recent(self, since: float) -> list[dict]:
        with self._lock:
            return [s for s in self.buffer if s["at"] >= since]

    # --- thread ------------------------------------------------------------------------------------------------
    def tick(self) -> dict:
        sample = self.collect()
        with self._lock:
            self.buffer.append(sample)
        now = time.time()
        if now - self._last_persist >= self.s.sample_persist_s:
            self._last_persist = now
            try:
                self.store.write_sample(self.instance, sample)
            except Exception as e:  # noqa: BLE001 - e.g. database briefly down: the live view still works
                log.warning("sample write failed error=%s", type(e).__name__)
        if now - self._last_purge >= 600:
            self._last_purge = now
            try:
                self.store.purge_samples_before(now - self.s.sample_retention_h * 3600)
                self.store.purge_logs_before(now - self.s.log_retention_days * 86400)
            except Exception as e:  # noqa: BLE001
                log.warning("sample purge failed error=%s", type(e).__name__)
        return sample

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="admin-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 - keep sampling
                log.error("sampling failed error=%s", type(e).__name__)
            self._stop.wait(max(0.5, self.s.sample_interval_s))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)
        try:  # a clean stop: other replicas don't report this one as missing
            last = self.latest() or self.collect()
            self.store.write_sample(self.instance, {**last, "at": time.time(), "stopping": True})
        except Exception:  # noqa: BLE001, S110
            pass
        self.gpus.close()
