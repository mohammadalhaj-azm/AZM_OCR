"""Load test: send N extract-and-answer jobs (or /v1/documents jobs) at the same time and measure how the server copes.

    export OCR_API_KEY=...                                   # never on the command line (shell history)
    uv run python scripts/load_test.py sample.pdf --jobs 10 --question "Who are the parties?"
    uv run python scripts/load_test.py sample.pdf --jobs 10 20 50 --url http://127.0.0.1:8000

All jobs of a round are submitted together, then each is polled (long-poll) until it is done or failed. Prints per
round: how many finished / failed / were refused, end-to-end time p50 / p95 / max, queue wait (submit until a worker
took it, from the server's own `timingsMs` when present), throughput, and the peak GPU memory (when `nvidia-smi` is on
this machine's PATH). Finished jobs are deleted right away (`--keep` to keep them for RESULT_TTL_S).

Prints numbers only: never the document's text, the answers or anything else from a result.
"""

import argparse
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ENDPOINTS = {"answer": "/v1/extract-and-answer", "documents": "/v1/documents"}


class GpuSampler(threading.Thread):
    """Peak `memory.used` (MiB) over all GPUs of this machine while the round runs; None without nvidia-smi."""

    def __init__(self, every_s: float = 1.0):
        super().__init__(daemon=True)
        self.every_s, self.peak, self._stop = every_s, None, threading.Event()
        self.exe = shutil.which("nvidia-smi")

    def run(self) -> None:
        while self.exe and not self._stop.is_set():
            try:
                cmd = [self.exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
                used = sum(int(line.split(",")[0]) for line in out.strip().splitlines() if line.strip())
                self.peak = max(self.peak or 0, used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.every_s)

    def stop(self) -> int | None:
        self._stop.set()
        return self.peak


def one_job(http: httpx.Client, args, data: bytes, t_start: float) -> dict:
    """Submit (retrying 503 queue_full after Retry-After until --give-up-s), then poll until final."""
    files = {"file": (Path(args.file).name, data)}
    form: dict = {}
    if args.endpoint == "answer":
        if args.question:
            form["questions"] = list(args.question)  # the field repeated once per question
        if args.no_correct:
            form["correct"] = "false"
    while True:
        r = http.post(ENDPOINTS[args.endpoint], files=files, data=form)
        if r.status_code == 202:
            break
        if r.status_code == 503 and time.monotonic() - t_start < args.give_up_s:
            time.sleep(float(r.headers.get("Retry-After") or 5))
            continue
        code = (r.json().get("error") or {}).get("code") if r.headers.get("content-type", "").startswith(
            "application/json") else None
        return {"outcome": "refused", "http": r.status_code, "code": code}
    job_id = r.json()["jobId"]
    submitted = time.monotonic()
    while True:
        r = http.get(f"/v1/jobs/{job_id}", params={"wait": 30})
        r.raise_for_status()
        body = r.json()
        if body["status"] in ("done", "failed"):
            break
        if time.monotonic() - t_start > args.give_up_s:
            return {"outcome": "timeout", "jobId": job_id}
    end = time.monotonic()
    out = {"outcome": body["status"], "seconds": end - t_start, "afterSubmit": end - submitted,
           "timings": {k: v for k, v in (body.get("timingsMs") or {}).items() if isinstance(v, (int, float))},
           "code": (body.get("error") or {}).get("code")}
    if not args.keep:
        try:
            http.delete(f"/v1/jobs/{job_id}")
        except httpx.HTTPError:
            pass
    return out


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, max(0, round(p / 100 * len(values) + 0.5) - 1))]


def run_round(args, data: bytes, n: int) -> None:
    gpu = GpuSampler()
    gpu.start()
    t_start = time.monotonic()
    headers = {"X-Api-Key": args.key}
    with httpx.Client(base_url=args.url, headers=headers, timeout=60) as http, ThreadPoolExecutor(n) as pool:
        results = list(pool.map(lambda _: _safe(http, args, data, t_start), range(n)))
    wall = time.monotonic() - t_start
    peak = gpu.stop()

    by = {}
    for r in results:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    done = [r for r in results if r["outcome"] == "done"]
    print(f"\n=== {n} jobs at once ({args.endpoint}) ===")
    print("outcomes:", ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    codes = sorted({str(r.get("code") or r.get("http") or r.get("error")) for r in results if r["outcome"] != "done"})
    if codes:
        print("not done:", ", ".join(codes))
    if done:
        secs = [r["seconds"] for r in done]
        print(f"end to end (s): p50 {pct(secs, 50):.1f}  p95 {pct(secs, 95):.1f}  max {max(secs):.1f}  "
              f"mean {statistics.mean(secs):.1f}")
        print(f"throughput: {len(done) / max(wall, 1e-3) * 60:.1f} jobs/min (round took {wall:.1f} s)")
        keys = sorted({k for r in done for k in r["timings"]})
        if keys:
            print("server timings, mean ms:", "  ".join(
                f"{k} {statistics.mean(r['timings'].get(k, 0) for r in done):.0f}" for k in keys))
    print("GPU memory peak:", f"{peak} MiB" if peak is not None else "n/a (no nvidia-smi on this machine)")


def _safe(http, args, data, t_start) -> dict:
    try:
        return one_job(http, args, data, t_start)
    except (httpx.HTTPError, ValueError, KeyError) as e:  # never the response body: it may hold document text
        return {"outcome": "error", "error": type(e).__name__}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", help="the sample document every job sends (use a non-sensitive one)")
    ap.add_argument("--jobs", type=int, nargs="+", default=[10], help="jobs at once, one round per value")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="answer")
    ap.add_argument("--question", action="append", default=[], help="a question (repeat for more); answer only")
    ap.add_argument("--no-correct", action="store_true", help="answer: correct=false (no correction LLM calls)")
    ap.add_argument("--give-up-s", type=float, default=1800, help="a round's jobs give up after this long")
    ap.add_argument("--keep", action="store_true", help="don't delete the finished jobs")
    ap.add_argument("--pause-s", type=float, default=10, help="pause between rounds")
    args = ap.parse_args()
    args.key = os.environ.get("OCR_API_KEY", "")
    if not args.key:
        print("set OCR_API_KEY (the X-Api-Key of a client with the documents scope)", file=sys.stderr)
        return 2
    data = Path(args.file).read_bytes()
    for i, n in enumerate(args.jobs):
        if i:
            time.sleep(args.pause_s)
        run_round(args, data, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
