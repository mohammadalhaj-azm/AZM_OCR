"""A local webhook receiver for testing callbacks: prints every delivery and checks its signature.

    .venv\\Scripts\\python.exe scripts\\webhook_receiver.py --secret-env WEBHOOK_SECRET_CIVIL_REGISTRY
    .venv\\Scripts\\python.exe scripts\\webhook_receiver.py --secret-env MY_TEST_SECRET --fail 3   # 500 three times

Then upload with `callbackUrl=http://127.0.0.1:9000/ocr-done` (the client's allowed hosts must include 127.0.0.1).
The secret is read from the environment variable named by --secret-env, or from .env (never from an argument, so it
stays out of the shell history). The verification below is what a real receiver should do.
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_AGE_S = 300  # reject signatures older than this (replay protection)


def read_secret(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name:
                return value.strip().strip('"').strip("'")
    return ""


def verify(secret: str, timestamp: str, body: bytes, signature: str) -> tuple[bool, str]:
    if not secret:
        return False, "no secret given (--secret-env)"
    expected = "sha256=" + hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature or ""):
        return False, "signature mismatch (wrong secret?)"
    try:
        age = abs(time.time() - int(timestamp))
    except ValueError:
        return False, "bad timestamp"
    if age > MAX_AGE_S:
        return False, f"timestamp too old ({int(age)} s)"
    return True, "valid"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="webhook_receiver.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--secret-env", default="", help="name of the env var (or .env entry) with the signing secret")
    parser.add_argument("--fail", type=int, default=0, help="answer 500 to the first N deliveries (to see retries)")
    args = parser.parse_args(argv)
    secret = read_secret(args.secret_env) if args.secret_env else ""
    state = {"left_to_fail": args.fail, "seen": set()}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            ok, why = verify(secret, self.headers.get("X-Webhook-Timestamp", ""), body,
                             self.headers.get("X-Webhook-Signature", ""))
            try:
                data = json.loads(body)
            except ValueError:
                data = {}
            delivery = self.headers.get("X-Webhook-Id", "")
            repeat = " (repeat: de-duplicate on deliveryId)" if delivery in state["seen"] else ""
            state["seen"].add(delivery)
            fail = state["left_to_fail"] > 0
            if fail:
                state["left_to_fail"] -= 1
            status = 401 if not ok else (500 if fail else 200)
            print(f"{time.strftime('%H:%M:%S')}  {data.get('event')}  job={data.get('jobId')}  "
                  f"delivery={delivery}{repeat}  signature={why}  -> answered {status}", flush=True)
            if data.get("error"):
                print(f"          error code: {data['error'].get('code')}", flush=True)
            self.send_response(status)
            self.end_headers()

        def log_message(self, *a):  # quiet: the line above says it all
            pass

    print(f"listening on http://127.0.0.1:{args.port}/  (secret {'loaded' if secret else 'NOT set'}"
          f"{f', failing the first {args.fail}' if args.fail else ''}); Ctrl+C to stop", flush=True)
    try:
        ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
