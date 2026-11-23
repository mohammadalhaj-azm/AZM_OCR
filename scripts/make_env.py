"""Write a production .env for the Docker deployment, from .env.example with freshly generated secrets.

    python3 scripts/make_env.py           # GPU server: docker-compose.yml + docker-compose.gpu.yml
    python3 scripts/make_env.py --cpu     # CPU only: docker-compose.yml, no handwriting model

Standard library only (runs with the host's python3, no uv needed). Refuses to overwrite an existing .env unless
--force is given (new secrets would lock you out of the stored results: RESULT_ENCRYPTION_KEY and the Postgres
password must stay the same once the database exists). The generated API key is printed once; keep it.
"""

import argparse
import base64
import os
import re
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def settings(gpu: bool) -> dict[str, str]:
    values = {
        "API_KEYS": secrets.token_urlsafe(32),
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "RESULT_ENCRYPTION_KEY": base64.b64encode(os.urandom(32)).decode(),
        "LOG_FORMAT": "json",  # .env.example says text; a server wants one JSON object per line
        # LLM on its own thread so OCR (CPU) and the LLM (Ollama) overlap; the setting's comment in .env.example
        "LLM_WORKER_THREADS": "1" if gpu else "0",
        "HANDWRITING_ENABLED": "true" if gpu else "false",
    }
    if gpu:
        values["COMPOSE_FILE"] = "docker-compose.yml:docker-compose.gpu.yml"
        values["HANDWRITING_DEVICE"] = "cuda"  # fail loudly instead of falling back to the CPU
    return values


def render(example: str, values: dict[str, str]) -> str:
    out, done = [], set()
    for line in example.splitlines():
        m = re.match(r"^#?\s*([A-Z][A-Z0-9_]*)=", line)
        if m and m.group(1) in values and m.group(1) not in done:
            key = m.group(1)
            out.append(f"{key}={values[key]}")
            done.add(key)
        else:
            out.append(line)
    missing = [k for k in values if k not in done]
    if missing:
        out += ["", "# Docker deployment (scripts/make_env.py)"] + [f"{k}={values[k]}" for k in missing]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="no GPU: base compose file only, handwriting off")
    parser.add_argument("--force", action="store_true", help="overwrite an existing .env")
    args = parser.parse_args()

    target, example = ROOT / ".env", ROOT / ".env.example"
    if target.exists() and not args.force:
        print(f"{target} already exists; not touching it (use --force to replace it, which changes every secret).",
              file=sys.stderr)
        return 1
    values = settings(gpu=not args.cpu)
    target.write_text(render(example.read_text(encoding="utf-8"), values), encoding="utf-8")
    target.chmod(0o600)
    print(f"wrote {target} (mode 600)")
    print(f"API key (send it as the X-Api-Key header; it is also in .env): {values['API_KEYS']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
