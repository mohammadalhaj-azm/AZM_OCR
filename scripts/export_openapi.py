"""Write the REST API's OpenAPI document to docs/openapi.json: the contract client teams generate their clients from.

    uv run python scripts/export_openapi.py           # after an intended API change; then review the diff
    uv run python scripts/export_openapi.py --check   # exit 1 if docs/openapi.json is out of date (e.g. in CI)

tests/test_api.py fails while the file is out of date, so every API change is a reviewed diff of this file.
In /v1 only additive changes are allowed (docs/INTEGRATION.md, "API versioning and compatibility").
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "openapi.json"


def render(app) -> str:
    return json.dumps(app.openapi(), indent=2, ensure_ascii=False) + "\n"


def build_app():
    from app.api.main import create_app
    from app.config import Settings

    # The document doesn't depend on the configuration; pin what could make create_app refuse to start.
    settings = Settings(_env_file=None, environment="development", api_keys="openapi-export", clients_file="",
                        webhook_secret="", webhook_allowed_hosts="", job_store="memory")
    return create_app(settings, services=object())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="export_openapi.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only check that docs/openapi.json is up to date")
    args = parser.parse_args(argv)
    text = render(build_app())
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if args.check:
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is out of date: run scripts/export_openapi.py", file=sys.stderr)
            return 1
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({'changed' if current != text else 'unchanged'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
