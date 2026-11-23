"""API keys and webhook secrets for the client registry (CLIENTS_FILE stores only key hashes).

    uv run python scripts/client_key.py new      # a new API key (give it to the client) + the hash for CLIENTS_FILE
    uv run python scripts/client_key.py hash     # the hash of an existing key (typed or piped in, never an argument)
    uv run python scripts/client_key.py secret   # a new webhook signing secret (for a client's secret_env variable)

Keys and secrets are printed once and never stored by this script. Send them over a secure channel.
"""

import argparse
import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.clients import key_hash  # noqa: E402

TOKEN_BYTES = 32  # token_urlsafe(32): 43 characters, above the 32-character production minimum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="client_key.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["new", "hash", "secret"])
    args = parser.parse_args(argv)

    if args.command == "new":
        key = secrets.token_urlsafe(TOKEN_BYTES)
        print(f"API key (give to the client):  {key}")
        print(f"key_sha256 (for CLIENTS_FILE): {key_hash(key)}")
    elif args.command == "hash":
        # Never from argv: it would end up in the shell history and process list.
        key = getpass.getpass("API key: ") if sys.stdin.isatty() else sys.stdin.readline()
        key = key.strip()
        if not key:
            print("no key given", file=sys.stderr)
            return 2
        print(key_hash(key))
    else:
        print(secrets.token_urlsafe(TOKEN_BYTES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
