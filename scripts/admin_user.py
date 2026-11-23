"""Admin dashboard users (ADMIN_USERS_FILE stores only scrypt password hashes).

    uv run python scripts/admin_user.py hash      # hash a password (typed twice, or piped in; never an argument)
    uv run python scripts/admin_user.py secret    # a new ADMIN_SESSION_SECRET

Put the hash in ADMIN_USERS_FILE (see config/admins.example.yaml):

    admins:
      alice: {name: "Alice Admin", role: admin, password_scrypt: "scrypt$16384$8$1$..."}
"""

import argparse
import getpass
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.admin.auth import hash_password  # noqa: E402

MIN_PASSWORD = 12


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="admin_user.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["hash", "secret"])
    args = parser.parse_args(argv)

    if args.command == "secret":
        print(secrets.token_urlsafe(32))
        return 0
    if sys.stdin.isatty():
        password = getpass.getpass("Password: ")
        if getpass.getpass("Repeat: ") != password:
            print("the passwords differ", file=sys.stderr)
            return 2
    else:
        password = sys.stdin.readline().rstrip("\r\n")
    if len(password) < MIN_PASSWORD:
        print(f"use at least {MIN_PASSWORD} characters", file=sys.stderr)
        return 2
    print(hash_password(password))
    return 0


if __name__ == "__main__":
    sys.exit(main())
