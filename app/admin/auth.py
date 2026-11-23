"""Admin dashboard authentication: users from ADMIN_USERS_FILE, scrypt password hashes, a signed session cookie.

    admins:
      alice:                                  # the user name (1-64 of a-z 0-9 . _ -)
        name: "Alice Admin"
        role: admin                           # admin: may change things; viewer: read-only
        password_scrypt: "scrypt$16384$8$1$<salt>$<hash>"   # uv run python scripts/admin_user.py hash

The file holds no secrets (like CLIENTS_FILE), and it is re-read when it changes: removing a user or changing a role
takes effect on the next request. Sessions are HMAC-signed with ADMIN_SESSION_SECRET (shared by all replicas) and
expire after ADMIN_SESSION_HOURS. Changing a user's password hash ends that user's sessions.

CSRF: the cookie is SameSite=Strict, and every request that changes something must also send `X-Admin-Request: 1`
(a header a cross-site form can't set) and, if the browser sends one, an Origin of this host.
"""

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from fastapi import Depends, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.api.errors import ApiError
from app.config import ConfigError, Settings

log = logging.getLogger("app.admin")
COOKIE = "ocr_admin"
USER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
Role = Literal["admin", "viewer"]
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
FAILURES_BEFORE_DELAY = 5
MAX_LOCK_S = 900


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    b64 = base64.b64encode
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${b64(salt).decode()}${b64(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        want = base64.b64decode(digest)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=len(want))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, want)


class _UserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    role: Role = "viewer"
    password_scrypt: str = Field(pattern=r"^scrypt\$\d+\$\d+\$\d+\$[A-Za-z0-9+/=]+\$[A-Za-z0-9+/=]+$")


class _UsersFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    admins: dict[str, _UserConfig] = Field(min_length=1)


@dataclass(frozen=True)
class AdminUser:
    username: str
    name: str
    role: Role
    password_scrypt: str

    @property
    def can_write(self) -> bool:
        return self.role == "admin"

    def public(self) -> dict:
        return {"username": self.username, "name": self.name, "role": self.role}

    def __repr__(self) -> str:  # never the hash
        return f"AdminUser({self.username!r}, role={self.role!r})"


def load_users(path: Path) -> dict[str, AdminUser]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as e:
        raise ConfigError(f"ADMIN_USERS_FILE {path} can't be read ({type(e).__name__})") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"ADMIN_USERS_FILE {path} is not valid YAML: {e}") from None
    try:
        cfg = _UsersFile.model_validate(raw)
    except ValidationError as e:  # holds only hashes: the details can be shown
        raise ConfigError(f"ADMIN_USERS_FILE {path} is invalid: {e}") from None
    for name in cfg.admins:
        if not USER_NAME.fullmatch(name):
            raise ConfigError(f"admin user name {name!r}: use 1-64 of a-z 0-9 . _ - (starting with a letter or digit)")
    return {name: AdminUser(name, u.name, u.role, u.password_scrypt) for name, u in cfg.admins.items()}


class Throttle:
    """Failed logins per (user, client address): after FAILURES_BEFORE_DELAY failures, each further attempt waits
    an exponentially growing time (up to MAX_LOCK_S)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._failures: dict[tuple[str, str], tuple[int, float]] = {}

    def retry_after(self, key: tuple[str, str]) -> float:
        with self._lock:
            _, until = self._failures.get(key, (0, 0.0))
        return max(0.0, until - time.time())

    def failed(self, key: tuple[str, str]) -> None:
        with self._lock:
            n, _ = self._failures.get(key, (0, 0.0))
            n += 1
            wait = 0.0 if n < FAILURES_BEFORE_DELAY else min(MAX_LOCK_S, 2 ** (n - FAILURES_BEFORE_DELAY))
            self._failures[key] = (n, time.time() + wait)
            if len(self._failures) > 10_000:  # bounded memory under a spray of names
                self._failures = dict(list(self._failures.items())[-5_000:])

    def succeeded(self, key: tuple[str, str]) -> None:
        with self._lock:
            self._failures.pop(key, None)


class AdminAuth:
    def __init__(self, settings: Settings):
        self.s = settings
        self.path = Path(settings.admin_users_file)
        self._secret = settings.admin_session_secret.encode() or secrets.token_bytes(32)
        if not settings.admin_session_secret:
            log.warning("ADMIN_SESSION_SECRET is not set: admin sessions are lost on restart and not shared between "
                        "replicas (development only)")
        self._users = load_users(self.path)  # ConfigError at startup when invalid
        self._mtime = self._stat()
        self._lock = threading.Lock()
        self.throttle = Throttle()

    def _stat(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def users(self) -> dict[str, AdminUser]:
        """The users, re-read when the file changed (an invalid new version keeps the previous one)."""
        mtime = self._stat()
        if mtime != self._mtime:
            with self._lock:
                if mtime != self._mtime:
                    self._mtime = mtime
                    try:
                        self._users = load_users(self.path)
                        log.info("admin users reloaded users=%d", len(self._users))
                    except ConfigError as e:
                        log.error("admin users file invalid, keeping the previous version: %s", e)
        return self._users

    # --- sessions ----------------------------------------------------------------------------------------------
    def _sign(self, payload: bytes) -> str:
        return base64.urlsafe_b64encode(hmac.new(self._secret, payload, hashlib.sha256).digest()).decode().rstrip("=")

    @staticmethod
    def _pw_tag(user: AdminUser) -> str:
        """Changes when the password hash changes, so a password change ends the user's sessions."""
        return hashlib.sha256(user.password_scrypt.encode()).hexdigest()[:12]

    def issue(self, user: AdminUser) -> str:
        now = int(time.time())
        payload = json.dumps({"u": user.username, "iat": now, "exp": now + int(self.s.admin_session_hours * 3600),
                              "pw": self._pw_tag(user)}, separators=(",", ":")).encode()
        body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        return f"{body}.{self._sign(payload)}"

    def session_user(self, token: str | None) -> AdminUser | None:
        if not token or "." not in token:
            return None
        body, sig = token.rsplit(".", 1)
        try:
            payload = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        except ValueError:
            return None
        if not hmac.compare_digest(self._sign(payload), sig):
            return None
        try:
            data = json.loads(payload)
        except ValueError:
            return None
        user = self.users().get(data.get("u"))
        if user is None or data.get("exp", 0) < time.time() or data.get("pw") != self._pw_tag(user):
            return None
        return user

    def login(self, username: str, password: str, address: str) -> tuple[AdminUser | None, float]:
        """(user, 0) on success; (None, 0) for a wrong name or password; (None, seconds to wait) when throttled
        (nothing was checked). Always does the scrypt work, so the answer time doesn't reveal whether the user
        exists."""
        key = (username.lower()[:64], address)
        wait = self.throttle.retry_after(key)
        if wait > 0:
            return None, wait
        user = self.users().get(username)
        ok = verify_password(password, user.password_scrypt if user else _dummy_hash())
        if user is None or not ok:
            self.throttle.failed(key)  # the next attempts may have to wait
            return None, 0.0
        self.throttle.succeeded(key)
        return user, 0.0

    def cookie_kwargs(self) -> dict:
        return {"key": COOKIE, "httponly": True, "samesite": "strict", "path": "/admin",
                "secure": self.s.environment == "production", "max_age": int(self.s.admin_session_hours * 3600)}


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password(secrets.token_urlsafe(16))


def client_address(request: Request) -> str:
    return request.client.host if request.client else "-"


def current_user(request: Request) -> AdminUser:
    """Dependency: the logged-in admin user (401 otherwise). Requests that change something must also pass the
    CSRF checks."""
    auth: AdminAuth = request.app.state.admin.auth
    user = auth.session_user(request.cookies.get(COOKIE))
    if user is None:
        raise ApiError(401, "unauthorized", "not logged in")
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        check_csrf(request)
    request.state.admin_user = user
    return user


def check_csrf(request: Request) -> None:
    if request.headers.get("x-admin-request") != "1":
        raise ApiError(403, "forbidden", "missing X-Admin-Request header")
    origin = request.headers.get("origin")
    if origin and origin.split("://", 1)[-1].rstrip("/") != request.headers.get("host", ""):
        raise ApiError(403, "forbidden", "cross-origin request")


def admin_user(user: AdminUser = Depends(current_user)) -> AdminUser:
    """Dependency: a logged-in user with the admin role (403 for viewers)."""
    if not user.can_write:
        raise ApiError(403, "forbidden", "this needs the admin role")
    return user
