"""Client registry: which systems may call the API, with which keys, scopes, webhook settings and limits.

With CLIENTS_FILE (YAML, see config/clients.example.yaml) every client system is listed by a stable id. The file holds
no secrets: API keys are stored as SHA-256 hashes (several per client, for rotation), and a client's webhook signing
secret is read from the environment variable the file names. The client id owns the client's jobs and appears in the
logs, so rotating a key never hides a client's jobs.

Without CLIENTS_FILE every API_KEYS entry is a client with all scopes and the global WEBHOOK_* settings, and its id
is the job owner used before the registry existed (a hash of the key), so existing jobs stay visible.

A client may use one or more processing profiles (`profiles:` in the same file, see app/profiles.py); the first one
in its list is its default, and a request can pick another with the `profile` field.

The registry is loaded at startup. With the admin dashboard, the registry can also be managed there (stored in the job
store's database, versioned): it then replaces CLIENTS_FILE, and every replica applies changes within CONFIG_POLL_S
(`ClientRegistry.replace_with`), without a restart.
"""

import hashlib
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from app.config import MIN_SECRET_CHARS, PLACEHOLDER, ConfigError, Settings
from app.profiles import DEFAULT, DEFAULT_PROFILE, Profile

Scope = Literal["extract", "documents"]
SCOPES: frozenset[str] = frozenset(Scope.__args__)
CLIENT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def key_hash(key: str) -> str:
    """What CLIENTS_FILE stores for an API key (scripts/client_key.py prints it)."""
    return hashlib.sha256(key.encode()).hexdigest()


# --- CLIENTS_FILE schema: unknown fields are errors, so a typo never silently drops a limit ------------------------

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebhookConfig(_Strict):
    secret_env: str | None = Field(None, pattern=r"^[A-Z_][A-Z0-9_]*$")  # name of the env var holding the secret
    managed_secret: bool = False  # the secret was generated in the admin dashboard (stored sealed in the database)
    allowed_hosts: list[str] = Field(min_length=1)  # callback host names this client may use
    retry_window_s: int | None = Field(None, ge=0)  # instead of the global WEBHOOK_RETRY_WINDOW_S

    @model_validator(mode="after")
    def _one_secret_source(self) -> "WebhookConfig":
        if (self.secret_env is None) == (not self.managed_secret):
            raise ValueError("set exactly one of secret_env and managed_secret")
        return self


class LimitsConfig(_Strict):
    max_queued_jobs: int | None = Field(None, ge=0)  # on top of the global MAX_QUEUED_JOBS
    extract_concurrent: int | None = Field(None, ge=1)  # on top of the global EXTRACT_MAX_CONCURRENT


class ClientConfig(_Strict):
    name: str = Field(min_length=1, max_length=200)
    key_sha256: list[str] = Field(min_length=1)
    scopes: list[Scope] = Field(default_factory=lambda: sorted(SCOPES), min_length=1)
    profiles: list[str] = Field(default_factory=lambda: [DEFAULT], min_length=1)  # the first one is the default
    webhook: WebhookConfig | None = None
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    enabled: bool = True

    @field_validator("key_sha256")
    @classmethod
    def _hex_hashes(cls, hashes: list[str]) -> list[str]:
        if not all(SHA256_HEX.fullmatch(h) for h in hashes):
            raise ValueError("entries must be 64 lowercase hex characters (see scripts/client_key.py)")
        return hashes


class RegistryFile(_Strict):
    profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)  # validated as app.profiles.Profile
    clients: dict[str, ClientConfig] = Field(min_length=1)


# --- Runtime ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Client:
    id: str  # owns the client's jobs; safe to log
    name: str
    scopes: frozenset[str] = SCOPES
    profiles: tuple[str, ...] = (DEFAULT,)  # allowed processing profiles; the first is the default
    enabled: bool = True
    webhook_secret: str = ""  # empty = this client can't use callbacks
    webhook_hosts: frozenset[str] = frozenset()
    webhook_retry_window_s: int | None = None  # None = WEBHOOK_RETRY_WINDOW_S
    max_queued_jobs: int | None = None
    extract_concurrent: int | None = None

    def __repr__(self) -> str:  # never show the webhook secret
        return f"Client(id={self.id!r}, name={self.name!r})"


Source = Literal["file", "api_keys", "dashboard"]


@dataclass(frozen=True)
class _Registry:
    by_id: dict[str, Client]
    by_hash: dict[str, str]  # sha256 hex of a key -> client id
    source: Source
    profiles: dict[str, Profile]


class ClientRegistry:
    def __init__(self, clients: list[Client], key_hashes: dict[str, str], source: Source,
                 profiles: dict[str, Profile] | None = None):
        # One immutable snapshot, swapped as a whole (replace_with): a request never sees half of a change.
        self._d = _Registry({c.id: c for c in clients}, dict(key_hashes), source,
                            {DEFAULT: DEFAULT_PROFILE, **(profiles or {})})

    def replace_with(self, other: "ClientRegistry") -> None:
        """Take over another registry's clients, keys and profiles (a change made in the admin dashboard). Everyone
        holding this object (middleware, routes, webhook dispatcher) sees the change at once."""
        self._d = other._d

    @property
    def source(self) -> Source:
        return self._d.source

    @property
    def profiles(self) -> dict[str, Profile]:
        return self._d.profiles

    @property
    def clients(self) -> list[Client]:
        return list(self._d.by_id.values())

    def key_count(self, client_id: str) -> int:
        return sum(1 for c in self._d.by_hash.values() if c == client_id)

    def key_fingerprints(self, client_id: str) -> list[str]:
        """The start of each key hash of a client (enough to tell keys apart; never the full hash)."""
        return sorted(h[:8] for h, c in self._d.by_hash.items() if c == client_id)

    def get(self, client_id: str) -> Client | None:
        return self._d.by_id.get(client_id)

    def lookup(self, key: str | None) -> Client | None:
        """The client this API key belongs to, enabled or not."""
        d = self._d
        client_id = d.by_hash.get(key_hash(key)) if key else None
        return d.by_id.get(client_id) if client_id else None

    def authenticate(self, key: str | None) -> Client | None:
        """The enabled client this API key belongs to, or None."""
        client = self.lookup(key)
        return client if client is not None and client.enabled else None

    def profile_for(self, client: Client, requested: str | None = None) -> Profile | None:
        """The profile to use: the requested one (None if the client may not use it), or the client's default."""
        name = requested or client.profiles[0]
        return self._d.profiles.get(name) if name in client.profiles else None

    def production_problems(self) -> list[str]:
        if self.source == "api_keys":
            return []  # API_KEYS and WEBHOOK_SECRET are checked by Settings.production_problems
        problems = []
        if not any(c.enabled for c in self.clients):
            problems.append(("CLIENTS_FILE" if self.source == "file" else "the dashboard registry")
                            + " has no enabled client")
        weak = sorted(c.id for c in self.clients if c.webhook_secret and (
            len(c.webhook_secret) < MIN_SECRET_CHARS or PLACEHOLDER.search(c.webhook_secret)))
        if weak:
            problems.append(f"webhook secret of client(s) {', '.join(weak)} is shorter than {MIN_SECRET_CHARS} "
                            "characters or a placeholder")
        return problems

    @classmethod
    def from_api_keys(cls, settings: Settings) -> "ClientRegistry":
        clients, hashes = [], {}
        for key in sorted(settings.api_key_set):
            h = key_hash(key)
            client = Client(id=h[:16], name="API_KEYS entry", webhook_secret=settings.webhook_secret,
                            webhook_hosts=frozenset(settings.webhook_host_set))
            clients.append(client)
            hashes[h] = client.id
        return cls(clients, hashes, "api_keys")


@lru_cache
def _dotenv(path: str = ".env") -> dict[str, str | None]:
    return dotenv_values(path) if Path(path).is_file() else {}


def _secret_from_env(name: str) -> str:
    """Like Settings: the process environment first, then a local .env file."""
    return os.environ.get(name) or _dotenv().get(name) or ""


def load_clients(settings: Settings) -> ClientRegistry:
    """The registry from CLIENTS_FILE, or from API_KEYS without it. ConfigError for an invalid configuration."""
    if not settings.clients_file:
        return ClientRegistry.from_api_keys(settings)
    if settings.api_key_set:
        raise ConfigError("set CLIENTS_FILE or API_KEYS, not both (list the keys' hashes in the clients file)")
    if settings.webhook_secret or settings.webhook_host_set:
        raise ConfigError("with CLIENTS_FILE, webhooks are configured per client: "
                          "remove WEBHOOK_SECRET and WEBHOOK_ALLOWED_HOSTS")
    path = Path(settings.clients_file)
    return registry_from_config(read_clients_file(path), settings, origin=f"CLIENTS_FILE {path}")


def read_clients_file(path: Path) -> dict:
    """The raw content of a clients file (ConfigError if it can't be read or isn't YAML)."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"CLIENTS_FILE {path} can't be read ({type(e).__name__})") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"CLIENTS_FILE {path} is not valid YAML: {e}") from None
    return raw if raw is not None else {}


def registry_from_config(raw: dict, settings: Settings, *, origin: str = "CLIENTS_FILE", source: Source = "file",
                         managed_secret: Callable[[str], str | None] | None = None) -> ClientRegistry:
    """Validate a registry (the CLIENTS_FILE format: profiles + clients) and build it. ConfigError when invalid.
    `managed_secret(client_id)`: the webhook secret generated in the admin dashboard (`managed_secret: true`)."""
    try:
        cfg = RegistryFile.model_validate(raw)
    except ValidationError as e:  # the file holds no secrets, so the details can be shown
        raise ConfigError(f"{origin} is invalid: {e}") from None

    profiles = _load_profiles(cfg.profiles, settings)
    clients, hashes = [], {}
    for client_id, c in cfg.clients.items():
        if not CLIENT_ID.fullmatch(client_id):
            raise ConfigError(f"client id {client_id!r}: use 1-64 of a-z 0-9 . _ - (starting with a letter or digit)")
        unknown = [name for name in c.profiles if name != DEFAULT and name not in profiles]
        if unknown:
            raise ConfigError(f"client {client_id}: profile(s) {', '.join(unknown)} are not defined under profiles:")
        for h in c.key_sha256:
            if h in hashes:
                raise ConfigError(f"an API key hash is listed twice (clients {hashes[h]} and {client_id})")
            hashes[h] = client_id
        secret, hosts, window = "", frozenset(), None
        if c.webhook is not None:
            window = c.webhook.retry_window_s
            if c.webhook.managed_secret:
                secret = managed_secret(client_id) if managed_secret else None
                if not secret:
                    raise ConfigError(f"client {client_id}: managed_secret needs a webhook secret generated in the "
                                      "admin dashboard")
            else:
                secret = _secret_from_env(c.webhook.secret_env)
                if not secret:
                    raise ConfigError(f"client {client_id}: environment variable {c.webhook.secret_env} "
                                      "(its webhook secret) is not set")
            hosts = frozenset(h.strip().lower() for h in c.webhook.allowed_hosts if h.strip())
        clients.append(Client(id=client_id, name=c.name, scopes=frozenset(c.scopes), profiles=tuple(c.profiles),
                              enabled=c.enabled,
                              webhook_secret=secret, webhook_hosts=hosts, webhook_retry_window_s=window,
                              max_queued_jobs=c.limits.max_queued_jobs,
                              extract_concurrent=c.limits.extract_concurrent))
    return ClientRegistry(clients, hashes, source, profiles)


def _load_profiles(raw: dict[str, dict[str, Any]], settings: Settings) -> dict[str, Profile]:
    from app.detection.rules import resolve_rules

    profiles = {}
    for name, spec in raw.items():
        if name == DEFAULT:
            raise ConfigError("profile name 'default' is reserved (it means the global settings)")
        if not CLIENT_ID.fullmatch(name):
            raise ConfigError(f"profile name {name!r}: use 1-64 of a-z 0-9 . _ - (starting with a letter or digit)")
        if "name" in (spec or {}):
            raise ConfigError(f"profile {name}: no 'name' field (the key under profiles: is the name)")
        try:
            profile = Profile.model_validate({**(spec or {}), "name": name})
        except ValidationError as e:
            raise ConfigError(f"profile {name} is invalid: {e}") from None
        if profile.max_pages is not None and profile.max_pages > settings.max_pages:
            raise ConfigError(f"profile {name}: max_pages {profile.max_pages} is above MAX_PAGES "
                              f"({settings.max_pages})")
        if profile.id_patterns:
            try:
                resolve_rules(profile.id_patterns)  # fail at startup, not on the first document
            except Exception as e:  # noqa: BLE001 - missing file, bad YAML, bad regex...
                raise ConfigError(f"profile {name}: id_patterns {profile.id_patterns} can't be loaded "
                                  f"({type(e).__name__})") from None
        profiles[name] = profile
    return profiles
