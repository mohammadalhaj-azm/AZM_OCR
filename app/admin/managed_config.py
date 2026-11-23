"""Configuration managed in the admin dashboard: settings overrides, the client registry (clients + profiles) and ID
rule sets, stored as append-only versions (`ocr_config_versions`).

- **Settings**: env vars stay the base; the dashboard stores overrides of an allow-list. `live` settings change on
  every replica within CONFIG_POLL_S; `restart` settings are applied when a replica starts (the dashboard shows them
  as pending). Infrastructure (database, LLM server URLs, keys, secrets, environment) can't be changed here.
- **Registry**: the CLIENTS_FILE format. Once a version exists in the database it replaces CLIENTS_FILE (the first
  dashboard edit starts from the file). Not available with API_KEYS (same rule as CLIENTS_FILE).
- **Rule sets**: rules YAML referenced as `db:<name>` by ID_PATTERNS_PATH or a profile's `id_patterns`.

Every change is validated with the same code the service uses at startup before it is saved, every replica checks
for new versions every CONFIG_POLL_S, and a replica that can't apply a version keeps the previous one and reports it
(alert `config_apply_failed`). A restore saves an old version's content as a new version.
"""

import copy
import logging
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from app.api.clients import ClientRegistry, read_clients_file, registry_from_config
from app.config import ConfigError, Settings
from app.detection.rules import MANAGED_PREFIX, load_rules_text, managed_rules, resolve_rules, set_managed_rules

log = logging.getLogger("app.admin")
SETTINGS, REGISTRY, RULES = "settings", "registry", "rules:"
SECRETS = {"llm_api_key", "api_keys", "database_url", "result_encryption_key", "webhook_secret",
           "admin_session_secret"}

# key: (group, mode, min, max). mode: live (applied at once) or restart. Every other setting is read-only here.
G_OCR, G_MODELS, G_DETECT, G_LIMITS, G_RETENTION, G_WEBHOOKS, G_MONITOR, G_SERVICE = (
    "OCR & languages", "AI models", "ID detection", "Limits & queue", "Retention", "Webhooks",
    "Monitoring & alerts", "Service & infrastructure")
EDITABLE: dict[str, tuple[str, str, float | None, float | None]] = {
    # OCR & languages
    "ocr_dpi": (G_OCR, "live", 72, 600), "min_text_chars_per_page": (G_OCR, "live", 0, 10000),
    "max_pages": (G_OCR, "live", 1, 10000),
    "ocr_det_model": (G_OCR, "restart", None, None), "ocr_rec_model_ar": (G_OCR, "restart", None, None),
    "ocr_rec_model_en": (G_OCR, "restart", None, None), "ocr_det_limit_side_len": (G_OCR, "restart", 16, 4000),
    "ocr_min_score": (G_OCR, "restart", 0, 1), "ocr_max_side": (G_OCR, "restart", 500, 20000),
    "ocr_preprocess": (G_OCR, "restart", None, None), "ocr_engines": (G_OCR, "restart", 1, 16),
    "ocr_cpu_threads": (G_OCR, "restart", 1, 256), "ocr_parallel_rec": (G_OCR, "restart", None, None),
    "ocr_rec_batch_size": (G_OCR, "restart", 1, 128), "ocr_warmup": (G_OCR, "restart", None, None),
    # AI models
    "llm_enabled": (G_MODELS, "live", None, None), "vision_enabled": (G_MODELS, "live", None, None),
    "llm_text_model": (G_MODELS, "live", None, None), "llm_vision_model": (G_MODELS, "live", None, None),
    "llm_timeout_s": (G_MODELS, "live", 1, 3600), "llm_vision_timeout_s": (G_MODELS, "live", 1, 3600),
    "llm_max_chars": (G_MODELS, "live", 500, 200000), "vision_max_side": (G_MODELS, "live", 256, 4096),
    "llm_extra_body": (G_MODELS, "restart", None, None), "llm_worker_threads": (G_MODELS, "restart", 0, 64),
    # ID detection
    "id_patterns_path": (G_DETECT, "live", None, None), "vision_threshold": (G_DETECT, "live", 0, 1),
    "rules_weight": (G_DETECT, "live", 0, 1), "vision_weight": (G_DETECT, "live", 0, 1),
    "llm_skip_min_pages": (G_DETECT, "live", 0, 10000), "llm_skip_min_chars": (G_DETECT, "live", 0, 10 ** 7),
    "llm_skip_confidence": (G_DETECT, "live", 0, 1),
    # Limits & queue
    "max_queued_jobs": (G_LIMITS, "live", 0, 10 ** 6), "extract_max_pages": (G_LIMITS, "live", 1, 10000),
    "retry_after_s": (G_LIMITS, "live", 1, 3600), "max_wait_s": (G_LIMITS, "live", 0, 3600),
    "job_stale_s": (G_LIMITS, "live", 10, 86400), "job_max_attempts": (G_LIMITS, "live", 1, 100),
    "worker_threads": (G_LIMITS, "restart", 1, 64), "extract_max_concurrent": (G_LIMITS, "restart", 1, 64),
    # Retention
    "result_ttl_s": (G_RETENTION, "live", 0, 10 ** 8), "audit_retention_days": (G_RETENTION, "live", 0, 36500),
    "log_retention_days": (G_RETENTION, "live", 1, 3650), "sample_retention_h": (G_RETENTION, "live", 1, 24 * 90),
    # Webhooks
    "webhook_retries": (G_WEBHOOKS, "live", 0, 1000), "webhook_backoff_s": (G_WEBHOOKS, "live", 0, 3600),
    "webhook_max_backoff_s": (G_WEBHOOKS, "live", 1, 86400),
    "webhook_retry_window_s": (G_WEBHOOKS, "live", 0, 30 * 86400),
    "webhook_timeout_s": (G_WEBHOOKS, "restart", 1, 300), "webhook_threads": (G_WEBHOOKS, "restart", 1, 64),
    # Monitoring & alerts
    "alert_cpu_pct": (G_MONITOR, "live", 1, 100), "alert_mem_pct": (G_MONITOR, "live", 1, 100),
    "alert_disk_pct": (G_MONITOR, "live", 1, 100), "alert_gpu_mem_pct": (G_MONITOR, "live", 1, 100),
    "alert_queue_age_s": (G_MONITOR, "live", 10, 86400), "alert_job_stuck_s": (G_MONITOR, "live", 60, 86400 * 7),
    "alert_failure_rate": (G_MONITOR, "live", 0, 1), "alert_for_s": (G_MONITOR, "live", 0, 86400),
    "alert_eval_s": (G_MONITOR, "live", 5, 3600), "sample_persist_s": (G_MONITOR, "live", 2, 3600),
    "admin_session_hours": (G_MONITOR, "live", 0.25, 720),
    "sample_interval_s": (G_MONITOR, "restart", 0.5, 60), "log_buffer_size": (G_MONITOR, "restart", 100, 10 ** 6),
    "config_poll_s": (G_MONITOR, "restart", 1, 600),
}
GROUP_OF_LOCKED = {
    "llm_base_url": G_MODELS, "llm_api_key": G_MODELS, "llm_vision_base_url": G_MODELS,
    "ocr_det_limit_type": G_OCR, "ocr_allow_download": G_OCR, "max_upload_mb": G_LIMITS, "upload_dir": G_LIMITS,
    "webhook_secret": G_WEBHOOKS, "webhook_allowed_hosts": G_WEBHOOKS, "metrics_enabled": G_MONITOR,
    "eval_out_dir": G_MONITOR, "admin_users_file": G_MONITOR, "admin_session_secret": G_MONITOR,
}


class ConfigConflict(Exception):
    """Somebody saved a newer version meanwhile."""


def _json(v: Any) -> Any:
    return str(v) if isinstance(v, Path) else v


def validate_overrides(base: Settings, overrides: dict) -> dict:
    """Coerce and check every override; ConfigError with the reason. Returns the clean overrides."""
    clean = {}
    for key, value in overrides.items():
        if key not in EDITABLE:
            raise ConfigError(f"{key} can't be changed in the dashboard")
        _, _, lo, hi = EDITABLE[key]
        try:
            v = TypeAdapter(Settings.model_fields[key].annotation).validate_python(value)
        except ValidationError as e:
            raise ConfigError(f"{key}: {e.errors()[0]['msg']}") from None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if lo is not None and v < lo or hi is not None and v > hi:
                raise ConfigError(f"{key} must be between {lo} and {hi}")
        if isinstance(v, str) and key.endswith("_model") and key != "ocr_rec_model_en" and not v.strip():
            raise ConfigError(f"{key} can't be empty")
        clean[key] = _json(v)
    if "llm_extra_body" in clean and clean["llm_extra_body"]:
        import json

        try:
            if not isinstance(json.loads(clean["llm_extra_body"]), dict):
                raise ValueError
        except ValueError:
            raise ConfigError("llm_extra_body must be a JSON object") from None
    candidate = base.model_copy(update={k: TypeAdapter(Settings.model_fields[k].annotation).validate_python(v)
                                        for k, v in clean.items()})
    if "id_patterns_path" in clean:
        try:
            resolve_rules(candidate.id_patterns_path)
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"id_patterns_path: {e}") from None
    if candidate.environment == "production" and (problems := candidate.production_problems()):
        raise ConfigError("; ".join(problems))
    return clean


class ManagedConfig:
    def __init__(self, settings: Settings, store, registry: ClientRegistry):
        self.s, self.store, self.registry = settings, store, registry
        self.base = settings.model_copy()  # env values, before any override
        self.applied: dict[str, int] = {}
        self.pending_restart: dict[str, Any] = {}
        self.error: dict | None = None
        self._lock = threading.RLock()
        self.on_change: list = []  # callables(kinds: set[str]) after a change was applied
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- current state -----------------------------------------------------------------------------------------
    def overrides(self) -> dict:
        v = self.store.config_latest(SETTINGS)
        return dict((v or {}).get("content", {}).get("overrides", {}))

    def registry_editable(self) -> tuple[bool, str]:
        if self.s.api_key_set:
            return False, "API_KEYS is set: move to a client registry (CLIENTS_FILE) to manage clients here"
        return True, ""

    def registry_content(self) -> tuple[dict, int | None, str]:
        """(content, version id or None, source) of the current registry, in the CLIENTS_FILE format."""
        v = self.store.config_latest(REGISTRY)
        if v is not None:
            return copy.deepcopy(v["content"]), v["id"], "dashboard"
        if self.s.clients_file:
            return read_clients_file(Path(self.s.clients_file)), None, "file"
        return {"profiles": {}, "clients": {}}, None, "api_keys" if self.s.api_key_set else "none"

    def rule_sets(self) -> dict[str, dict]:
        """{name: {id, at, actor, yaml}} of the dashboard-managed rule sets (deleted ones left out)."""
        out = {}
        for kind in self.store.config_heads():
            if kind.startswith(RULES):
                v = self.store.config_latest(kind)
                if v and not v["content"].get("deleted"):
                    out[kind[len(RULES):]] = {"id": v["id"], "at": v["at"], "actor": v["actor"],
                                              "yaml": v["content"].get("yaml", "")}
        return out

    # --- applying ----------------------------------------------------------------------------------------------
    def build_registry(self, content: dict, version: int | str = "new") -> ClientRegistry:
        secrets = self.store.secrets()
        reg = registry_from_config(content, self.s, origin=f"dashboard registry (version {version})",
                                   source="dashboard", managed_secret=secrets.get)
        if self.s.environment == "production" and (problems := reg.production_problems()):
            raise ConfigError("; ".join(problems))
        return reg

    def _apply_settings(self, overrides: dict, startup: bool) -> set[str]:
        changed = set()
        self.pending_restart = {}
        for key, (_, mode, _, _) in EDITABLE.items():
            want = TypeAdapter(Settings.model_fields[key].annotation).validate_python(
                overrides[key]) if key in overrides else getattr(self.base, key)
            if getattr(self.s, key) == want:
                continue
            if mode == "live" or startup:
                setattr(self.s, key, want)
                changed.add(key)
            else:
                self.pending_restart[key] = _json(want)
        return changed

    def apply(self, startup: bool = False) -> set[str]:
        """Apply the newest versions this process hasn't applied yet. Returns the kinds that changed."""
        with self._lock:
            heads = self.store.config_heads()
            todo = {k for k, v in heads.items() if self.applied.get(k) != v}
            if not todo:
                return set()
            kind, version = "", 0
            try:
                rule_kinds = sorted(k for k in todo if k.startswith(RULES))
                if rule_kinds:
                    kind, version = rule_kinds[0], heads[rule_kinds[0]]
                    sets = {}
                    for k in (k for k in heads if k.startswith(RULES)):
                        v = self.store.config_latest(k)
                        if v and not v["content"].get("deleted"):
                            load_rules_text(v["content"]["yaml"])
                            sets[k[len(RULES):]] = v["content"]["yaml"]
                    set_managed_rules(sets)
                if SETTINGS in todo:
                    kind, version = SETTINGS, heads[SETTINGS]
                    changed = self._apply_settings(self.store.config_latest(SETTINGS)["content"]["overrides"], startup)
                    if changed:
                        log.info("settings applied version=%d changed=%s", version, ",".join(sorted(changed)))
                if REGISTRY in todo:
                    kind, version = REGISTRY, heads[REGISTRY]
                    v = self.store.config_latest(REGISTRY)
                    self.registry.replace_with(self.build_registry(v["content"], v["id"]))
                    log.info("client registry applied version=%d clients=%d", v["id"], len(self.registry.clients))
            except Exception as e:  # noqa: BLE001 - keep the previous configuration
                reason = str(e) if isinstance(e, (ConfigError, ValueError, FileNotFoundError)) else type(e).__name__
                self.error = {"kind": kind.rstrip(":"), "version": version, "error": reason[:500], "at": time.time()}
                log.error("configuration not applied kind=%s version=%s error=%s", kind, version, reason[:200])
                if startup and self.s.environment == "production":
                    raise
                for k in todo:  # don't retry the same broken version every poll
                    self.applied[k] = heads[k]
                return set()
            self.error = None
            self.applied.update(heads)
        for fn in self.on_change:
            try:
                fn(todo)
            except Exception as e:  # noqa: BLE001
                log.error("configuration change hook failed error=%s", type(e).__name__)
        return todo

    # --- saving ------------------------------------------------------------------------------------------------
    def _check_base(self, kind: str, base_version: int | None) -> None:
        if base_version is None:
            return
        head = self.store.config_heads().get(kind)
        if (head or 0) != base_version:
            raise ConfigConflict(kind)

    def save_settings(self, patch: dict, actor: str, comment: str = "", base_version: int | None = None) -> int:
        """`patch`: {key: value} to set, {key: None} to go back to the env value."""
        with self._lock:
            self._check_base(SETTINGS, base_version)
            merged = {**self.overrides(), **patch}
            merged = {k: v for k, v in merged.items() if v is not None}
            clean = validate_overrides(self.base, merged)
            vid = self.store.config_save(SETTINGS, {"overrides": clean}, actor, comment)
        self.apply()
        return vid

    def save_registry(self, content: dict, actor: str, comment: str = "", base_version: int | None = None) -> int:
        ok, why = self.registry_editable()
        if not ok:
            raise ConfigError(why)
        with self._lock:
            self._check_base(REGISTRY, base_version)
            self.build_registry(content)  # ConfigError when invalid
            vid = self.store.config_save(REGISTRY, content, actor, comment)
        self.apply()
        return vid

    def save_rules(self, name: str, text: str, actor: str, comment: str = "", base_version: int | None = None) -> int:
        from app.api.clients import CLIENT_ID

        if not CLIENT_ID.fullmatch(name):
            raise ConfigError("rule set name: use 1-64 of a-z 0-9 . _ - (starting with a letter or digit)")
        load_rules_text(text)  # ValueError when invalid
        with self._lock:
            self._check_base(RULES + name, base_version)
            vid = self.store.config_save(RULES + name, {"yaml": text}, actor, comment)
        self.apply()
        return vid

    def rule_set_users(self, name: str) -> list[str]:
        ref = MANAGED_PREFIX + name
        users = ["settings: id_patterns_path"] if str(self.s.id_patterns_path) == ref else []
        content, _, _ = self.registry_content()
        users += [f"profile {p}" for p, spec in (content.get("profiles") or {}).items()
                  if (spec or {}).get("id_patterns") == ref]
        return users

    def delete_rules(self, name: str, actor: str) -> int:
        if users := self.rule_set_users(name):
            raise ConfigError(f"rule set {name} is still used by: {', '.join(users)}")
        if name not in managed_rules():
            raise ConfigError(f"rule set {name} doesn't exist")
        with self._lock:
            vid = self.store.config_save(RULES + name, {"deleted": True}, actor, "deleted")
        self.apply()
        return vid

    def restore(self, version_id: int, actor: str) -> int:
        v = self.store.config_version(version_id)
        if v is None:
            raise KeyError(version_id)
        comment = f"restored version {version_id}"
        if v["kind"] == SETTINGS:
            with self._lock:
                clean = validate_overrides(self.base, v["content"].get("overrides", {}))
                vid = self.store.config_save(SETTINGS, {"overrides": clean}, actor, comment)
            self.apply()
            return vid
        if v["kind"] == REGISTRY:
            return self.save_registry(v["content"], actor, comment)
        if v["kind"].startswith(RULES):
            name = v["kind"][len(RULES):]
            if v["content"].get("deleted"):
                return self.delete_rules(name, actor)
            return self.save_rules(name, v["content"]["yaml"], actor, comment)
        raise KeyError(version_id)

    # --- watcher -----------------------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="admin-config", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(max(1.0, self.s.config_poll_s)):
            try:
                self.apply()
            except Exception as e:  # noqa: BLE001 - e.g. database briefly down
                log.warning("configuration poll failed error=%s", type(e).__name__)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(5)

    # --- view --------------------------------------------------------------------------------------------------
    def settings_view(self) -> list[dict]:
        overrides = self.overrides()
        out = []
        for key, field in Settings.model_fields.items():
            if key in EDITABLE:
                group, mode, lo, hi = EDITABLE[key]
            else:
                group, mode, lo, hi = GROUP_OF_LOCKED.get(key, G_SERVICE), "locked", None, None
            secret = key in SECRETS
            value = getattr(self.s, key)
            if key in overrides:
                source = "dashboard"
            elif key in self.base.model_fields_set:
                source = "env"
            else:
                source = "default"
            row = {"key": key, "env": key.upper(), "group": group, "mode": mode, "source": source,
                   "type": _type_name(field.annotation), "secret": secret, "min": lo, "max": hi}
            if secret:
                row["value"] = "set" if value else "not set"
            else:
                row["value"] = _json(value)
                row["envValue"] = _json(getattr(self.base, key))
                row["default"] = _json(field.default)
            if key in self.pending_restart:
                row["pending"] = self.pending_restart[key]
            if key in overrides:
                row["override"] = overrides[key]
            out.append(row)
        return out


def _type_name(annotation) -> str:
    name = getattr(annotation, "__name__", None) or str(annotation)
    return {"bool": "bool", "int": "int", "float": "float", "str": "str", "Path": "str"}.get(
        name, "enum" if "Literal" in str(annotation) else ("int?" if "int" in str(annotation) else "str"))
