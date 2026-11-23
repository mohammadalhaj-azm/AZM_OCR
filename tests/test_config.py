import base64
import re

import pytest

from app.config import ConfigError, Settings

SAFE_PRODUCTION = dict(
    _env_file=None, environment="production", job_store="postgres",
    database_url="postgresql://ocr:Vq3pT9xLw2@db:5432/ocr",
    result_encryption_key=base64.b64encode(b"k" * 32).decode(),
    api_keys="a" * 40 + "," + "b" * 40,
)


def test_defaults_point_to_local_ollama():
    s = Settings(_env_file=None)
    assert s.llm_base_url.startswith("http://127.0.0.1:")
    assert s.llm_text_model == "qwen3:8b"
    assert s.max_upload_bytes == 20 * 1024 * 1024


def test_env_overrides_backend(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://gpu-server:8000/v1")
    monkeypatch.setenv("LLM_TEXT_MODEL", "Qwen/Qwen3-8B")
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    s = Settings(_env_file=None)
    assert s.llm_base_url == "http://gpu-server:8000/v1"
    assert s.llm_text_model == "Qwen/Qwen3-8B"
    assert s.max_upload_bytes == 5 * 1024 * 1024


def test_api_keys_parsed_from_comma_list(monkeypatch):
    monkeypatch.setenv("API_KEYS", " key-a, key-b ,,")
    assert Settings(_env_file=None).api_key_set == {"key-a", "key-b"}


def test_development_has_no_production_checks():
    Settings(_env_file=None).check_production()  # memory store, no keys: fine for local development


def test_production_accepts_a_safe_configuration():
    s = Settings(**SAFE_PRODUCTION)
    assert s.production_problems() == []
    s.check_production()
    Settings(**{**SAFE_PRODUCTION, "webhook_secret": "w" * 32, "webhook_allowed_hosts": "hooks.internal"}
             ).check_production()


@pytest.mark.parametrize("override,fragment", [
    ({"job_store": "memory"}, "JOB_STORE must be postgres"),
    ({"result_encryption_key": ""}, "RESULT_ENCRYPTION_KEY must be set"),
    ({"api_keys": ""}, "API_KEYS is empty"),
    ({"api_keys": "change-me-1," + "b" * 40}, "API_KEYS: 1 key(s)"),
    ({"api_keys": "Change_Me" + "x" * 40}, "API_KEYS: 1 key(s)"),
    ({"api_keys": "short-key"}, "API_KEYS: 1 key(s)"),
    ({"webhook_secret": "change-me"}, "WEBHOOK_SECRET is shorter"),
    ({"webhook_secret": "too-short"}, "WEBHOOK_SECRET is shorter"),
    ({"webhook_allowed_hosts": "hooks.internal"}, "WEBHOOK_SECRET is empty"),
    ({"database_url": "postgresql://ocr:change-me@db:5432/ocr"}, "DATABASE_URL contains a placeholder"),
    ({"ocr_allow_download": True}, "OCR_ALLOW_DOWNLOAD must be false"),
])
def test_production_rejects_unsafe_settings(override, fragment):
    with pytest.raises(ConfigError, match=re.escape(fragment)):
        Settings(**{**SAFE_PRODUCTION, **override}).check_production()


def test_production_errors_never_contain_secret_values():
    s = Settings(**{**SAFE_PRODUCTION, "api_keys": "zz-change-me-qq", "webhook_secret": "qq-short-zz",
                    "database_url": "postgresql://ocr:yy-change-me-xx@db/ocr"})
    message = "; ".join(s.production_problems())
    assert len(s.production_problems()) == 3
    for secret in ("zz-change-me-qq", "qq-short-zz", "yy-change-me-xx"):
        assert secret not in message


def test_environment_comes_from_env(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert Settings(_env_file=None).environment == "production"
    monkeypatch.setenv("ENVIRONMENT", "staging")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
