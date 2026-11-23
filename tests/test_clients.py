"""Client registry (CLIENTS_FILE / API_KEYS): loading, validation, authentication, production checks."""

import base64

import pytest
import yaml

from app.api.clients import SCOPES, ClientRegistry, key_hash, load_clients
from app.config import ConfigError, Settings

KEY_A1, KEY_A2, KEY_B = "a1-" + "x" * 40, "a2-" + "y" * 40, "b-" + "z" * 40
SECRET_A = "wa-" + "s" * 40


def write_clients(tmp_path, clients: dict) -> str:
    path = tmp_path / "clients.yaml"
    path.write_text(yaml.safe_dump({"clients": clients}), encoding="utf-8")
    return str(path)


def two_clients() -> dict:
    return {
        "civil-registry": {"name": "Civil Registry", "key_sha256": [key_hash(KEY_A1), key_hash(KEY_A2)],
                           "webhook": {"secret_env": "OCR_TEST_SECRET_A", "allowed_hosts": ["Hooks.Civil.Internal"]},
                           "limits": {"max_queued_jobs": 5, "extract_concurrent": 1}},
        "archive": {"name": "Archive", "key_sha256": [key_hash(KEY_B)], "scopes": ["extract"]},
    }


def settings_for(tmp_path, clients: dict, **kw) -> Settings:
    return Settings(_env_file=None, clients_file=write_clients(tmp_path, clients), **kw)


@pytest.fixture(autouse=True)
def webhook_secret_env(monkeypatch):
    monkeypatch.setenv("OCR_TEST_SECRET_A", SECRET_A)


def test_registry_from_file(tmp_path):
    reg = load_clients(settings_for(tmp_path, two_clients()))
    assert reg.source == "file" and {c.id for c in reg.clients} == {"civil-registry", "archive"}
    a = reg.get("civil-registry")
    assert a.scopes == SCOPES and a.webhook_secret == SECRET_A and a.webhook_hosts == {"hooks.civil.internal"}
    assert a.max_queued_jobs == 5 and a.extract_concurrent == 1
    b = reg.get("archive")
    assert b.scopes == {"extract"} and b.webhook_secret == "" and b.max_queued_jobs is None
    assert SECRET_A not in repr(a)


def test_key_rotation_two_keys_one_client(tmp_path):
    reg = load_clients(settings_for(tmp_path, two_clients()))
    assert reg.authenticate(KEY_A1).id == reg.authenticate(KEY_A2).id == "civil-registry"
    assert reg.authenticate(KEY_B).id == "archive"
    assert reg.authenticate("wrong") is None and reg.authenticate("") is None and reg.authenticate(None) is None


def test_disabled_client_is_not_authenticated(tmp_path):
    clients = two_clients()
    clients["archive"]["enabled"] = False
    reg = load_clients(settings_for(tmp_path, clients))
    assert reg.authenticate(KEY_B) is None and reg.lookup(KEY_B).id == "archive"


def test_api_keys_mode_keeps_the_old_job_owner():
    s = Settings(_env_file=None, api_keys="key-a,key-b", webhook_secret="s", webhook_allowed_hosts="hooks.internal")
    reg = load_clients(s)
    a = reg.authenticate("key-a")
    assert reg.source == "api_keys" and len(reg.clients) == 2
    assert a.id == key_hash("key-a")[:16]  # = sha256(key)[:16], the owner of jobs created before the registry
    assert a.scopes == SCOPES and a.webhook_secret == "s" and a.webhook_hosts == {"hooks.internal"}
    assert reg.production_problems() == []  # API_KEYS is checked by Settings


@pytest.mark.parametrize("change,fragment", [
    (lambda c: c["archive"].update(key_sha256=["not-a-hash"]), "64 lowercase hex"),
    (lambda c: c["archive"].update(key_sha256=[key_hash(KEY_B).upper()]), "64 lowercase hex"),
    (lambda c: c["archive"].update(key_sha256=[]), "key_sha256"),
    (lambda c: c["archive"].update(scopes=["admin"]), "scopes"),
    (lambda c: c["archive"].update(max_queued=3), "max_queued"),  # typo: unknown fields are errors
    (lambda c: c["archive"].update(limits={"max_queued_jobs": -1}), "max_queued_jobs"),
    (lambda c: c["archive"].update(webhook={"secret_env": "lower_case", "allowed_hosts": ["h"]}), "secret_env"),
    (lambda c: c["archive"].update(key_sha256=[key_hash(KEY_A1)]), "listed twice"),
    (lambda c: c.update({"Bad Id": c.pop("archive")}), "client id"),
    (lambda c: c["archive"].update(webhook={"secret_env": "OCR_TEST_SECRET_MISSING", "allowed_hosts": ["h"]}),
     "OCR_TEST_SECRET_MISSING"),
])
def test_invalid_clients_file_is_rejected(tmp_path, change, fragment):
    clients = two_clients()
    change(clients)
    with pytest.raises(ConfigError, match=fragment):
        load_clients(settings_for(tmp_path, clients))


def test_clients_file_problems(tmp_path):
    with pytest.raises(ConfigError, match="can't be read"):
        load_clients(Settings(_env_file=None, clients_file=str(tmp_path / "missing.yaml")))
    (tmp_path / "bad.yaml").write_text("clients: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_clients(Settings(_env_file=None, clients_file=str(tmp_path / "bad.yaml")))
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        load_clients(Settings(_env_file=None, clients_file=str(tmp_path / "empty.yaml")))


@pytest.mark.parametrize("kw,fragment", [
    ({"api_keys": "key-a"}, "not both"),
    ({"webhook_secret": "s"}, "configured per client"),
    ({"webhook_allowed_hosts": "hooks.internal"}, "configured per client"),
])
def test_clients_file_excludes_the_global_settings(tmp_path, kw, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_clients(settings_for(tmp_path, two_clients(), **kw))


def test_production_checks_the_registry(tmp_path, monkeypatch):
    production = dict(environment="production", job_store="postgres", database_url="postgresql://ocr:Vq3pT9@db/ocr",
                      result_encryption_key=base64.b64encode(b"k" * 32).decode())
    s = settings_for(tmp_path, two_clients(), **production)
    reg = load_clients(s)
    assert reg.production_problems() == [] and s.production_problems() == []  # no API_KEYS needed with a file
    s.check_production(reg.production_problems())

    monkeypatch.setenv("OCR_TEST_SECRET_A", "change-me")
    weak = load_clients(s).production_problems()
    assert weak == ["webhook secret of client(s) civil-registry is shorter than 32 characters or a placeholder"]
    with pytest.raises(ConfigError, match="civil-registry"):
        s.check_production(weak)

    clients = two_clients()
    for c in clients.values():
        c["enabled"] = False
    assert "CLIENTS_FILE has no enabled client" in load_clients(settings_for(tmp_path, clients)).production_problems()


def test_webhook_secret_can_come_from_a_dotenv_file(tmp_path, monkeypatch):
    from app.api import clients as clients_module

    monkeypatch.delenv("OCR_TEST_SECRET_A")
    (tmp_path / ".env").write_text(f"OCR_TEST_SECRET_A={SECRET_A}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    clients_module._dotenv.cache_clear()
    try:
        assert load_clients(settings_for(tmp_path, two_clients())).get("civil-registry").webhook_secret == SECRET_A
    finally:
        clients_module._dotenv.cache_clear()


def test_example_clients_file_is_valid(monkeypatch):
    """config/clients.example.yaml must stay loadable (with its secret variable set)."""
    monkeypatch.setenv("WEBHOOK_SECRET_CIVIL_REGISTRY", "w" * 40)
    reg = load_clients(Settings(_env_file=None, clients_file="config/clients.example.yaml"))
    assert isinstance(reg, ClientRegistry) and len(reg.clients) >= 2
    assert {"civil-ids", "id-check-only", "archive-text"} <= set(reg.profiles)
    assert reg.profile_for(reg.get("civil-registry")).name == "civil-ids"


# --- Processing profiles in CLIENTS_FILE ------------------------------------------------------------------------

def write_registry(tmp_path, clients: dict, profiles: dict) -> str:
    path = tmp_path / "clients.yaml"
    path.write_text(yaml.safe_dump({"profiles": profiles, "clients": clients}), encoding="utf-8")
    return str(path)


def with_profiles(tmp_path, profiles: dict | None = None, **kw) -> Settings:
    clients = two_clients()
    clients["civil-registry"]["profiles"] = ["civil", "default"]
    clients["archive"]["profiles"] = ["archive-text"]
    profiles = profiles if profiles is not None else {
        "civil": {"id_patterns": "config/id_patterns.yaml", "result_ttl_s": 86400, "return_mrz": True},
        "archive-text": {"detect_id": False, "include_boxes": True},
    }
    return Settings(_env_file=None, clients_file=write_registry(tmp_path, clients, profiles), **kw)


def test_profiles_are_loaded_and_the_first_is_the_default(tmp_path):
    reg = load_clients(with_profiles(tmp_path))
    civil, archive = reg.get("civil-registry"), reg.get("archive")
    assert civil.profiles == ("civil", "default") and archive.profiles == ("archive-text",)
    assert reg.profile_for(civil).name == "civil" and reg.profile_for(civil).result_ttl_s == 86400
    assert reg.profile_for(civil, "default").is_default
    assert reg.profile_for(archive).detect_id is False and reg.profile_for(archive).include_boxes is True
    assert reg.profile_for(archive, "civil") is None  # not one of archive's profiles
    assert reg.profile_for(civil, "no-such-profile") is None


def test_clients_without_profiles_use_the_default(tmp_path):
    reg = load_clients(settings_for(tmp_path, two_clients()))
    assert reg.get("archive").profiles == ("default",) and reg.profile_for(reg.get("archive")).is_default


def test_api_keys_mode_has_only_the_default_profile():
    reg = load_clients(Settings(_env_file=None, api_keys="key-a"))
    assert reg.profile_for(reg.authenticate("key-a")).is_default and set(reg.profiles) == {"default"}


@pytest.mark.parametrize("profiles,fragment", [
    ({"civil": {"retrun_text": False}, "archive-text": {}}, "profile civil is invalid"),
    ({"civil": {"database_url": "x"}, "archive-text": {}}, "profile civil is invalid"),  # infrastructure: never
    ({"civil": {"max_pages": 500}, "archive-text": {}}, "above MAX_PAGES"),
    ({"civil": {"id_patterns": "config/no-such-rules.yaml"}, "archive-text": {}}, "can't be loaded"),
    ({"civil": {"name": "other"}, "archive-text": {}}, "no 'name' field"),
    ({"civil": {}, "archive-text": {}, "default": {"detect_id": False}}, "reserved"),
    ({"civil": {}}, "archive-text are not defined"),
])
def test_invalid_profiles_fail_the_start(tmp_path, profiles, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_clients(with_profiles(tmp_path, profiles))


def test_a_profile_rules_file_with_a_bad_pattern_fails_the_start(tmp_path):
    rules = tmp_path / "bad-rules.yaml"
    rules.write_text("countries:\n  XX:\n    - {name: broken, pattern: '([', weight: 0.1}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="can't be loaded"):
        load_clients(with_profiles(tmp_path, {"civil": {"id_patterns": str(rules)}, "archive-text": {}}))
