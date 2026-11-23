"""Processing profiles: settings overrides, the job copy, data minimization."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.profiles import DEFAULT_PROFILE, Profile, withhold


def test_default_profile_changes_nothing():
    s = Settings(_env_file=None)
    assert DEFAULT_PROFILE.is_default and DEFAULT_PROFILE.settings_for(s) is s
    assert not Profile(name="same-as-default-but-named").is_default


def test_settings_for_applies_only_what_the_profile_sets():
    s = Settings(_env_file=None)
    t = Profile(name="civil", id_patterns="config/id_patterns.yaml", llm_enabled=False, ocr_dpi=400,
                max_pages=10, min_text_chars_per_page=5).settings_for(s)
    assert t.id_patterns_path == Path("config/id_patterns.yaml") and t.llm_enabled is False
    assert (t.ocr_dpi, t.max_pages, t.min_text_chars_per_page) == (400, 10, 5)
    assert t.vision_enabled == s.vision_enabled and t.llm_base_url == s.llm_base_url  # everything else unchanged
    assert s.ocr_dpi == 300  # the global settings are not modified


@pytest.mark.parametrize("bad", [
    {"database_url": "postgresql://elsewhere"}, {"llm_base_url": "http://elsewhere"}, {"api_keys": "x"},
    {"retrun_text": False},  # a typo is an error, never silently ignored
    {"ocr_dpi": 5000}, {"max_pages": 0}, {"result_ttl_s": -1},
])
def test_unknown_infrastructure_and_out_of_range_settings_are_rejected(bad):
    with pytest.raises(ValidationError):
        Profile(name="p", **bad)


def test_the_job_copy_round_trips():
    p = Profile(name="archive", detect_id=False, include_boxes=True, result_ttl_s=60)
    stored = json.loads(json.dumps(p.to_job()))  # as a job stores it
    assert stored == {"name": "archive", "detect_id": False, "include_boxes": True, "result_ttl_s": 60}
    assert Profile.model_validate(stored) == p


def sample_result() -> dict:
    return {
        "pages": [{"page": 1, "method": "ocr", "text": "SECRET_TEXT", "confidence": 0.9,
                   "lines": [{"text": "SECRET_TEXT", "confidence": 0.9, "bbox": [0, 0, 1, 1]}]}],
        "fullText": "SECRET_TEXT", "normalizedText": "SECRET_TEXT", "languages": ["en"],
        "idDetection": {"isId": True, "docType": "passport", "confidence": 0.99, "decidedBy": ["rules", "llm"],
                        "reasons": ["rules: valid TD3 MRZ (passport, 5/5 check digits)", "llm: holder SECRET_NAME"],
                        "mrz": {"format": "TD3", "surname": "SECRET_SURNAME", "valid": True, "checks": {}}},
        "processing": {"version": "v", "profile": "p"},
    }


def test_withhold_text_and_mrz():
    out = withhold(sample_result(), Profile(name="p", return_text=False, return_mrz=False))
    assert "SECRET" not in json.dumps(out)
    assert out["pages"][0]["text"] == "" and "lines" not in out["pages"][0]
    assert out["fullText"] == out["normalizedText"] == "" and out["languages"] == ["en"]
    assert out["idDetection"]["reasons"] == ["rules: valid TD3 MRZ (passport, 5/5 check digits)"]
    assert out["idDetection"]["docType"] == "passport" and "mrz" not in out["idDetection"]  # the decision stays
    assert out["processing"]["withheld"] == ["text", "mrz"]


def test_withhold_only_marks_what_was_there():
    no_mrz = sample_result()
    del no_mrz["idDetection"]["mrz"]
    assert "withheld" not in withhold(no_mrz, Profile(name="p", return_mrz=False))["processing"]
    assert withhold(sample_result(), DEFAULT_PROFILE) == sample_result()
    text_only = copy.deepcopy(sample_result())
    del text_only["idDetection"]
    assert withhold(text_only, Profile(name="p", return_text=False))["processing"]["withheld"] == ["text"]
