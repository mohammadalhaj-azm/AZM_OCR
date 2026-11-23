"""Combination logic with a stub LLM (no network)."""

import numpy as np
import pytest

from app.config import Settings
from app.detection.detector import IdDetector
from app.detection.llm import LlmError, Verdict
from app.normalize import clean_text, normalize_for_match
from tests import synth


class StubLlm:
    def __init__(self, text: Verdict | Exception | None = None, image: Verdict | Exception | None = None,
                 models=("qwen3:8b", "qwen3-vl:8b-instruct")):
        self.text, self.image, self.models = text, image, set(models)
        self.text_calls = self.image_calls = 0

    def available_models(self, max_age_s: float = 60):
        return self.models

    def classify_text(self, text):
        self.text_calls += 1
        if isinstance(self.text, Exception):
            raise self.text
        return self.text

    def classify_image(self, img):
        self.image_calls += 1
        if isinstance(self.image, Exception):
            raise self.image
        return self.image


def V(doc_type, conf, reason="r"):
    return Verdict(isId=doc_type != "not_id", docType=doc_type, confidence=conf, reasons=[reason])


S = Settings(_env_file=None)
IMG = np.full((100, 100, 3), 255, np.uint8)
LOW = ["Name: SAMPLE TESTER", "Date of Birth: 01/02/1990"]  # weak rules + unsure LLM -> confidence < 0.7


def detect(lines, llm, image=None, settings=S):
    t = clean_text("\n".join(lines))
    return IdDetector(settings, llm).detect(t, normalize_for_match(t), image)[0]


def test_valid_mrz_decides_without_llm():
    llm = StubLlm(V("not_id", 0.99))
    d = detect(synth.PASSPORT_LINES + synth.passport_mrz(), llm, IMG)
    assert d.is_id and d.doc_type == "passport" and d.confidence == 0.99 and d.decided_by == ["rules"]
    assert llm.text_calls == 0 and llm.image_calls == 0


def test_rules_and_llm_agree():
    d = detect(synth.ID_CARD_LINES, StubLlm(V("national_id", 0.95, "card title")))
    assert d.is_id and d.doc_type == "national_id" and d.decided_by == ["rules", "llm"]
    assert d.confidence > 0.9 and "llm: card title" in d.reasons


def test_llm_overrides_weak_rules_on_hard_negative():
    d = detect(synth.MENTIONS_ID_LINES, StubLlm(V("not_id", 0.9)), IMG)
    assert not d.is_id and d.doc_type == "not_id"
    assert "vision" not in d.decided_by  # confident enough


def test_llm_doc_type_preferred_over_rules_hint():
    d = detect(["PASSPORT", "بطاقة شخصية"], StubLlm(V("passport", 0.9)))
    assert d.doc_type == "passport"


def test_llm_unavailable_falls_back_to_rules_with_capped_confidence():
    d = detect(synth.ID_CARD_LINES, StubLlm(LlmError("down")))
    assert d.is_id and d.decided_by == ["rules"] and d.confidence <= 0.65
    assert "llm: unavailable, using rules only" in d.reasons


def test_low_confidence_with_image_uses_vision():
    llm = StubLlm(V("not_id", 0.55), V("national_id", 0.95, "card layout with photo"))
    d = detect(LOW, llm, IMG)
    assert d.decided_by == ["rules", "llm", "vision"] and llm.image_calls == 1
    assert d.is_id and d.doc_type == "national_id" and "vision: card layout with photo" in d.reasons


def test_vision_not_used_without_image():
    llm = StubLlm(V("not_id", 0.55), V("national_id", 0.95))
    d = detect(LOW, llm, None)
    assert "vision" not in d.decided_by and llm.image_calls == 0


def test_vision_model_missing_is_reported():
    llm = StubLlm(V("not_id", 0.55), models=("qwen3:8b",))
    d = detect(LOW, llm, IMG)
    assert llm.image_calls == 0 and any("not available" in r for r in d.reasons)


def test_vision_disabled_by_config(monkeypatch):
    monkeypatch.setenv("VISION_ENABLED", "false")
    llm = StubLlm(V("not_id", 0.55), V("national_id", 0.95))
    detect(LOW, llm, IMG, settings=Settings(_env_file=None))
    assert llm.image_calls == 0


def test_no_text_image_decided_by_vision_alone():
    llm = StubLlm(V("not_id", 0.9), V("passport", 0.9))
    d = detect([], llm, IMG)
    assert llm.text_calls == 0 and d.decided_by == ["rules", "vision"]
    assert d.is_id and d.doc_type == "passport" and d.confidence == pytest.approx(0.9)
    assert "no text extracted" in d.reasons


# --- LLM skip for long documents without any ID signal -------------------------------------------------

def test_long_document_without_id_signals_skips_llm_and_vision():
    llm = StubLlm(V("national_id", 0.99), V("national_id", 0.99))
    t = clean_text("Quarterly report. Revenue grew in all regions.")
    d, _ = IdDetector(S, llm).detect(t, normalize_for_match(t), IMG, pages=12)
    assert not d.is_id and d.doc_type == "not_id" and d.decided_by == ["rules"]
    assert d.confidence == S.llm_skip_confidence and "LLM skipped" in d.reasons[-1]
    assert llm.text_calls == 0 and llm.image_calls == 0


def test_long_text_without_id_signals_skips_llm():
    llm = StubLlm(V("national_id", 0.99))
    t = clean_text("Lorem ipsum dolor sit amet. " * 400)
    d, _ = IdDetector(S, llm).detect(t, normalize_for_match(t), None, pages=1)
    assert d.doc_type == "not_id" and llm.text_calls == 0


def test_long_document_with_an_id_signal_still_uses_llm():
    llm = StubLlm(V("national_id", 0.95))
    t = clean_text("\n".join(synth.ID_CARD_LINES))
    d, _ = IdDetector(S, llm).detect(t, normalize_for_match(t), None, pages=12)
    assert llm.text_calls == 1 and "llm" in d.decided_by


def test_short_document_never_skips_and_skip_can_be_disabled():
    t = clean_text("Quarterly report.")
    llm = StubLlm(V("not_id", 0.9))
    IdDetector(S, llm).detect(t, normalize_for_match(t), None, pages=1)
    assert llm.text_calls == 1
    off = S.model_copy(update={"llm_skip_min_pages": 0, "llm_skip_min_chars": 0})
    llm = StubLlm(V("not_id", 0.9))
    IdDetector(off, llm).detect(t, normalize_for_match(t), None, pages=50)
    assert llm.text_calls == 1
