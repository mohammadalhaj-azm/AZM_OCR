from pathlib import Path

import pytest

from app.detection.rules import evaluate, load_rules
from app.normalize import clean_text, normalize_for_match
from tests import synth

RULES = load_rules(Path("config/id_patterns.yaml"))


def run(text: str):
    t = clean_text(text)
    return evaluate(t, normalize_for_match(t), RULES)


def test_id_card_text_scores_high():
    r = run("\n".join(synth.ID_CARD_LINES))
    assert r.score > 0.9 and r.doc_type == "national_id" and not r.decisive
    assert any("بطاقة شخصية" in x for x in r.reasons)
    assert any("IDENTITY CARD" in x for x in r.reasons)


def test_ocr_text_without_spaces_still_matches():
    r = run("الرقمالوطني 01234567890\nبطاقةشخصية")
    assert {"keyword 'الرقم الوطني'", "keyword 'بطاقة شخصية'"} <= set(r.reasons)


def test_alef_and_ta_marbuta_variants_match():
    r = run("بطاقه الهويه\nاقامة")  # no hamza, ه for ة
    assert "keyword 'بطاقة الهوية'" in r.reasons and "keyword 'إقامة'" in r.reasons


def test_valid_mrz_is_decisive():
    r = run("\n".join(synth.PASSPORT_LINES + synth.passport_mrz()))
    assert r.decisive and r.doc_type == "passport" and r.score > 0.99
    assert any("valid TD3 MRZ" in x for x in r.reasons)


def test_broken_mrz_is_only_a_signal():
    mrz = synth.passport_mrz()
    mrz[1] = mrz[1][:19] + str((int(mrz[1][19]) + 1) % 10) + mrz[1][20:]  # wrong birth-date check digit
    r = run("\n".join(mrz))
    # The composite digit covers the birth-date check digit too, so two checks fail (3/5): weak signal only.
    assert not r.decisive and r.mrz.valid_count == 3 and 0.3 <= r.score < 0.9
    assert any("MRZ-like" in x for x in r.reasons)


@pytest.mark.parametrize("lines", [synth.INVOICE_LINES, synth.LETTER_LINES])
def test_non_id_documents_score_low(lines):
    assert run("\n".join(lines)).score < 0.1


def test_letter_mentioning_ids_is_dampened():
    r = run("\n".join(synth.MENTIONS_ID_LINES))
    assert r.negatives and r.score < 0.5  # keywords hit, but letter phrases pull it down


def test_country_id_patterns():
    assert any("AE ID-number" in x for x in run("رقم الهوية 784-1990-1234567-1").reasons)
    assert any("EG ID-number" in x for x in run("الرقم القومي29001011234567").reasons)  # glued to Arabic
    assert not any("EG ID-number" in x for x in run("29013011234567").reasons)  # month 13: invalid


def test_short_keywords_need_word_boundaries():
    assert not run("PRIVATE information").negatives  # contains "VAT"
    assert run("VAT 15%").negatives


def test_empty_text():
    r = run("")
    assert r.score == 0 and r.doc_type is None and not r.signals
