"""Pipeline end-to-end with real OCR; the LLM is not needed for these cases."""

import pytest

from app.detection.detector import IdDetector
from app.pipeline import Services, analyze
from app.profiles import Profile
from tests.test_detector import StubLlm, V


def services(settings, engine, llm):
    return Services(settings, engine, llm, IdDetector(settings, llm))


def test_passport_image_decided_by_ocr_mrz(settings, engine, fixtures):
    llm = StubLlm(V("not_id", 0.99))
    out = analyze(fixtures["passport.png"], "p.png", services(settings, engine, llm))
    det = out["idDetection"]
    assert det == {**det, "isId": True, "docType": "passport", "decidedBy": ["rules"]}
    assert llm.text_calls == 0  # MRZ read by OCR, check digits valid -> no LLM needed
    assert set(det) == {"isId", "docType", "confidence", "decidedBy", "reasons", "mrz"}
    mrz = det["mrz"]  # read from the OCR'd image
    assert {k: mrz[k] for k in ("format", "documentNumber", "surname", "givenNames", "nationality", "birthDate",
                                "sex", "expiryDate", "expired", "valid")} == {
        "format": "TD3", "documentNumber": "AB1234567", "surname": "TESTER", "givenNames": "SAMPLE",
        "nationality": "UTO", "birthDate": "1990-02-01", "sex": "M", "expiryDate": "2030-01-01", "expired": False,
        "valid": True}
    assert mrz["namesSource"] == "mrz+text"  # OCR dropped letters in the names line; repaired from the VIZ


def test_include_boxes_adds_lines_with_normalized_bboxes(settings, engine, fixtures):
    svc = services(settings, engine, StubLlm(V("national_id", 0.9)))
    out = analyze(fixtures["letter_invoice_arabic.pdf"], "l.pdf", svc, detect_id=False, include_boxes=True)
    text_page, ocr_page = out["pages"][0], out["pages"][1]
    assert "lines" not in text_page and ocr_page["method"] == "ocr"
    assert ocr_page["width"] > 1000 and ocr_page["height"] > 1000  # rendered at OCR_DPI
    lines = ocr_page["lines"]
    assert len(lines) >= 3 and "\n".join(ln["text"] for ln in lines) == ocr_page["text"]
    for ln in lines:
        x0, y0, x1, y1 = ln["bbox"]
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and 0 < ln["confidence"] <= 1
    assert [ln["bbox"][1] for ln in lines] == sorted(ln["bbox"][1] for ln in lines)  # top to bottom
    plain = analyze(fixtures["letter_invoice_arabic.pdf"], "l.pdf", svc, detect_id=False)
    assert "lines" not in plain["pages"][1] and "width" not in plain["pages"][1]


class NoDetector:
    def detect(self, *a, **kw):
        raise AssertionError("text-only extraction must not run ID detection")


def test_text_only_skips_detection(settings, engine, fixtures):
    llm = StubLlm(V("national_id", 0.99))
    svc = Services(settings, engine, llm, NoDetector())
    out = analyze(fixtures["id_card.png"], "c.png", svc, detect_id=False)
    assert set(out) == {"file", "pages", "fullText", "normalizedText", "languages", "timingsMs", "processing"}
    assert out["languages"] == ["ar", "en"] and out["fullText"].strip()
    assert {"extract", "ocr", "total"} <= set(out["timingsMs"]) and "llm" not in out["timingsMs"]
    assert llm.text_calls == 0 and llm.image_calls == 0


def test_invoice_with_llm_stub(settings, engine, fixtures):
    llm = StubLlm(V("not_id", 0.95, "tax invoice"))
    out = analyze(fixtures["invoice.png"], "i.png", services(settings, engine, llm))
    det = out["idDetection"]
    assert not det["isId"] and det["docType"] == "not_id" and det["decidedBy"] == ["rules", "llm"]
    assert out["timingsMs"]["total"] >= out["timingsMs"]["extract"]
    assert out["file"]["mimeType"] == "image/png"


# --- Processing profiles -----------------------------------------------------------------------------------------

def test_profile_can_switch_detection_off(settings, engine, fixtures):
    llm = StubLlm(V("national_id", 0.9))
    out = analyze(fixtures["id_card.png"], "c.png", services(settings, engine, llm),
                  profile={"name": "archive", "detect_id": False})
    assert "idDetection" not in out and llm.text_calls == 0
    assert out["processing"]["profile"] == "archive" and "rules" not in out["processing"]


def test_profile_withholds_the_mrz_but_keeps_the_decision(settings, engine, fixtures):
    out = analyze(fixtures["passport.png"], "p.png", services(settings, engine, StubLlm(V("not_id", 0.9))),
                  profile={"name": "no-mrz", "return_mrz": False})
    assert out["idDetection"]["docType"] == "passport" and "mrz" not in out["idDetection"]
    assert out["processing"]["withheld"] == ["mrz"] and "TESTER" in out["fullText"]  # text is still returned


def test_profile_withholds_text_and_the_models_reasons(settings, engine, fixtures):
    llm = StubLlm(V("national_id", 0.9, "holder name is SAMPLE"))
    out = analyze(fixtures["id_card.png"], "c.png", services(settings, engine, llm), include_boxes=True,
                  profile={"name": "id-only", "return_text": False})
    assert out["fullText"] == out["normalizedText"] == "" and all(p["text"] == "" for p in out["pages"])
    assert all("lines" not in p for p in out["pages"]) and out["languages"] == ["ar", "en"]
    assert out["idDetection"]["isId"] and "llm" in out["idDetection"]["decidedBy"]  # detection used the text
    assert all(r.startswith("rules:") for r in out["idDetection"]["reasons"])
    assert out["processing"]["withheld"] == ["text"] and "SAMPLE" not in str(out)


def test_profile_can_keep_the_llm_out(settings, engine, fixtures):
    llm = StubLlm(V("national_id", 0.9))
    svc = services(settings, engine, llm)
    out = analyze(fixtures["id_card.png"], "c.png", svc, profile={"name": "rules-only", "llm_enabled": False,
                                                                   "vision_enabled": False})
    assert out["idDetection"]["decidedBy"] == ["rules"] and llm.text_calls == 0 and llm.image_calls == 0
    assert "llm" not in out["processing"]
    assert svc.detector_for(Profile(name="other", llm_enabled=False, vision_enabled=False)) is \
        svc.detector_for(Profile(name="rules-only", llm_enabled=False, vision_enabled=False))  # shared detector


def test_profile_rules_file_dpi_and_page_limit(settings, engine, fixtures, tmp_path):
    import hashlib

    from app.extractors import FileRejected

    rules = tmp_path / "civil-rules.yaml"
    rules.write_text("keywords:\n  - {text: \"PASSPORT\", weight: 0.4, doc_type: passport}\n", encoding="utf-8")
    svc = services(settings, engine, StubLlm(V("not_id", 0.95)))
    out = analyze(fixtures["letter_invoice_arabic.pdf"], "l.pdf", svc,
                  profile={"name": "civil", "id_patterns": str(rules), "ocr_dpi": 150})
    p = out["processing"]
    assert p["rules"] == hashlib.sha256(rules.read_bytes()).hexdigest()[:12] != svc.detector.rules_version
    assert p["ocr"]["dpi"] == 150 and out["pages"][1]["method"] == "ocr"
    with pytest.raises(FileRejected):  # 3 pages, profile allows 2
        analyze(fixtures["letter_invoice_arabic.pdf"], "l.pdf", svc, profile={"name": "short", "max_pages": 2})
    with pytest.raises(FileRejected):  # the endpoint's limit still applies below the profile's
        analyze(fixtures["letter_invoice_arabic.pdf"], "l.pdf", svc, max_pages=2,
                profile={"name": "long", "max_pages": 50})
