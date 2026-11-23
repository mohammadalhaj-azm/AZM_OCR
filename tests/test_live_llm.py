"""Live tests against the local Ollama (skipped if unreachable). Slow on CPU: ~15-40 s per LLM call.

Vision (~60-90 s per call) runs only with RUN_VISION_TESTS=1.
"""

import os

import pytest

from app.detection.detector import IdDetector
from app.detection.llm import LlmClient
from app.pipeline import Services, analyze
from tests import synth


@pytest.fixture(scope="module")
def llm(settings):
    client = LlmClient(settings)
    if settings.llm_text_model not in client.available_models():
        pytest.skip(f"Ollama not reachable at {settings.llm_base_url} or {settings.llm_text_model} not installed")
    return client


@pytest.mark.parametrize("lines,is_id,doc_type", [
    (synth.ID_CARD_LINES, True, "national_id"),
    (synth.INVOICE_LINES, False, "not_id"),
    (synth.MENTIONS_ID_LINES, False, "not_id"),  # hard negative: only mentions IDs
])
def test_text_classifier(llm, lines, is_id, doc_type):
    v = llm.classify_text("\n".join(lines))
    assert (v.isId, v.docType) == (is_id, doc_type), v.reasons
    assert v.reasons and all(r.isascii() or any(c.isascii() and c.isalpha() for c in r) for r in v.reasons)


def test_pipeline_id_card_image(llm, settings, engine, fixtures):
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    out = analyze(fixtures["id_card.png"], "id_card.png", svc)
    det = out["idDetection"]
    assert det["isId"] and det["docType"] == "national_id" and "llm" in det["decidedBy"]
    assert {"file", "pages", "fullText", "normalizedText", "languages", "idDetection", "timingsMs",
            "processing"} == set(out)
    assert out["processing"]["llm"] == settings.llm_text_model  # the LLM that decided is recorded


@pytest.mark.skipif(os.environ.get("RUN_VISION_TESTS") != "1", reason="set RUN_VISION_TESTS=1 (slow)")
def test_vision_classifier(llm, settings):
    from app.extractors.image import pil_to_bgr

    if settings.llm_vision_model not in llm.available_models():
        pytest.skip("vision model not installed")
    v = llm.classify_image(pil_to_bgr(synth.passport_image()))
    assert v.isId and v.docType == "passport", v.reasons
