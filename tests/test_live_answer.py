"""Live extract-and-answer against the local Ollama (skipped if unreachable): real PaddleOCR and the real text model;
the handwriting model is not needed (the fixtures are printed). Slow on CPU: a few LLM calls of ~15-60 s each.

What it checks is behaviour that must hold for any model output: protected values survive the correction, quotes
and citations are checked, and a question the document can't answer is answered "not found". It is not a
measure of correction quality: that needs real documents and a person.
"""

import pytest

from app.answer import TASK
from app.detection.detector import IdDetector
from app.detection.llm import LlmClient
from app.pipeline import Services, analyze


@pytest.fixture(scope="module")
def llm(settings):
    client = LlmClient(settings)
    if settings.llm_text_model not in client.available_models():
        pytest.skip(f"Ollama not reachable at {settings.llm_base_url} or {settings.llm_text_model} not installed")
    return client


def test_invoice_corrected_and_answered(llm, settings, engine, fixtures):
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    out = analyze(fixtures["invoice.png"], "invoice.png", svc, task=TASK,
                  answer={"questions": ["What is the invoice number?", "What is the customer's passport number?"]})
    page = out["pages"][0]
    assert page["engine"] == "paddleocr" and page["route"]["kind"] == "printed"
    assert out["correction"]["status"] in ("done", "partial")
    digits = [t for t in page["text"].split() if any(ch.isdigit() for ch in t)]
    for token in digits:  # every number of the original is still there (masked, then restored)
        assert token.strip(".,:;") in out["correctedText"], token
    number, passport = out["answers"]
    assert number["status"] == "answered" and number["citations"], number
    assert passport["status"] == "not_found", passport  # the invoice has no passport number
    assert all(isinstance(c["verified"], bool) for c in number["citations"])
