"""Extract-and-answer end to end in the pipeline (app/answer/service.py via app/pipeline.py), with a scripted
PaddleOCR engine, a stub handwriting backend and a scripted LLM server (tests/fakes_answer.py)."""

import httpx
import pytest

from app.answer import TASK
from app.config import Settings
from app.pipeline import analyze, extract_stage
from app.profiles import Profile, withhold
from tests.fakes_answer import (
    HANDWRITTEN,
    PRINTED,
    UNCLEAR,
    FakeEngine,
    FakeLlmServer,
    StubHw,
    ocr,
    png,
    question_count,
    services,
    text_of,
)

S = Settings(_env_file=None, ocr_warmup=False)


def run(engine, server=None, hw=None, questions=(), mode="auto", correct=True, settings=S, data=None,
        hw_enabled=True, **kw):
    svc = services(settings, engine, server, hw, hw_enabled)
    return analyze(data or png(), "doc.png", svc, task=TASK,
                   answer={"questions": list(questions), "mode": mode, "correct": correct}, **kw)


def test_printed_page_stays_on_paddleocr_and_is_corrected():
    server = FakeLlmServer(page_correction=lambda u: {
        "correctedText": text_of(u).replace("INVOICE No", "INVOICE No.").replace("Customer:", "Customer:\n"),
        "reviewItems": []})
    hw = StubHw()
    out = run(FakeEngine(ocr(PRINTED)), server, hw)
    page = out["pages"][0]
    assert page["engine"] == "paddleocr" and hw.calls == 0
    assert page["route"]["kind"] == "printed" and page["route"]["decidedBy"] == "ocr_signals"
    assert page["route"]["uncertain"] is False and page["route"]["signals"]["read"] == 3
    assert page["text"] == "INVOICE No 4471\nCustomer: Sample Tester\nTotal: 1,250.00 USD"  # the original
    assert page["correctedText"].startswith("INVOICE No. 4471")
    assert {"original": "No", "corrected": "No."} in page["changes"]  # computed by the service
    assert page["confidence"] == pytest.approx(0.98, abs=0.01)
    assert out["correction"] == {"status": "done", "model": "qwen3:8b"}
    assert out["routing"] == {"document": "printed", "pages": {"printed": 1}, "uncertainPages": []}
    assert out["fullText"] == page["text"] and out["correctedText"] == page["correctedText"]
    assert "answers" not in out and server.names() == ["page_correction"]  # no questions: extraction only
    assert out["processing"]["llm"] == "qwen3:8b" and "handwriting" not in out["processing"]
    assert out["processing"]["routing"]["printedMinConf"] == S.routing_printed_min_conf


def test_protected_values_never_reach_the_llm():
    server = FakeLlmServer()
    run(FakeEngine(ocr(PRINTED)), server)
    user = server.calls[0]["body"]["messages"][1]["content"]
    assert "4471" not in user and "1,250.00" not in user and "⟦P1⟧" in user


def test_handwritten_page_goes_to_the_handwriting_model_and_keeps_paddles_reading():
    hw = StubHw("lorem ipsum")
    out = run(FakeEngine(ocr(HANDWRITTEN + [("Ref 4471", 0.95)])), hw=hw)
    page = out["pages"][0]
    assert hw.calls == 1 and page["engine"] == "handwriting" and page["text"] == "lorem ipsum"
    assert page["confidence"] is None  # the handwriting model has no calibrated confidence
    assert [r["engine"] for r in page["readings"]] == ["handwriting", "paddleocr"]
    assert page["route"]["kind"] == "handwritten" and out["routing"]["document"] == "handwritten"
    # PaddleOCR read "4471" confidently; the handwriting reading lacks it: a person should check
    disagreement = [it for it in out["review"] if it["source"] == "readings"]
    assert disagreement and disagreement[0]["quote"] == "4471" and disagreement[0]["category"] == "inconsistent"
    assert out["processing"]["handwriting"] == {"backend": "transformers", "name": "stub-hw", "device": "cpu"}


def test_unclear_page_is_classified_by_the_vision_model():
    server = FakeLlmServer(page_writing={"kind": "mixed", "reasons": ["printed form filled in by hand"]})
    out = run(FakeEngine(ocr(UNCLEAR)), server)
    route = out["pages"][0]["route"]
    assert route["kind"] == "mixed" and route["decidedBy"] == "vision" and route["uncertain"] is False
    assert route["engines"] == ["handwriting", "paddleocr"] and route["reasons"] == ["printed form filled in by hand"]
    assert server.calls[0]["body"]["model"] == S.llm_vision_model and out["processing"]["vision"] == S.llm_vision_model
    # the mixed page's correction gets PaddleOCR's confident (printed) lines as a reference
    assert "REFERENCE" in server.calls[1]["body"]["messages"][1]["content"]


@pytest.mark.parametrize("why,server_kw,settings_kw", [
    ("vision model not served", {"models": ("qwen3:8b",)}, {}),
    ("vision model fails", {"page_writing": httpx.ConnectError("down")}, {}),
    ("vision switched off", {}, {"routing_vision": False}),
])
def test_unclear_page_without_a_vision_verdict_is_uncertain_never_silently_handwritten(why, server_kw, settings_kw):
    seen = []
    svc = services(S.model_copy(update=settings_kw), FakeEngine(ocr(UNCLEAR)), FakeLlmServer(**server_kw))
    svc.on_llm_fallback = seen.append
    out = analyze(png(), "doc.png", svc, task=TASK, answer={})
    page = out["pages"][0]
    assert page["route"]["kind"] == "unknown" and page["route"]["uncertain"] is True, why
    assert page["route"]["decidedBy"] == "fallback" and "fallback: both" in page["route"]["reasons"]
    assert [r["engine"] for r in page["readings"]] == ["handwriting", "paddleocr"]  # both kept
    assert out["routing"]["uncertainPages"] == [1]
    assert any(it["category"] == "routing" and it["page"] == 1 for it in out["review"])
    assert seen == ([] if settings_kw else ["routing_vision"])  # a missing model is reported, a setting is not


def test_uncertain_fallback_printed_keeps_paddleocr_only():
    s = S.model_copy(update={"routing_uncertain_fallback": "printed", "routing_vision": False})
    hw = StubHw()
    out = run(FakeEngine(ocr(UNCLEAR)), hw=hw, settings=s)
    page = out["pages"][0]
    assert page["engine"] == "paddleocr" and hw.calls == 0 and page["route"]["uncertain"] is True


def test_mode_overrides_routing():
    hw = StubHw()
    out = run(FakeEngine(ocr(PRINTED)), hw=hw, mode="handwritten")
    assert out["pages"][0]["engine"] == "handwriting" and out["pages"][0]["route"]["decidedBy"] == "override"
    out = run(FakeEngine(ocr(HANDWRITTEN)), hw=StubHw(), mode="printed")
    assert out["pages"][0]["engine"] == "paddleocr" and out["pages"][0]["route"]["kind"] == "printed"


def test_mode_handwritten_renders_pdf_pages_that_have_a_text_layer(fixtures):
    engine, hw = FakeEngine(ocr(PRINTED)), StubHw()
    auto = run(engine, hw=hw, data=fixtures["letter_invoice_arabic.pdf"])
    assert "text_layer" in [p["engine"] for p in auto["pages"]]
    forced = run(FakeEngine(ocr(PRINTED)), hw=StubHw(), data=fixtures["letter_invoice_arabic.pdf"],
                 mode="handwritten")
    assert {p["engine"] for p in forced["pages"]} == {"handwriting"}


def test_handwriting_model_disabled_falls_back_to_paddleocr_and_says_so():
    out = run(FakeEngine(ocr(HANDWRITTEN)), hw_enabled=False)
    page = out["pages"][0]
    assert page["engine"] == "paddleocr" and page["warnings"] == ["handwriting_unavailable"]
    assert any(it["category"] == "unavailable" and it["source"] == "handwriting" for it in out["review"])


def test_handwriting_warnings_become_review_items():
    out = run(FakeEngine(ocr(HANDWRITTEN)), hw=StubHw("same same same " * 20, finish="length"))
    reasons = [it["reason"] for it in out["review"] if it["source"] == "handwriting"]
    assert any("cut off" in r for r in reasons) and any("repeated" in r for r in reasons)
    assert out["pages"][0]["readings"][0]["warnings"] == ["truncated", "repetition_loop"]


def test_blank_page_is_no_text_and_not_sent_anywhere():
    server, hw = FakeLlmServer(), StubHw()
    out = run(FakeEngine(ocr([], detected=0)), server, hw)
    page = out["pages"][0]
    assert page["route"]["kind"] == "no_text" and page["text"] == "" and hw.calls == 0
    assert server.calls == [] and out["routing"]["document"] == "no_text"


def test_low_confidence_lines_are_marked_for_the_llm_and_listed_for_review():
    lines = PRINTED + [("Signed: Smaple", 0.85)]  # still printed overall, one weak line
    s = S.model_copy(update={"routing_low_line_conf": 0.9, "routing_printed_max_low_share": 0.3})
    server = FakeLlmServer()
    out = run(FakeEngine(ocr(lines)), server, settings=s)
    assert "⟪Signed: Smaple⟫" in server.calls[0]["body"]["messages"][1]["content"]
    item = next(it for it in out["review"] if it["source"] == "ocr_confidence")
    assert item["quote"] == "Signed: Smaple" and item["confidence"] == 0.85 and item["located"] is True
    assert "⟪" not in out["pages"][0]["correctedText"]


@pytest.mark.parametrize("reply", [
    lambda u: {"correctedText": text_of(u).replace("⟦P1⟧", ""), "reviewItems": []},  # dropped a value
    lambda u: {"correctedText": text_of(u) + " 9999", "reviewItems": []},  # invented a number
])
def test_a_broken_protected_value_restores_the_original_and_flags_it(reply):
    out = run(FakeEngine(ocr(PRINTED)), FakeLlmServer(page_correction=reply))
    page = out["pages"][0]
    assert page["correctedText"] == page["text"] and page["changes"] == []
    assert any(it["category"] == "protected_value" for it in out["review"])
    assert out["correction"]["status"] == "done" and out["correctedText"] == page["text"]  # restored, flagged


def test_llm_review_items_are_unmasked_and_checked_against_the_page():
    def reply(u):
        return {"correctedText": text_of(u), "reviewItems": [
            {"quote": "INVOICE No ⟦P1⟧", "suggestion": "INVOICE No. ⟦P1⟧", "reason": "punctuation",
             "category": "possibly_misread"},
            {"quote": "not on the page", "suggestion": "", "reason": "made up", "category": "illegible"}]}
    out = run(FakeEngine(ocr(PRINTED)), FakeLlmServer(page_correction=reply))
    real, invented = [it for it in out["review"] if it["source"] == "llm"]
    assert real["quote"] == "INVOICE No 4471" and real["suggestion"] == "INVOICE No. 4471" and real["located"]
    assert invented["located"] is False and invented["suggestion"] is None


def test_long_pages_are_corrected_in_chunks():
    lines = [(f"Line {chr(65 + i % 26)} " + "word " * 30, 0.97) for i in range(40)]
    server = FakeLlmServer()
    out = run(FakeEngine(ocr(lines)), server, settings=S.model_copy(update={"answer_chunk_chars": 800}))
    assert len(server.calls) > 1 and all(len(text_of(c["body"]["messages"][1]["content"])) <= 800
                                         for c in server.calls)
    assert out["pages"][0]["correctedText"] == out["pages"][0]["text"]


def test_llm_down_never_fails_the_job():
    server = FakeLlmServer(page_correction=httpx.ConnectError("down"), document_qa=httpx.ConnectError("down"))
    seen = []
    svc = services(S, FakeEngine(ocr(PRINTED)), server)
    svc.on_llm_fallback = seen.append
    out = analyze(png(), "doc.png", svc, task=TASK, answer={"questions": ["What is the total?"]})
    assert out["correction"] == {"status": "unavailable", "model": None} and out["correctedText"] is None
    assert out["pages"][0]["correctedText"] is None and out["fullText"]  # the extraction is still there
    assert out["answers"] == [{"question": "What is the total?", "status": "unavailable", "answer": None,
                               "citations": [], "needsReview": True}]
    assert "answer_llm" in seen and "llm" not in out["processing"]


def test_an_output_cut_off_at_max_tokens_is_not_used():
    server = FakeLlmServer()
    server.finish = "length"
    out = run(FakeEngine(ocr(PRINTED)), server)
    assert out["correction"]["status"] == "unavailable" and len(server.calls) == 1  # no pointless retry


def test_correct_false_skips_the_llm():
    server = FakeLlmServer()
    out = run(FakeEngine(ocr(PRINTED)), server, correct=False)
    assert out["correction"] == {"status": "skipped", "model": None} and out["correctedText"] is None
    assert server.calls == []


def qa(answers):
    return lambda u: {"answers": [{"index": i + 1, **a} for i, a in enumerate(answers[:question_count(u)])]}


def test_questions_are_answered_with_verified_citations():
    server = FakeLlmServer(document_qa=qa([
        {"found": True, "answer": "1,250.00 USD", "citations": [{"page": 1, "quote": "Total: 1,250.00 USD"}]},
        {"found": False, "answer": "", "citations": []},
        {"found": True, "answer": "Sample", "citations": [{"page": 1, "quote": "Customer: Someone Else"}]},
    ]))
    out = run(FakeEngine(ocr(PRINTED)), server, questions=["What is the total?", "ما هو رقم الهاتف؟", "Who?"])
    total, missing, wrong = out["answers"]
    assert total == {"question": "What is the total?", "status": "answered", "answer": "1,250.00 USD",
                     "citations": [{"page": 1, "quote": "Total: 1,250.00 USD", "verified": True}],
                     "needsReview": False}
    assert missing["status"] == "not_found" and missing["answer"] == "لا يحتوي المستند على هذه المعلومة."
    assert wrong["citations"][0]["verified"] is False and wrong["needsReview"] is True
    qa_user = server.calls[-1]["body"]["messages"][1]["content"]
    assert "[Page 1]" in qa_user and "1,250.00" in qa_user  # answered from the real (unmasked) text
    assert out["processing"]["llm"] == "qwen3:8b"


def test_an_answer_citing_flagged_text_needs_review():
    lines = PRINTED + [("Signed: Smaple", 0.6)]
    s = S.model_copy(update={"routing_printed_max_low_share": 0.3, "routing_handwritten_max_conf": 0.5})
    server = FakeLlmServer(document_qa=qa([{"found": True, "answer": "Smaple",
                                            "citations": [{"page": 1, "quote": "Smaple"}]}]))
    out = run(FakeEngine(ocr(lines)), server, questions=["Who signed?"], settings=s)
    assert out["answers"][0]["citations"][0]["verified"] and out["answers"][0]["needsReview"] is True


def test_long_documents_are_answered_from_the_first_pages_and_flagged():
    page = [("word " * 100, 0.97)]
    server = FakeLlmServer(document_qa=qa([{"found": False, "answer": "", "citations": []}]))
    out = run(FakeEngine(ocr(page)), server, questions=["Q?"], data=png(3), correct=False,
              settings=S.model_copy(update={"answer_context_chars": 1000}))
    assert out["warnings"] == ["context_truncated"]
    assert out["answers"][0]["needsReview"] is True  # "not found" in the part that was read
    assert server.calls[-1]["body"]["messages"][1]["content"].count("[Page ") == 2


def test_include_boxes_keeps_paddles_lines():
    out = run(FakeEngine(ocr(PRINTED)), include_boxes=True)
    assert out["pages"][0]["lines"][0]["text"] == "INVOICE No 4471" and out["pages"][0]["width"] == 100
    assert "lines" not in run(FakeEngine(ocr(PRINTED)))["pages"][0]


def test_page_limit_is_answer_max_pages(fixtures):
    from app.extractors import FileRejected

    svc = services(S.model_copy(update={"answer_max_pages": 1}), FakeEngine(ocr(PRINTED)))
    with pytest.raises(FileRejected):
        extract_stage(png(2), "a.tiff", svc, task=TASK, answer={})
    with pytest.raises(ValueError):
        extract_stage(png(), "a.png", svc, task="nope")


def test_a_text_free_profile_withholds_everything_that_can_quote_the_document():
    server = FakeLlmServer(page_writing={"kind": "mixed", "reasons": ["quotes the page"]},
                           page_correction=lambda u: {"correctedText": text_of(u), "reviewItems": [
                               {"quote": "Form A-", "suggestion": "", "reason": "x", "category": "ambiguous"}]},
                           document_qa=qa([{"found": True, "answer": "A-17",
                                            "citations": [{"page": 1, "quote": "Form A-17"}]}]))
    out = run(FakeEngine(ocr(UNCLEAR)), server, questions=["Which form?"])
    out = withhold(out, Profile(name="id-only", return_text=False))
    page = out["pages"][0]
    assert page["text"] == "" and page["readings"] is None and page["correctedText"] is None
    assert page["route"]["reasons"] == [] and out["correctedText"] is None
    assert all(it["quote"] is None for it in out["review"])
    assert out["answers"][0]["answer"] is None and out["answers"][0]["citations"] == []


def test_the_default_task_is_unchanged(fixtures):
    """/v1/documents and /v1/extract never route, correct or answer."""
    svc = services(S, FakeEngine(ocr(PRINTED)), FakeLlmServer())
    out = analyze(png(), "a.png", svc, detect_id=False)
    assert set(out) == {"file", "pages", "fullText", "normalizedText", "languages", "timingsMs", "processing"}
    assert "engine" not in out["pages"][0]


def test_chat_json_retries_once_and_counts_calls():
    replies = iter(["not json", '{"kind": "printed", "reasons": []}'])
    server = FakeLlmServer(page_writing=lambda u: next(replies))
    svc = services(S, FakeEngine(ocr(UNCLEAR)), server)
    out = analyze(png(), "doc.png", svc, task=TASK, answer={"correct": False})
    assert out["pages"][0]["route"]["kind"] == "printed" and server.names() == ["page_writing", "page_writing"]
    assert "not valid JSON" in server.calls[1]["body"]["messages"][-1]["content"]
    stats = svc.llm.stats.snapshot()[S.llm_vision_model]
    assert stats["calls"] == 1 and stats["ok"] == 1 and stats["retried"] == 1


def test_id_detection_requests_are_unchanged():
    server = FakeLlmServer(id_detection={"isId": False, "docType": "not_id", "confidence": 0.9, "reasons": ["r"]})
    svc = services(S, FakeEngine(ocr(PRINTED)), server)
    svc.llm.classify_text("some text")
    body = server.calls[0]["body"]
    assert body["max_tokens"] == 400 and body["response_format"]["json_schema"]["name"] == "id_detection"


def test_handwriting_blank_but_paddle_read_text_is_flagged():
    svc = services(S, FakeEngine(ocr(HANDWRITTEN)), FakeLlmServer(), StubHw(), blank_check=True)
    out = analyze(png(), "doc.png", svc, task=TASK, answer={"correct": False})
    assert out["pages"][0]["readings"][0]["warnings"] == ["blank"]
    assert any("found the page blank" in it["reason"] for it in out["review"])
