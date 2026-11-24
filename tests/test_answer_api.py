"""POST /v1/extract-and-answer over the API: an async job like /v1/documents, with the real pipeline and fakes for
PaddleOCR, the handwriting model and the LLM (tests/fakes_answer.py)."""

import pytest

from app.pipeline import analyze
from tests.fakes_answer import (
    HANDWRITTEN,
    PRINTED,
    FakeEngine,
    FakeLlmServer,
    StubHw,
    ocr,
    png,
    question_count,
    services,
)
from tests.test_api import (  # noqa: F401 - fixtures
    ARCHIVE,
    CIVIL_1,
    KEY,
    OPS,
    H,
    events_of,
    k,
    make_client,
    make_settings,
    recording,
    registry,
    upload,
    wait_done,
    with_profiles,
)

URL = "/v1/extract-and-answer"


def ask(client, data=None, questions=(), headers=H, **form):
    fields = {name: str(v) for name, v in form.items()}
    if questions:
        fields["questions"] = list(questions)  # a repeated form field
    return client.post(URL, files={"file": ("doc.png", data or png(), "application/octet-stream")}, data=fields,
                       headers=headers)


def answers_for(n_found):
    def reply(user):
        return {"answers": [{"index": i + 1, "found": True, "answer": "1,250.00 USD",
                             "citations": [{"page": 1, "quote": "Total: 1,250.00 USD"}]}
                            for i in range(question_count(user))]}
    return reply


@pytest.fixture
def answer_client(make_client, tmp_path):  # noqa: F811 - the fixture from tests.test_api
    def _make(engine=None, server=None, hw=None, analyze_fn=analyze, **settings_kw):
        s = make_settings(tmp_path, **settings_kw)
        svc = services(s, engine or FakeEngine(ocr(PRINTED)), server or FakeLlmServer(document_qa=answers_for(1)),
                       hw or StubHw())
        return make_client(analyze_fn=analyze_fn, services=svc, **settings_kw)
    return _make


def test_job_with_questions(answer_client):
    c, _ = answer_client()
    r = ask(c, questions=["What is the total?"])
    assert r.status_code == 202
    job = wait_done(c, r.json()["jobId"])
    assert job["status"] == "done"
    assert {"pages", "fullText", "correctedText", "routing", "review", "correction", "answers", "processing",
            "timingsMs"} <= set(job)
    assert "idDetection" not in job  # this task does not detect IDs
    assert job["answers"][0]["status"] == "answered" and job["answers"][0]["citations"][0]["verified"] is True
    page = job["pages"][0]
    assert page["engine"] == "paddleocr" and page["route"]["kind"] == "printed" and page["readings"]


def test_extraction_only_without_questions(answer_client):
    c, _ = answer_client()
    job = wait_done(c, ask(c).json()["jobId"])
    assert job["status"] == "done" and "answers" not in job and job["correction"]["status"] == "done"


def test_typed_models_return_the_answer_json_unchanged(answer_client):
    """The response models must not drop, rename, null or add a field (pages union, correctedText null)."""
    rec = recording(analyze)
    c, _ = answer_client(analyze_fn=rec, engine=FakeEngine(ocr(HANDWRITTEN + [("Ref 4471", 0.95)])))
    job = wait_done(c, ask(c, questions=["Q?"], includeBoxes="true").json()["jobId"])
    assert {k: v for k, v in job.items() if k not in ("jobId", "status")} == rec.seen[-1]
    job = wait_done(c, ask(c, correct="false").json()["jobId"])
    assert job["correctedText"] is None and "correctedText" in job  # null, not absent
    assert {k: v for k, v in job.items() if k not in ("jobId", "status")} == rec.seen[-1]


def test_documents_results_keep_their_shape(answer_client):
    c, _ = answer_client(server=FakeLlmServer(id_detection={"isId": False, "docType": "not_id", "confidence": 0.9,
                                                            "reasons": ["an invoice"]}))
    job = wait_done(c, upload(c, png()).json()["jobId"])
    assert job["status"] == "done"
    assert not {"correctedText", "routing", "review", "correction", "answers"} & set(job)
    assert "engine" not in job["pages"][0]


@pytest.mark.parametrize("form,questions,message", [
    ({"mode": "fast"}, (), "mode must be one of"),
    ({}, ["x" * 501], "at most 500 characters"),
    ({}, [f"q{i}" for i in range(11)], "at most 10 questions"),
])
def test_invalid_requests_are_422(answer_client, form, questions, message):
    c, _ = answer_client()
    r = ask(c, questions=questions, **form)
    assert r.status_code == 422 and message in r.json()["error"]["message"]


def test_disabled_is_503(answer_client):
    c, _ = answer_client(answer_enabled=False)
    r = ask(c)
    assert r.status_code == 503 and r.json()["error"]["code"] == "answer_unavailable"


def test_questions_and_options_reach_the_job(answer_client):
    c, _ = answer_client()
    job_id = ask(c, questions=["  What is the total?  ", ""], mode="printed", correct="false").json()["jobId"]
    job = c.app.state.store.get(job_id)
    assert job.options == {"task": "extract_and_answer",
                           "answer": {"questions": ["What is the total?"], "mode": "printed", "correct": False}}
    received = next(e for e in events_of(c, job_id=job_id) if e["event"] == "job.received")
    assert received["data"]["endpoint"] == "extract-and-answer" and received["data"]["questions"] == 1
    assert "What is the total" not in str(received)  # counts only, never the questions
    wait_done(c, job_id)
    done = next(e for e in events_of(c, job_id=job_id) if e["event"] == "job.done")
    assert done["data"]["routing"]["document"] == "printed" and done["data"]["answers"] == {"answered": 1}


def test_idempotency_covers_the_questions(answer_client):
    c, _ = answer_client()
    hk = {**H, "Idempotency-Key": "order-7"}
    first = ask(c, questions=["Q1?"], headers=hk).json()["jobId"]
    again = ask(c, questions=["Q1?"], headers=hk)
    assert again.status_code == 202 and again.json()["jobId"] == first
    assert again.headers["Idempotent-Replayed"] == "true"
    other = ask(c, questions=["Q2?"], headers=hk)
    assert other.status_code == 409 and other.json()["error"]["code"] == "idempotency_conflict"


def test_scope_and_profiles(answer_client, registry):  # noqa: F811
    c, _ = answer_client(**with_profiles(registry))
    assert ask(c, headers=k(ARCHIVE)).status_code == 403  # extract scope only
    assert ask(c, headers=k(OPS), profile="id-only").status_code == 403  # a text-free profile
    r = ask(c, headers=k(CIVIL_1), profile="archive-text")
    assert r.status_code == 202
    job = wait_done(c, r.json()["jobId"], headers=k(CIVIL_1))
    assert job["processing"]["profile"] == "archive-text" and "lines" in job["pages"][0]  # its include_boxes


def test_openapi_lists_the_endpoint_as_experimental(answer_client):
    c, _ = answer_client()
    spec = c.get("/openapi.json").json()
    op = spec["paths"][URL]["post"]
    assert op["tags"] == ["experimental"] and "202" in op["responses"]
    schemas = spec["components"]["schemas"]
    assert {"ReviewItem", "Answer", "Route"} <= set(schemas)
    page = schemas["Page"]["properties"]
    assert page["method"]["enum"] == ["text_layer", "ocr"]  # unchanged
    assert {"engine", "route", "readings", "correctedText"} <= set(page)  # additive, optional
    assert "items" in schemas["JobView"]["properties"]["pages"]["anyOf"][0]  # still a plain list of Page
    assert schemas["JobView"]["properties"]["pages"]["anyOf"][0]["items"] == {"$ref": "#/components/schemas/Page"}
