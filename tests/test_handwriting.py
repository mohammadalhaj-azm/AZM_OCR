"""Experimental handwriting OCR (app/handwriting, POST /v1/handwriting). No model is loaded here: backends are stubs
or fakes; tests/test_live_handwriting.py runs the real model (opt-in)."""

import json
import logging
from contextlib import nullcontext

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import ConfigError
from app.extractors.filetype import PNG, FileRejected
from app.extractors.pdf import render_pdf_pages
from app.handwriting import HandwritingBusy, HandwritingService, HandwritingUnavailable
from app.handwriting.backends import (
    DEFAULT_PROMPT,
    HwOutput,
    Loaded,
    OpenAIBackend,
    TransformersBackend,
    load_transformers,
    model_cached,
)
from app.handwriting.guards import is_blank, repetition_loop
from app.handwriting.service import tidy
from tests import synth
from tests import synth_handwriting as hw
from tests.test_api import ARCHIVE, KEY, OPS, H, StubServices, fake_analyze, k, make_settings, registry  # noqa: F401


class StubBackend:
    """Records what it was asked to read; answers with a fixed output."""
    name = "transformers"

    def __init__(self, text="SECRET_HANDWRITING", finish="stop", error: Exception | None = None):
        self.text, self.finish, self.error = text, finish, error
        self.calls: list[np.ndarray] = []

    def transcribe(self, img):
        self.calls.append(img)
        if self.error:
            raise self.error
        return HwOutput(self.text, 5, 3, self.finish)

    def info(self):
        return {"backend": self.name, "name": "stub-model", "device": "cpu"}

    def status(self):
        return {"backend": self.name, "loaded": True}

    def load(self):
        return None


def service(tmp_path, backend=None, **kw) -> HandwritingService:
    s = make_settings(tmp_path, **{"handwriting_enabled": True, **kw})
    return HandwritingService(s, backend=backend or StubBackend())


@pytest.fixture
def make_hw_client(tmp_path):
    clients = []

    def _make(backend=None, svc=None, **settings_kw):
        s = make_settings(tmp_path, **{"handwriting_enabled": True, **settings_kw})
        svc = svc or HandwritingService(s, backend=backend or StubBackend())
        c = TestClient(create_app(s, services=StubServices(), analyze_fn=fake_analyze, handwriting_service=svc))
        c.__enter__()
        clients.append(c)
        return c, svc

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def post(c, data, name="note.png", headers=H):
    return c.post("/v1/handwriting", files={"file": (name, data, "application/octet-stream")}, headers=headers)


# --- API -------------------------------------------------------------------------------------------------------------

def test_disabled_by_default(tmp_path):
    s = make_settings(tmp_path)
    with TestClient(create_app(s, services=StubServices(), analyze_fn=fake_analyze)) as c:
        r = post(c, hw.png(hw.few_words()))
        assert r.status_code == 503 and r.json()["error"]["code"] == "handwriting_unavailable"
        assert c.get("/health/handwriting").json() == {"enabled": False}


def test_image_result_shape_and_null_confidence(make_hw_client):
    backend = StubBackend("نعم موافق")
    c, _ = make_hw_client(backend)
    r = post(c, hw.png(hw.few_words()), name="form.png")
    assert r.status_code == 200
    out = r.json()
    assert set(out) == {"file", "pages", "fullText", "languages", "timingsMs", "model"}
    assert out["file"] == {"name": "form.png", "mimeType": PNG, "pages": 1}
    assert out["pages"] == [{"page": 1, "text": "نعم موافق", "confidence": None, "ms": out["pages"][0]["ms"],
                             "warnings": []}]
    assert out["fullText"] == "نعم موافق" and out["languages"] == ["ar"]
    assert out["model"] == {"backend": "transformers", "name": "stub-model", "device": "cpu",
                            "maxPixels": 1280 * 28 * 28, "segmenter": "whole_page"}
    assert set(out["timingsMs"]) == {"decode", "model", "total"}
    assert len(backend.calls) == 1 and backend.calls[0].shape == (hw.H, hw.W, 3)  # whole page, BGR


def test_pdf_pages_are_always_rendered(make_hw_client, fixtures):
    backend = StubBackend()
    c, _ = make_hw_client(backend)
    # the PDF has a text layer: its pages are still rendered and sent to the model
    r = post(c, fixtures["letter_invoice_arabic.pdf"], name="a.pdf")
    assert r.status_code == 200 and r.json()["file"]["pages"] == len(backend.calls) >= 1
    assert r.json()["pages"][0]["text"] == "SECRET_HANDWRITING"


def test_multipage_tiff_and_page_cap(make_hw_client, fixtures):
    c, _ = make_hw_client()
    assert [p["page"] for p in post(c, fixtures["two_pages.tiff"]).json()["pages"]] == [1, 2]
    c, _ = make_hw_client(handwriting_max_pages=1)
    r = post(c, fixtures["two_pages.tiff"])
    assert r.status_code == 413 and r.json()["error"]["code"] == "too_many_pages"


def test_docx_is_415(make_hw_client, fixtures):
    backend = StubBackend()
    c, _ = make_hw_client(backend)
    r = post(c, fixtures["delivery_note.docx"], name="a.docx")
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_type" and not backend.calls


def test_busy_is_503_and_extract_still_works(make_hw_client, fixtures):
    c, svc = make_hw_client()
    svc._slot.acquire()  # another handwriting request is running
    try:
        r = post(c, hw.png(hw.few_words()))
        assert r.status_code == 503 and r.json()["error"]["code"] == "busy" and "Retry-After" in r.headers
        assert c.get("/health/handwriting").json()["busy"] is True
        # the other endpoints don't share its limit
        assert c.post("/v1/extract", files={"file": ("a.png", fixtures["id_card.png"])}, headers=H).status_code == 200
    finally:
        svc._slot.release()


def test_extract_and_documents_never_touch_the_handwriting_model(make_hw_client, fixtures):
    backend = StubBackend()
    c, _ = make_hw_client(backend)
    assert c.post("/v1/extract", files={"file": ("a.png", fixtures["id_card.png"])}, headers=H).status_code == 200
    assert c.post("/v1/documents", files={"file": ("a.png", fixtures["id_card.png"])}, headers=H).status_code == 202
    assert backend.calls == []


def test_unavailable_backend_is_503(make_hw_client):
    c, _ = make_hw_client(StubBackend(error=HandwritingUnavailable("the handwriting model is not downloaded")))
    r = post(c, hw.png(hw.few_words()))
    assert r.status_code == 503 and r.json()["error"] == {
        "code": "handwriting_unavailable", "message": "the handwriting model is not downloaded",
        "requestId": r.headers["X-Request-Id"]}


def test_failure_hides_details_and_logs_no_content(make_hw_client, caplog):
    caplog.set_level(logging.DEBUG)
    c, _ = make_hw_client(StubBackend(error=RuntimeError("SECRET_HANDWRITING in exception")))
    r = post(c, hw.png(hw.few_words()), name="SECRET_NAME_HW.png")
    assert r.status_code == 500 and r.json()["error"]["message"] == "processing failed"
    c, _ = make_hw_client()
    assert post(c, hw.png(hw.few_words()), name="SECRET_NAME_HW.png").status_code == 200
    assert "handwriting finished" in caplog.text
    assert "SECRET_HANDWRITING" not in caplog.text and "SECRET_NAME_HW" not in caplog.text


def test_audit_event_has_metadata_only(make_hw_client):
    c, _ = make_hw_client()
    assert post(c, hw.png(hw.few_words()), name="SECRET_NAME_HW.png").status_code == 200
    events = [e for e in c.app.state.store.audit_events() if e["event"] == "handwriting.done"]
    assert len(events) == 1 and events[0]["data"]["pages"] == 1 and events[0]["data"]["endpoint"] == "handwriting"
    assert "SECRET" not in json.dumps(events[0], ensure_ascii=False)


def test_api_key_and_scope(make_hw_client, registry):  # noqa: F811
    c, _ = make_hw_client()
    assert post(c, hw.png(hw.blank()), headers={}).status_code == 401
    c, _ = make_hw_client(**registry())
    assert post(c, hw.png(hw.blank()), headers=k(ARCHIVE)).status_code == 200  # the extract scope
    assert post(c, hw.png(hw.blank()), headers=k(OPS)).status_code == 403  # documents only


def test_invalid_configuration_refuses_to_start(tmp_path):
    with pytest.raises(ConfigError, match="HANDWRITING_BASE_URL"):
        HandwritingService(make_settings(tmp_path, handwriting_enabled=True, handwriting_backend="openai"))
    with pytest.raises(ConfigError, match="HANDWRITING_MAX_PIXELS"):
        HandwritingService(make_settings(tmp_path, handwriting_enabled=True, handwriting_max_pixels=0))
    HandwritingService(make_settings(tmp_path, handwriting_backend="openai"))  # disabled: not checked


def test_openapi_marks_it_experimental_and_keeps_the_existing_contract(make_hw_client):
    spec = make_hw_client()[0].get("/openapi.json").json()
    op = spec["paths"]["/v1/handwriting"]["post"]
    assert op["tags"] == ["experimental"] and "Experimental" in op["summary"]
    schemas = spec["components"]["schemas"]
    page = schemas["HandwritingPage"]["properties"]
    assert page["confidence"]["type"] == "null"
    # the existing result models are untouched
    assert set(schemas["ExtractResult"]["properties"]) == {"file", "pages", "fullText", "normalizedText",
                                                            "languages", "timingsMs", "processing"}
    assert schemas["Page"]["properties"]["method"]["enum"] == ["text_layer", "ocr"]


# --- Guards: no invented text ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("page", [hw.blank, hw.noisy_blank, lambda: hw.noisy_blank(7)],
                         ids=["white", "scanned-noise", "scanned-noise-2"])
def test_blank_pages_are_not_sent_to_the_model(tmp_path, page):
    backend = StubBackend("invented text")
    out = service(tmp_path, backend).transcribe(hw.png(page()), PNG)
    assert backend.calls == []
    assert out["pages"][0]["text"] == "" and out["pages"][0]["skipped"] == "blank" and out["fullText"] == ""
    assert out["pages"][0]["confidence"] is None


@pytest.mark.parametrize("page", [hw.signature, lambda: hw.signature(0.4, 2), hw.stamp, hw.few_words,
                                  lambda: hw.few_words("نعم", 28), lambda: hw.few_words("ب", 20)],
                         ids=["signature", "small-signature", "stamp", "few-words", "one-small-word", "one-letter"])
def test_little_ink_still_goes_to_the_model_and_output_is_unchanged(tmp_path, page):
    backend = StubBackend("  محمد\r\nأحمد ١٢٣ ـ  ")  # whatever the model says is returned as read
    out = service(tmp_path, backend).transcribe(hw.png(page()), PNG)
    assert len(backend.calls) == 1 and out["pages"][0]["skipped"] is None
    assert out["pages"][0]["text"] == "محمد\nأحمد ١٢٣ ـ"  # no normalization: tatweel, digits, hamza kept


def test_blank_check_can_be_turned_off(tmp_path):
    backend = StubBackend("")
    out = service(tmp_path, backend, handwriting_blank_check=False).transcribe(hw.png(hw.blank()), PNG)
    assert len(backend.calls) == 1 and out["pages"][0]["skipped"] is None and out["pages"][0]["text"] == ""


def test_is_blank_threshold(tmp_path):
    assert is_blank(hw.blank(), 0.00001) and is_blank(hw.noisy_blank(), 0.00001)
    assert not is_blank(hw.signature(0.4, 2), 0.00001) and not is_blank(hw.few_words("ب", 20), 0.00001)


def test_warnings_for_loops_and_truncation(tmp_path):
    loop = "بسم الله " + "الرحمن الرحيم " * 8
    out = service(tmp_path, StubBackend(loop, finish="length")).transcribe(hw.png(hw.few_words()), PNG)
    assert out["pages"][0]["warnings"] == ["truncated", "repetition_loop"]
    assert out["pages"][0]["text"] == loop.strip()  # flagged, not cut


@pytest.mark.parametrize("text,loop", [
    ("اسم الطالب: محمد\nالصف: الثالث", False),
    ("الاسم: ..................\nالتوقيع: ..................\nالتاريخ: ..................", False),
    ("the the cat", False),
    ("ok " + "word " * 12, True),
    ("سطر\n" + "نفس السطر\n" * 4, True),
    ("a b c d " * 4 + "a b", True),  # cut off in the middle of the loop
    ("ههههه" * 7, True),
])
def test_repetition_loop(text, loop):
    assert repetition_loop(text) is loop


def test_tidy_touches_only_whitespace():
    assert tidy(" \r\nمُحَمَّد  \r\n\r\nـ٣ ") == "مُحَمَّد\n\nـ٣"


# --- Backends --------------------------------------------------------------------------------------------------------

class FakeInputs(dict):
    def to(self, device):
        self["device"] = device
        return self


class FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append(("chat", messages, tokenize, add_generation_prompt))
        return "CHAT"

    def __call__(self, text, images, padding, return_tensors):
        self.calls.append(("call", text, images))
        return FakeInputs(input_ids=np.zeros((1, 7), np.int64))

    def batch_decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens
        return [" ".join(str(int(t)) for t in ids[0])]


class FakeModel:
    class generation_config:  # noqa: N801 - mimics the attribute
        eos_token_id = [151645, 151643]

    def __init__(self, new_tokens):
        self.new_tokens = new_tokens
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.concatenate([kwargs["input_ids"], np.array([self.new_tokens], np.int64)], axis=1)


def fake_loaded(new_tokens):
    return Loaded(FakeModel(new_tokens), FakeProcessor(), "cpu", "float32", lambda **kw: kw, nullcontext)


def test_transformers_backend_decodes_greedily_without_min_tokens(tmp_path):
    s = make_settings(tmp_path, handwriting_max_new_tokens=4)
    loaded = fake_loaded([11, 12, 151645])
    backend = TransformersBackend(s, loader=lambda _: loaded)
    out = backend.transcribe(hw.few_words())
    gc = loaded.model.kwargs["generation_config"]
    assert gc == {"do_sample": False, "num_beams": 1, "max_new_tokens": 4, "repetition_penalty": 1.1}
    assert "min_new_tokens" not in gc and "temperature" not in gc
    _, messages, tokenize, gen_prompt = loaded.processor.calls[0]
    assert messages == [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DEFAULT_PROMPT}]}]
    assert tokenize is False and gen_prompt is True
    image = loaded.processor.calls[1][2][0]
    assert image.size == (hw.W, hw.H) and image.mode == "RGB"
    assert out.text == "11 12 151645" and out.tokens == 3 and out.finish_reason == "stop"
    assert backend.info() == {"backend": "transformers", "name": s.handwriting_model, "device": "cpu"}


def test_transformers_backend_reports_truncation(tmp_path):
    s = make_settings(tmp_path, handwriting_max_new_tokens=3)
    backend = TransformersBackend(s, loader=lambda _: fake_loaded([11, 12, 13]))
    assert backend.transcribe(hw.few_words()).finish_reason == "length"


def test_transformers_backend_bgr_to_rgb(tmp_path):
    loaded = fake_loaded([151645])
    img = np.zeros((40, 50, 3), np.uint8)
    img[:, :, 0] = 255  # pure blue in BGR
    TransformersBackend(make_settings(tmp_path), loader=lambda _: loaded).transcribe(img)
    assert loaded.processor.calls[1][2][0].getpixel((0, 0)) == (0, 0, 255)


def test_transformers_backend_load_errors(tmp_path):
    def oom(_):
        raise MemoryError("SECRET")

    backend = TransformersBackend(make_settings(tmp_path), loader=oom)
    with pytest.raises(HandwritingUnavailable, match="MemoryError"):
        backend.transcribe(hw.few_words())
    assert backend.status()["error"] == "the handwriting model could not be loaded (MemoryError)"
    assert backend.status()["loaded"] is False


def test_load_transformers_without_the_model(tmp_path):
    s = make_settings(tmp_path, handwriting_model=str(tmp_path / "missing-model"))
    with pytest.raises(HandwritingUnavailable, match="download-handwriting-model|uv sync --extra handwriting"):
        load_transformers(s)


def test_model_cached_needs_every_weight_shard(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir()
    for f in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        (folder / f).write_text("{}")
    assert not model_cached(str(folder))  # no weights at all
    (folder / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}))
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"x")
    assert not model_cached(str(folder))  # one shard missing
    (folder / "model-00002-of-00002.safetensors").write_bytes(b"x")
    assert model_cached(str(folder))
    assert not model_cached("someone/not-a-cached-model-xyz")


def test_openai_backend_request_is_explicit(tmp_path):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "نص"}, "finish_reason": "length"}],
                                         "usage": {"completion_tokens": 9}})

    s = make_settings(tmp_path, handwriting_enabled=True, handwriting_backend="openai",
                      handwriting_base_url="http://gpu:8002/v1", handwriting_api_key="k1", handwriting_model="hw")
    backend = OpenAIBackend(s, transport=httpx.MockTransport(handler))
    out = backend.transcribe(hw.few_words())
    assert out.text == "نص" and out.tokens == 9 and out.finish_reason == "length"
    req = seen[0]
    assert req.url.path == "/v1/chat/completions" and req.headers["Authorization"] == "Bearer k1"
    body = json.loads(req.content)
    assert body["model"] == "hw" and body["temperature"] == 0 and body["top_p"] == 1
    assert body["max_tokens"] == 1024 and body["repetition_penalty"] == 1.1
    image, text = body["messages"][0]["content"]
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert text == {"type": "text", "text": DEFAULT_PROMPT}


def test_openai_backend_errors_and_status(tmp_path):
    def handler(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "hw"}]})
        return httpx.Response(500, text="SECRET")

    s = make_settings(tmp_path, handwriting_backend="openai", handwriting_base_url="http://gpu:8002/v1",
                      handwriting_model="hw")
    backend = OpenAIBackend(s, transport=httpx.MockTransport(handler))
    with pytest.raises(HandwritingUnavailable, match="HTTPStatusError"):
        backend.transcribe(hw.few_words())
    assert backend.status() == {"backend": "openai", "reachable": True, "modelServed": True}


def test_custom_prompt(tmp_path):
    s = make_settings(tmp_path, handwriting_prompt="Read the handwriting.")
    loaded = fake_loaded([151645])
    TransformersBackend(s, loader=lambda _: loaded).transcribe(hw.few_words())
    assert loaded.processor.calls[0][1][0]["content"][1]["text"] == "Read the handwriting."


def test_default_prompt_is_the_model_cards():
    # the fine-tuning prompt, verbatim (its spelling included): changing it changes what the model does
    assert DEFAULT_PROMPT == ("ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار "
                              "ودون ذيادة او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:")


# --- PDF rendering ---------------------------------------------------------------------------------------------------

def test_render_pdf_pages_ignores_the_text_layer():
    images = render_pdf_pages(synth.text_pdf(), dpi=100, max_pages=5)
    assert len(images) >= 1 and images[0].ndim == 3 and images[0].dtype == np.uint8
    with pytest.raises(FileRejected) as e:
        render_pdf_pages(synth.text_pdf(), dpi=100, max_pages=0)
    assert e.value.code == "too_many_pages"
    with pytest.raises(FileRejected) as e:
        render_pdf_pages(b"%PDF-1.4 broken", dpi=100, max_pages=5)
    assert e.value.code == "corrupt"


def test_more_than_one_slot_needs_a_batching_server(tmp_path):
    with pytest.raises(ConfigError, match="HANDWRITING_MAX_CONCURRENT > 1 needs HANDWRITING_BACKEND=openai"):
        service(tmp_path, handwriting_max_concurrent=2)
    with pytest.raises(ConfigError, match="HANDWRITING_MAX_CONCURRENT must be at least 1"):
        service(tmp_path, handwriting_max_concurrent=0)
    service(tmp_path, handwriting_max_concurrent=4, handwriting_backend="openai", handwriting_base_url="http://hw")


def test_slots_let_that_many_pages_through_and_no_more(tmp_path):
    import threading

    gate, inside = threading.Event(), []

    class Slow(StubBackend):
        def transcribe(self, img):
            inside.append(1)
            gate.wait(5)
            return super().transcribe(img)

    svc = service(tmp_path, backend=Slow(), handwriting_max_concurrent=2, handwriting_backend="openai",
                  handwriting_base_url="http://hw", handwriting_blank_check=False)
    page = np.full((60, 60, 3), 255, np.uint8)
    threads = [threading.Thread(target=svc.transcribe_image, args=(page,)) for _ in range(2)]
    for t in threads:
        t.start()
    for _ in range(100):
        if len(inside) == 2:
            break
        threading.Event().wait(0.02)
    assert len(inside) == 2  # two pages at the model at the same time
    st = svc.status()
    assert st["busy"] is True and st["slots"] == {"size": 2, "inUse": 2}
    with pytest.raises(HandwritingBusy):  # a third waits for a slot, then gives up
        svc.transcribe_image(page, wait_s=0.05)
    gate.set()
    for t in threads:
        t.join(5)
    assert svc.status()["slots"] == {"size": 2, "inUse": 0} and svc.status()["busy"] is False
