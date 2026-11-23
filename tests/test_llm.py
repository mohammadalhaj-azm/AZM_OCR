"""LLM client against a mocked OpenAI-compatible server (no network)."""

import json
import logging

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.detection.llm import LlmClient, LlmError, parse_verdict, truncate

GOOD = {"isId": True, "docType": "passport", "confidence": 0.93, "reasons": ["has PASSPORT", "has MRZ"]}


def client_with(replies: list, settings: Settings | None = None):
    """replies: content strings (or Exceptions) returned in order. Returns (client, captured request bodies)."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen3:8b"}, {"id": "qwen3-vl:8b-instruct"}]})
        calls.append(json.loads(request.content))
        reply = replies[len(calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    s = settings or Settings(_env_file=None)
    return LlmClient(s, transport=httpx.MockTransport(handler)), calls


def test_valid_reply_and_request_shape():
    llm, calls = client_with([json.dumps(GOOD)])
    v = llm.classify_text("PASSPORT\nP<UTO...")
    assert v.isId and v.docType == "passport" and v.confidence == 0.93
    body = calls[0]
    assert body["model"] == "qwen3:8b" and body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"
    assert body["reasoning_effort"] == "none"  # thinking disabled (from LLM_EXTRA_BODY)
    assert "PASSPORT" in body["messages"][1]["content"]
    assert body["messages"][0]["role"] == "system" and "English" in body["messages"][0]["content"]


def test_invalid_json_retried_once():
    llm, calls = client_with(["sorry, here it is: {isId: yes", json.dumps(GOOD)])
    assert llm.classify_text("x").docType == "passport"
    assert len(calls) == 2
    assert calls[1]["messages"][-1]["role"] == "user" and "not valid JSON" in calls[1]["messages"][-1]["content"]


def test_invalid_twice_raises():
    llm, calls = client_with(["nope", "still nope"])
    with pytest.raises(LlmError):
        llm.classify_text("x")
    assert len(calls) == 2


def test_server_down_raises_llm_error():
    llm, _ = client_with([httpx.ConnectError("refused")])
    with pytest.raises(LlmError):
        llm.classify_text("x")


def test_think_tags_and_fences_stripped():
    v = parse_verdict("<think>hmm</think>\n```json\n" + json.dumps(GOOD) + "\n```")
    assert v is not None and v.docType == "passport"


def test_is_id_made_consistent_with_doc_type():
    v = parse_verdict(json.dumps({**GOOD, "isId": False}))
    assert v.isId is True  # docType=passport wins
    v = parse_verdict(json.dumps({"isId": True, "docType": "not_id", "confidence": 0.8, "reasons": []}))
    assert v.isId is False


def test_out_of_schema_values_rejected():
    assert parse_verdict(json.dumps({**GOOD, "docType": "invoice"})) is None
    assert parse_verdict(json.dumps({**GOOD, "confidence": 1.7})) is None


def test_truncate_keeps_head_and_tail():
    text = "HEAD" + "x" * 10_000 + "P<UTOTAIL"
    t = truncate(text, 1000)
    assert len(t) < 1100 and t.startswith("HEAD") and t.endswith("P<UTOTAIL") and "omitted" in t
    assert truncate("short", 1000) == "short"


def test_vision_request_sends_image():
    llm, calls = client_with([json.dumps(GOOD)])
    llm.classify_image(np.full((2000, 1000, 3), 255, np.uint8))
    content = calls[0]["messages"][1]["content"]
    assert calls[0]["model"] == "qwen3-vl:8b-instruct"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_available_models():
    llm, _ = client_with([])
    assert llm.available_models() == {"qwen3:8b", "qwen3-vl:8b-instruct"}


def test_vllm_style_extra_body(monkeypatch):
    monkeypatch.setenv("LLM_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    llm, calls = client_with([json.dumps(GOOD)], Settings(_env_file=None))
    llm.classify_text("x")
    assert calls[0]["chat_template_kwargs"] == {"enable_thinking": False} and "reasoning_effort" not in calls[0]


def test_separate_vision_server(monkeypatch):
    """vLLM serves one model per server: vision requests go to LLM_VISION_BASE_URL."""
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append((request.url.host, request.url.path))
        if request.url.path.endswith("/models"):
            model = "Qwen/Qwen3-VL-8B-Instruct" if request.url.host == "vision-gpu" else "Qwen/Qwen3-8B"
            return httpx.Response(200, json={"data": [{"id": model}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(GOOD)}}]})

    s = Settings(_env_file=None, llm_base_url="http://text-gpu:8000/v1", llm_text_model="Qwen/Qwen3-8B",
                 llm_vision_base_url="http://vision-gpu:8001/v1", llm_vision_model="Qwen/Qwen3-VL-8B-Instruct")
    llm = LlmClient(s, transport=httpx.MockTransport(handler))
    assert llm.available_models() == {"Qwen/Qwen3-8B", "Qwen/Qwen3-VL-8B-Instruct"}
    llm.classify_text("x")
    llm.classify_image(np.full((50, 50, 3), 255, np.uint8))
    posts = [h for h in hosts if h[1].endswith("/chat/completions")]
    assert posts == [("text-gpu", "/v1/chat/completions"), ("vision-gpu", "/v1/chat/completions")]


def test_document_text_never_logged(caplog):
    caplog.set_level(logging.DEBUG)
    llm, _ = client_with([json.dumps(GOOD)])
    llm.classify_text("SECRET-NAME-12345")
    assert "SECRET-NAME-12345" not in caplog.text and "has PASSPORT" not in caplog.text
