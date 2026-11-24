"""Fakes for the extract-and-answer tests: a scripted PaddleOCR engine, a scripted LLM server (the real LlmClient over
httpx.MockTransport) and a stub handwriting backend. No model is loaded."""

import io
import json
import re

import httpx
import numpy as np
from PIL import Image

from app.config import Settings
from app.detection.detector import IdDetector
from app.detection.llm import LlmClient
from app.handwriting import HandwritingService
from app.handwriting.backends import HwOutput
from app.ocr.engine import OcrOutput
from app.ocr.merge import Box
from app.pipeline import Services


def ocr(lines: list[tuple[str, float]], detected: int | None = None) -> OcrOutput:
    """PaddleOCR's output for a page with these (text, confidence) lines."""
    boxes = [Box([[0, i * 10], [100, i * 10], [100, i * 10 + 8], [0, i * 10 + 8]], t, c)
             for i, (t, c) in enumerate(lines)]
    total = sum(len(t) for t, _ in lines)
    conf = round(sum(c * len(t) for t, c in lines) / max(total, 1), 4) if lines else None
    items = [{"text": t, "confidence": c, "bbox": [0.0, i / 10, 1.0, (i + 0.8) / 10]} for i, (t, c) in
             enumerate(lines)]
    return OcrOutput("\n".join(t for t, _ in lines), conf, boxes, items, 100, 100,
                     detected=len(lines) if detected is None else detected)


PRINTED = [("INVOICE No 4471", 0.98), ("Customer: Sample Tester", 0.97), ("Total: 1,250.00 USD", 0.99)]
HANDWRITTEN = [("Iorem ipsm dolr", 0.45), ("sit amet consctetur", 0.40)]
UNCLEAR = [("Form A-17 Applicant", 0.95), ("Date 12/05/2024", 0.93), ("scribble", 0.72), ("more", 0.71)]


class FakeEngine:
    """Returns the scripted outputs in order, one per OCR'd image."""
    size = 1

    def __init__(self, *outputs: OcrOutput):
        self.outputs = list(outputs)
        self.calls = 0

    def recognize(self, img):
        out = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return out

    def load(self):
        return self

    def models_present(self):
        return True

    def stats(self):
        return {}


class StubHw:
    name = "transformers"

    def __init__(self, text="Lorem ipsum 4471", finish="stop", error: Exception | None = None):
        self.text, self.finish, self.error = text, finish, error
        self.calls = 0

    def transcribe(self, img):
        self.calls += 1
        if self.error:
            raise self.error
        return HwOutput(self.text, 5, 3, self.finish)

    def info(self):
        return {"backend": self.name, "name": "stub-hw", "device": "cpu"}

    def status(self):
        return {"backend": self.name, "loaded": True}

    def load(self):
        return None


def echo_correction(user: str) -> dict:
    """Return the text unchanged (marks removed), no review items."""
    return {"correctedText": text_of(user).replace("⟪", "").replace("⟫", ""), "reviewItems": []}


def text_of(user: str) -> str:
    return user.split("Text:\n<<<\n", 1)[1].split("\n>>>", 1)[0]


class FakeLlmServer:
    """Scripted replies per call kind (the JSON schema name): a dict (sent as JSON), a str (sent as is), an
    Exception (raised), or a callable(user_content) returning one of those. `finish` sets finish_reason."""

    def __init__(self, models=("qwen3:8b", "qwen3-vl:8b-instruct"), **replies):
        self.models = list(models)
        self.replies = {"page_correction": echo_correction, **replies}
        self.calls: list[dict] = []
        self.finish = "stop"

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": m} for m in self.models]})
        body = json.loads(request.content)
        name = body["response_format"]["json_schema"]["name"]
        self.calls.append({"name": name, "body": body})
        reply = self.replies.get(name)
        if reply is None:
            raise AssertionError(f"unexpected LLM call {name}")
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(body["messages"][1]["content"])
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{"message": {"content": content},
                                                      "finish_reason": self.finish}]})

    def names(self) -> list[str]:
        return [c["name"] for c in self.calls]


def services(settings: Settings, engine: FakeEngine, server: FakeLlmServer | None = None,
             hw: StubHw | None = None, hw_enabled: bool = True, blank_check: bool = False) -> Services:
    """blank_check: the handwriting service's blank-page guard (off by default: the test pages are white)."""
    server = server or FakeLlmServer()
    llm = LlmClient(settings, transport=httpx.MockTransport(server.handler))
    svc = Services(settings, engine, llm, IdDetector(settings, llm))
    hs = settings.model_copy(update={"handwriting_enabled": hw_enabled, "handwriting_blank_check": blank_check})
    svc.handwriting = HandwritingService(hs, backend=hw or StubHw())
    return svc


def png(n: int = 1, value: int = 255) -> bytes:
    """n white pages as PNG (n == 1) or multi-page TIFF."""
    frames = [Image.fromarray(np.full((200, 160, 3), value, np.uint8)) for _ in range(n)]
    buf = io.BytesIO()
    if n == 1:
        frames[0].save(buf, "PNG")
    else:
        frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def question_count(user: str) -> int:
    return len(re.findall(r"^\d+\. ", user.split("Questions:\n", 1)[1], re.MULTILINE))
