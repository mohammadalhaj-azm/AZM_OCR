"""Handwriting OCR of one file: decode / render its pages, skip blank ones, transcribe each region, flag loops.

Separate from the main pipeline (app/pipeline.py): nothing here is used by /v1/extract or /v1/documents.
"""

import logging
import threading
import time

import numpy as np

from app.config import ConfigError, Settings
from app.extractors.filetype import IMAGE_TYPES, PDF, FileRejected
from app.extractors.image import decode_images
from app.extractors.pdf import render_pdf_pages
from app.handwriting.backends import HandwritingBackend, HandwritingUnavailable, make_backend
from app.handwriting.guards import is_blank, repetition_loop
from app.handwriting.regions import Segmenter, make_segmenter
from app.normalize.arabic import detect_languages

log = logging.getLogger(__name__)
ACCEPTED = IMAGE_TYPES | {PDF}


class HandwritingBusy(RuntimeError):
    """Every slot of the model is taken (HANDWRITING_MAX_CONCURRENT pages at a time)."""


class Slots:
    """At most `size` pages at the model at once, across all requests and jobs of this process. A semaphore that also
    says how many slots are taken (health / admin dashboard)."""

    def __init__(self, size: int):
        self.size = max(1, size)
        self._sem = threading.BoundedSemaphore(self.size)
        self._lock = threading.Lock()
        self.in_use = 0

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        got = self._sem.acquire(True, timeout) if blocking else self._sem.acquire(False)
        if got:
            with self._lock:
                self.in_use += 1
        return got

    def release(self) -> None:
        with self._lock:
            self.in_use -= 1
        self._sem.release()

    def full(self) -> bool:
        return self.in_use >= self.size


def config_problems(s: Settings) -> list[str]:
    problems = []
    if s.handwriting_backend == "openai" and not s.handwriting_base_url:
        problems.append("HANDWRITING_BACKEND=openai needs HANDWRITING_BASE_URL")
    for name in ("handwriting_max_pixels", "handwriting_min_pixels", "handwriting_max_side",
                 "handwriting_max_new_tokens", "handwriting_max_pages", "handwriting_dpi", "handwriting_timeout_s"):
        if getattr(s, name) <= 0:
            problems.append(f"{name.upper()} must be positive")
    if s.handwriting_min_pixels > s.handwriting_max_pixels:
        problems.append("HANDWRITING_MIN_PIXELS must not exceed HANDWRITING_MAX_PIXELS")
    if s.handwriting_max_concurrent < 1:
        problems.append("HANDWRITING_MAX_CONCURRENT must be at least 1")
    elif s.handwriting_max_concurrent > 1 and s.handwriting_backend != "openai":
        # the in-process model reads one page at a time; parallel pages need a server that batches them (vLLM)
        problems.append("HANDWRITING_MAX_CONCURRENT > 1 needs HANDWRITING_BACKEND=openai (e.g. vLLM)")
    if s.handwriting_repetition_penalty < 1:
        problems.append("HANDWRITING_REPETITION_PENALTY must be >= 1")
    if not 0 <= s.handwriting_blank_ink_ratio < 1:
        problems.append("HANDWRITING_BLANK_INK_RATIO must be in [0, 1)")
    return problems


def tidy(text: str) -> str:
    """Line endings and surrounding whitespace only. No Unicode or Arabic normalization: the text stays as read."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(ln.rstrip() for ln in lines).strip()


class HandwritingService:
    def __init__(self, settings: Settings, backend: HandwritingBackend | None = None,
                 segmenter: Segmenter | None = None):
        self.s = settings
        self.enabled = settings.handwriting_enabled
        if self.enabled and (problems := config_problems(settings)):
            raise ConfigError("handwriting configuration is invalid: " + "; ".join(problems))
        self.backend = backend or (make_backend(settings) if self.enabled else None)
        self.segmenter = segmenter or make_segmenter(settings.handwriting_segmenter)
        self._slot = Slots(settings.handwriting_max_concurrent)

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if close:
            close()

    def warmup(self) -> None:
        """Load the model now (in a background thread at startup), so the first request doesn't wait for it."""
        try:
            self.backend.load()
        except HandwritingUnavailable as e:
            log.warning("handwriting warmup failed: %s", e)

    def status(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "segmenter": self.segmenter.name, "busy": self._slot.full(),
                "slots": {"size": self._slot.size, "inUse": self._slot.in_use},
                **self.backend.status()}

    def pages(self, data: bytes, mime: str) -> list[np.ndarray]:
        if mime == PDF:
            return render_pdf_pages(data, self.s.handwriting_dpi, self.s.handwriting_max_pages)
        return decode_images(data, self.s.handwriting_max_pages)

    def transcribe(self, data: bytes, mime: str) -> dict:
        """The result dict of POST /v1/handwriting (without `file.name`). Raises HandwritingUnavailable,
        HandwritingBusy (no free slot right now) or FileRejected. Takes one slot for the whole file."""
        if not self.enabled or self.backend is None:
            raise HandwritingUnavailable("handwriting OCR is disabled (HANDWRITING_ENABLED=false)")
        if mime not in ACCEPTED:
            raise FileRejected("unsupported_type", "handwriting OCR accepts PDF, JPG, PNG, WEBP or TIFF")
        if not self._slot.acquire(blocking=False):
            raise HandwritingBusy()
        try:
            t0 = time.perf_counter()
            images = self.pages(data, mime)
            t_decode = time.perf_counter()
            pages = [self._page(i + 1, img) for i, img in enumerate(images)]
            t_end = time.perf_counter()
        finally:
            self._slot.release()
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        info = self.backend.info()
        return {
            "file": {"mimeType": mime, "pages": len(pages)},
            "pages": pages,
            "fullText": full_text,
            "languages": detect_languages(full_text),
            "timingsMs": {"decode": round((t_decode - t0) * 1000), "model": round((t_end - t_decode) * 1000),
                          "total": round((t_end - t0) * 1000)},
            "model": {**info, "maxPixels": self.s.handwriting_max_pixels, "segmenter": self.segmenter.name},
        }

    def transcribe_image(self, img: np.ndarray, wait_s: float | None = None) -> dict:
        """One page image (BGR), for the main pipeline (extract-and-answer jobs): {text, confidence (null), ms,
        skipped, warnings}. Unlike `transcribe`, it waits for a free slot, at most `wait_s` seconds
        (None = HANDWRITING_TIMEOUT_S). Thread-safe: extract-and-answer reads several pages at once.
        Raises HandwritingUnavailable or HandwritingBusy (waited too long)."""
        if not self.enabled or self.backend is None:
            raise HandwritingUnavailable("handwriting OCR is disabled (HANDWRITING_ENABLED=false)")
        if not self._slot.acquire(timeout=self.s.handwriting_timeout_s if wait_s is None else wait_s):
            raise HandwritingBusy()
        try:
            page = self._page(0, img)
        finally:
            self._slot.release()
        page.pop("page")
        return page

    def model_info(self) -> dict:
        """What read the pages (provenance): backend, model name, device."""
        return self.backend.info() if self.backend is not None else {}

    def _page(self, number: int, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        page = {"page": number, "text": "", "confidence": None, "ms": 0, "skipped": None, "warnings": []}
        if self.s.handwriting_blank_check and is_blank(img, self.s.handwriting_blank_ink_ratio):
            page["skipped"] = "blank"
        else:
            texts, warnings = [], []
            for region in self.segmenter.split(img):
                out = self.backend.transcribe(region.image)
                text = tidy(out.text)
                if text:
                    texts.append(text)
                if out.finish_reason == "length":
                    warnings.append("truncated")
                if repetition_loop(text):
                    warnings.append("repetition_loop")
            page["text"] = "\n".join(texts)
            page["warnings"] = list(dict.fromkeys(warnings))
        page["ms"] = round((time.perf_counter() - t0) * 1000)
        return page
