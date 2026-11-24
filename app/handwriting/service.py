"""Handwriting OCR of one file: decode / render its pages, skip blank ones, transcribe each region, flag loops.

Several models can be configured side by side (HANDWRITING_MODELS_FILE, app/handwriting/registry.py); a request picks
one by id, or gets the default. One request at a time for all of them: they share the GPU.

Separate from the main pipeline (app/pipeline.py): nothing here is used by /v1/extract or /v1/documents.
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from app.config import ConfigError, Settings
from app.extractors.filetype import IMAGE_TYPES, PDF, FileRejected
from app.extractors.image import decode_images
from app.extractors.pdf import render_pdf_pages
from app.handwriting.backends import HandwritingBackend, HandwritingUnavailable, make_backend
from app.handwriting.guards import is_blank, repetition_loop
from app.handwriting.regions import SEGMENTERS, Segmenter, make_segmenter
from app.handwriting.registry import DEFAULT_ID, load_models
from app.normalize.arabic import detect_languages

log = logging.getLogger(__name__)
ACCEPTED = IMAGE_TYPES | {PDF}


class HandwritingBusy(RuntimeError):
    """The model is already transcribing another request (one at a time)."""


class UnknownChoice(ValueError):
    """A request asked for a model or segmenter that isn't configured. The message is safe to return."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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
    if s.handwriting_repetition_penalty < 1:
        problems.append("HANDWRITING_REPETITION_PENALTY must be >= 1")
    if not 0 <= s.handwriting_blank_ink_ratio < 1:
        problems.append("HANDWRITING_BLANK_INK_RATIO must be in [0, 1)")
    if s.handwriting_max_lines <= 0:
        problems.append("HANDWRITING_MAX_LINES must be positive")
    if s.handwriting_backend == "openai" and (s.handwriting_adapter or s.handwriting_quantization != "none"):
        problems.append("HANDWRITING_ADAPTER and HANDWRITING_QUANTIZATION apply to the transformers backend only "
                        "(the openai backend's server loads the model)")
    return problems


@dataclass
class HwModel:
    id: str
    s: Settings  # the service settings with this model's HANDWRITING_* values
    backend: HandwritingBackend | None


def tidy(text: str) -> str:
    """Line endings and surrounding whitespace only. No Unicode or Arabic normalization: the text stays as read."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(ln.rstrip() for ln in lines).strip()


class HandwritingService:
    def __init__(self, settings: Settings, backend: HandwritingBackend | None = None,
                 segmenter: Segmenter | None = None):
        """`backend` / `segmenter` (tests): used for the default model instead of the configured ones."""
        self.s = settings
        self.enabled = settings.handwriting_enabled
        default, specs = load_models(settings) if self.enabled else (DEFAULT_ID, {DEFAULT_ID: settings})
        if self.enabled:
            problems = [(f"{mid}: " if len(specs) > 1 else "") + p for mid, ms in specs.items()
                        for p in config_problems(ms)]
            if problems:
                raise ConfigError("handwriting configuration is invalid: " + "; ".join(problems))
        self.default = default
        self.models: dict[str, HwModel] = {
            mid: HwModel(mid, ms, (backend if backend is not None and mid == default else
                                   make_backend(ms) if self.enabled else None))
            for mid, ms in specs.items()}
        self.multi = bool(settings.handwriting_models_file)
        self._segmenters: dict[str, Segmenter] = {}
        default_s = self.models[default].s
        self.segmenter = segmenter or self._segmenter(default_s.handwriting_segmenter, default_s)
        self._segmenters.setdefault(self.segmenter.name, self.segmenter)
        self.backend = self.models[default].backend
        self._slot = threading.Lock()

    def _segmenter(self, name: str, s: Settings) -> Segmenter:
        """One instance per kind, shared by all models (the line detector loads once)."""
        if name not in self._segmenters:
            self._segmenters[name] = make_segmenter(name, s)
        return self._segmenters[name]

    def close(self) -> None:
        for m in self.models.values():
            close = getattr(m.backend, "close", None)
            if close:
                close()

    def warmup(self) -> None:
        """Load the default model now (in a background thread at startup), so the first request doesn't wait."""
        try:
            self.backend.load()
        except HandwritingUnavailable as e:
            log.warning("handwriting warmup failed: %s", e)

    def model_info(self, m: HwModel, segmenter: Segmenter) -> dict:
        info = {**m.backend.info(), "maxPixels": m.s.handwriting_max_pixels, "segmenter": segmenter.name}
        if self.multi:
            info["id"] = m.id
        if m.s.handwriting_adapter:
            info["adapter"] = m.s.handwriting_adapter
        if m.s.handwriting_quantization != "none":
            info["quantization"] = m.s.handwriting_quantization
        return info

    def status(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        out = {"enabled": True, "segmenter": self.segmenter.name, "busy": self._slot.locked(),
               **self.backend.status()}
        if self.multi:
            out["default"] = self.default
            out["models"] = {mid: {"segmenter": m.s.handwriting_segmenter, "name": m.s.handwriting_model,
                                   **({"adapter": m.s.handwriting_adapter} if m.s.handwriting_adapter else {}),
                                   **({"quantization": m.s.handwriting_quantization}
                                      if m.s.handwriting_quantization != "none" else {}),
                                   **m.backend.status()}
                             for mid, m in self.models.items()}
        return out

    def pages(self, data: bytes, mime: str) -> list[np.ndarray]:
        if mime == PDF:
            return render_pdf_pages(data, self.s.handwriting_dpi, self.s.handwriting_max_pages)
        return decode_images(data, self.s.handwriting_max_pages)

    def resolve(self, model: str | None = None, segmenter: str | None = None) -> tuple[HwModel, Segmenter]:
        """The model and segmenter a request asked for (None = the defaults). Raises UnknownChoice."""
        mid = model or self.default
        if mid not in self.models:
            raise UnknownChoice("unknown_model", f"unknown handwriting model {mid!r}; configured: "
                                                 f"{', '.join(self.models)}")
        m = self.models[mid]
        if segmenter is not None and segmenter not in SEGMENTERS:
            raise UnknownChoice("invalid_request", f"unknown segmenter {segmenter!r}; one of: "
                                                   f"{', '.join(SEGMENTERS)}")
        name = segmenter or m.s.handwriting_segmenter
        if m.id == self.default and segmenter is None:
            return m, self.segmenter
        return m, self._segmenter(name, m.s)

    def transcribe(self, data: bytes, mime: str, model: str | None = None, segmenter: str | None = None) -> dict:
        """The result dict of POST /v1/handwriting (without `file.name`). `model` / `segmenter`: None = the
        defaults. Raises HandwritingUnavailable, HandwritingBusy, UnknownChoice or FileRejected."""
        if not self.enabled or self.backend is None:
            raise HandwritingUnavailable("handwriting OCR is disabled (HANDWRITING_ENABLED=false)")
        m, seg = self.resolve(model, segmenter)
        if mime not in ACCEPTED:
            raise FileRejected("unsupported_type", "handwriting OCR accepts PDF, JPG, PNG, WEBP or TIFF")
        if not self._slot.acquire(blocking=False):
            raise HandwritingBusy()
        try:
            t0 = time.perf_counter()
            images = self.pages(data, mime)
            t_decode = time.perf_counter()
            pages = [self._page(i + 1, img, m, seg) for i, img in enumerate(images)]
            t_end = time.perf_counter()
        finally:
            self._slot.release()
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        info = self.model_info(m, seg)
        return {
            "file": {"mimeType": mime, "pages": len(pages)},
            "pages": pages,
            "fullText": full_text,
            "languages": detect_languages(full_text),
            "timingsMs": {"decode": round((t_decode - t0) * 1000), "model": round((t_end - t_decode) * 1000),
                          "total": round((t_end - t0) * 1000)},
            "model": info,
        }

    def _page(self, number: int, img: np.ndarray, m: HwModel, segmenter: Segmenter) -> dict:
        t0 = time.perf_counter()
        page = {"page": number, "text": "", "confidence": None, "ms": 0, "skipped": None, "warnings": []}
        if m.s.handwriting_blank_check and is_blank(img, m.s.handwriting_blank_ink_ratio):
            page["skipped"] = "blank"
        else:
            texts, warnings = [], []
            regions = segmenter.split(img)
            if not regions:
                warnings.append("no_lines")  # nothing found: never fall back to the whole page (invented text)
            if len(regions) > m.s.handwriting_max_lines:
                warnings.append("too_many_lines")
                regions = regions[:m.s.handwriting_max_lines]
            for region in regions:
                out = m.backend.transcribe(region.image)
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
