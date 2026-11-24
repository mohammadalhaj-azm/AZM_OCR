"""Text extraction entry point: detect type -> extract per page."""

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from app.config import Settings
from app.extractors.docx import extract_docx
from app.extractors.filetype import DOCX, IMAGE_TYPES, PDF, FileRejected, validate_upload
from app.extractors.image import decode_images
from app.extractors.pdf import extract_pdf
from app.normalize import clean_text, detect_languages, normalize_for_match
from app.ocr.engine import OcrEngine, OcrOutput
from app.schemas import ExtractionResult, PageResult

__all__ = ["FileRejected", "extract"]


Inspect = Callable[[np.ndarray, OcrOutput], Any]


def extract(data: bytes, settings: Settings, engine: OcrEngine, inspect: Inspect | None = None) -> ExtractionResult:
    """inspect: called with every OCR'd image and its OCR output (on the OCR thread); what it returns is kept on the
    page (`PageResult.inspection`). Used by extract-and-answer routing; None changes nothing."""
    mime = validate_upload(data, settings.max_upload_bytes)
    t_start = time.perf_counter()
    ocr_ms = 0.0
    ocr_pages = 0
    first_ocr_image = None

    lock = threading.Lock()  # ocr_page may run on several threads (parallel PDF pages)

    def remember(img: np.ndarray) -> None:
        """Keep the first OCR'd page image for the vision fallback (called in page order)."""
        nonlocal first_ocr_image
        with lock:
            if first_ocr_image is None:
                first_ocr_image = img

    def ocr_page(img: np.ndarray) -> PageResult:
        nonlocal ocr_ms, ocr_pages
        remember(img)
        t0 = time.perf_counter()
        out = engine.recognize(img)
        with lock:
            ocr_ms += (time.perf_counter() - t0) * 1000
            ocr_pages += 1
        page = PageResult(page=0, method="ocr", text=out.text, confidence=out.confidence,
                          lines=out.lines, width=out.width or None, height=out.height or None)
        if inspect is not None:
            page._inspection = inspect(img, out)
        return page

    if mime == PDF:
        pages = extract_pdf(data, settings, ocr_page, parallel=getattr(engine, "size", 1), on_render=remember)
    elif mime == DOCX:
        pages = extract_docx(data, settings, ocr_page)
    elif mime in IMAGE_TYPES:
        frames = decode_images(data, settings.max_pages)
        pages = [ocr_page(img).model_copy(update={"page": i + 1}) for i, img in enumerate(frames)]
    else:  # validate_upload only returns supported types
        raise FileRejected("unsupported_type", mime)

    t_norm = time.perf_counter()
    # OCR output is already in logical order; only text layers can come out reversed.
    pages = [p.model_copy(update={"text": clean_text(p.text, fix_reversed=p.method == "text_layer"),
                                  "lines": [ln.model_copy(update={"text": clean_text(ln.text)}) for ln in p.lines]
                                  if p.lines else p.lines})
             for p in pages]
    result = ExtractionResult(mime_type=mime, pages=pages)
    result._vision_image = first_ocr_image
    result.normalized_text = normalize_for_match(result.full_text)
    result.languages = detect_languages(result.full_text)
    now = time.perf_counter()
    # "ocr" = summed OCR time of all pages; with OCR_ENGINES > 1 pages overlap, so it can exceed "extract".
    result.timings_ms = {"extract": round((t_norm - t_start) * 1000), "ocr": round(ocr_ms), "ocrPages": ocr_pages,
                         "normalize": round((now - t_norm) * 1000)}
    return result
