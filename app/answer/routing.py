"""Which engine reads a page: PaddleOCR (printed text) or the handwriting model (pure functions, no I/O).

Every scanned page or image is OCR'd by PaddleOCR first (it is needed anyway, and fast). Its recognition scores
decide the clear cases; an unclear page goes to the vision model (app/answer/llm.py), and without a verdict from it
the configured fallback applies and the page is marked `uncertain`. A page is never put in a category silently.

The thresholds (ROUTING_* settings) are STARTING VALUES, NOT CALIBRATED: eval/run_routing_eval.py measures them on
labeled pages.

Route record of a page (returned in the result as `pages[].route`):
    kind: printed | handwritten | mixed | no_text | unknown   (unknown: unclear and no vision verdict)
    decidedBy: text_layer | ocr_signals | vision | override | fallback
    uncertain: bool
    engines: the engines whose readings were kept, the primary one first
    signals: PaddleOCR's numbers for the page (null for text-layer pages)
    reasons: short reasons (the vision model's, or ours)
"""

from typing import Literal

from app.config import Settings
from app.ocr.engine import OcrOutput

Decision = Literal["printed", "handwritten", "no_text", "unclear"]
Kind = Literal["printed", "handwritten", "mixed", "no_text", "unknown"]
PADDLE, HANDWRITING, TEXT_LAYER = "paddleocr", "handwriting", "text_layer"


def ocr_signals(out: OcrOutput, ink_ratio: float | None, s: Settings) -> dict:
    """What PaddleOCR's output says about a page. Numbers only (no text)."""
    read = len(out.boxes)
    low = sum(1 for b in out.boxes if b.score < s.routing_low_line_conf)
    return {
        "detected": out.detected,  # text regions found by the detector
        "read": read,  # regions recognized with a score >= OCR_MIN_SCORE
        "meanConfidence": None if out.confidence is None else round(out.confidence, 4),
        "lowConfidenceShare": round(low / read, 4) if read else None,
        "inkRatio": None if ink_ratio is None else round(ink_ratio, 6),
    }


def decide(sig: dict, s: Settings) -> tuple[Decision, str]:
    """(decision, reason) from the signals alone."""
    detected, read, conf, low = sig["detected"], sig["read"], sig["meanConfidence"], sig["lowConfidenceShare"]
    if detected == 0:
        ink = sig.get("inkRatio")
        if ink is not None and ink < s.handwriting_blank_ink_ratio:
            return "no_text", "no ink on the page"
        return "unclear", "PaddleOCR found no text regions, but the page is not blank"
    if read == 0 or read / detected < s.routing_min_kept_share:
        return "handwritten", f"PaddleOCR could read only {read} of {detected} text regions"
    if conf is not None and conf < s.routing_handwritten_max_conf:
        return "handwritten", f"low PaddleOCR confidence ({conf:.2f})"
    if conf is not None and conf >= s.routing_printed_min_conf and (low or 0) <= s.routing_printed_max_low_share:
        return "printed", f"high PaddleOCR confidence ({conf:.2f})"
    return "unclear", "PaddleOCR confidence is neither clearly high nor clearly low (possibly mixed content)"


def engines_for(kind: Kind, decided_by: str, s: Settings, arabic: bool, paddle_has_text: bool) -> list[str]:
    """The engines whose readings are kept, primary first."""
    if kind == "printed":
        if arabic and s.route_printed_arabic == "handwriting":
            return [HANDWRITING, PADDLE]
        return [PADDLE]
    if kind in ("handwritten", "mixed"):
        # PaddleOCR already read the page: its reading is kept as a reference (printed parts of mixed pages)
        return [HANDWRITING, PADDLE] if paddle_has_text else [HANDWRITING]
    if kind == "no_text":
        return [PADDLE]
    fallback = s.routing_uncertain_fallback  # unknown
    if fallback == "printed":
        return [PADDLE]
    if fallback == "handwriting":
        return [HANDWRITING]
    return [HANDWRITING, PADDLE]


def needs_image(decision: Decision, s: Settings, arabic: bool, mode: str) -> bool:
    """Keep the page image for the vision model / the handwriting model?"""
    if mode == "printed":
        return False
    if mode == "handwritten" or decision in ("handwritten", "unclear"):
        return True
    return decision == "printed" and arabic and s.route_printed_arabic == "handwriting"


def document_kind(kinds: list[str]) -> str:
    """printed | handwritten | mixed | no_text | unknown for the whole document."""
    content = {k for k in kinds if k != "no_text"}
    if not content:
        return "no_text"
    if "mixed" in content or {"printed", "handwritten"} <= content:
        return "mixed"
    if content == {"unknown"}:
        return "unknown"
    content.discard("unknown")
    return content.pop() if len(content) == 1 else "mixed"
