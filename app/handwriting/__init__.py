"""Experimental handwriting OCR (Arabic first, English too): POST /v1/handwriting.

Kept apart from the main OCR pipeline until it has been evaluated (eval/run_handwriting_eval.py). Torch and
transformers are imported only when the in-process model is loaded (pip extra `handwriting`).
"""

from app.handwriting.backends import HandwritingUnavailable
from app.handwriting.service import HandwritingBusy, HandwritingService, UnknownChoice

__all__ = ["HandwritingBusy", "HandwritingService", "HandwritingUnavailable", "UnknownChoice"]
