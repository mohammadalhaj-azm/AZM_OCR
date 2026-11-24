"""Extract-and-answer (experimental, POST /v1/extract-and-answer): routed OCR (PaddleOCR for printed text, the
handwriting model for handwritten and mixed pages), LLM correction with review items, and questions answered from
the document only. See app/answer/service.py."""

from app.answer.service import TASK, AnswerOptions, Routed, extract_routed, finish

__all__ = ["TASK", "AnswerOptions", "Routed", "extract_routed", "finish"]
