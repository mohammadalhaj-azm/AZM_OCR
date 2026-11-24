"""The extract-and-answer LLM calls, over the service's LlmClient (same server, statistics and retry rules).

Each call returns plain, validated data; LlmError when the model is unreachable or its reply is unusable. Nothing
here logs prompts, document text or replies.
"""

from dataclasses import dataclass

import numpy as np

from app.answer import prompts
from app.config import Settings
from app.detection.llm import LlmClient, LlmError, image_data_url

__all__ = ["AnswerLlm", "Correction", "LlmError"]


@dataclass
class Correction:
    text: str  # still masked
    review: list[dict]  # [{quote, suggestion, reason, category}], quotes still masked


def _writing(d: dict) -> tuple[str, list[str]]:
    kind = d["kind"]
    if kind not in ("printed", "handwritten", "mixed", "no_text"):
        raise ValueError("kind")
    return kind, [str(r) for r in d.get("reasons") or []][:3]


def _correction(d: dict) -> Correction:
    text = d["correctedText"]
    if not isinstance(text, str):
        raise TypeError("correctedText")
    items = []
    for it in d.get("reviewItems") or []:
        cat = it.get("category")
        items.append({"quote": str(it["quote"]), "suggestion": str(it.get("suggestion") or "") or None,
                      "reason": str(it.get("reason") or ""),
                      "category": cat if cat in prompts.REVIEW_CATEGORIES else "ambiguous"})
    return Correction(text, items)


class AnswerLlm:
    def __init__(self, llm: LlmClient, settings: Settings):
        self.llm, self.s = llm, settings

    @property
    def text_model(self) -> str:
        return self.s.answer_llm_model or self.s.llm_text_model

    def vision_available(self) -> bool:
        s = self.s
        return (s.routing_vision and s.vision_enabled and s.llm_enabled and self.llm is not None
                and s.llm_vision_model in self.llm.available_models())

    def text_available(self) -> bool:
        return self.s.llm_enabled and self.llm is not None

    def classify_writing(self, img: np.ndarray) -> tuple[str, list[str]]:
        """(printed | handwritten | mixed | no_text, reasons) from the vision model."""
        content = [{"type": "text", "text": "How is the text on this page written?"},
                   {"type": "image_url", "image_url": {"url": image_data_url(img, self.s.vision_max_side)}}]
        return self.llm.chat_json(self.s.llm_vision_model, prompts.WRITING_PROMPT, content, prompts.WRITING_SCHEMA,
                                  "page_writing", self.s.llm_vision_timeout_s, 300, _writing)

    def correct(self, masked: str, engine: str, warnings: list[str], reference: str | None) -> Correction:
        user = prompts.correction_input(masked, engine, warnings, reference)
        return self.llm.chat_json(self.text_model, prompts.CORRECTION_PROMPT, user, prompts.CORRECTION_SCHEMA,
                                  "page_correction", self.s.answer_timeout_s, self.s.answer_max_tokens, _correction)

    def answer(self, pages: list[tuple[int, str]], questions: list[str], truncated: bool) -> list[dict]:
        """[{found, answer, citations: [{page, quote}]}] in the order of `questions`."""
        n = len(questions)

        def parse(d: dict) -> list[dict]:
            got = d["answers"]
            if not isinstance(got, list) or len(got) != n:
                raise ValueError("answers")
            by_index = {a.get("index"): a for a in got if isinstance(a, dict)}
            ordered = [by_index.get(i + 1) for i in range(n)]
            if any(a is None for a in ordered):  # indexes missing or repeated: fall back to the order given
                ordered = got
            return [{"found": bool(a["found"]), "answer": str(a.get("answer") or ""),
                     "citations": [{"page": int(c["page"]), "quote": str(c["quote"])}
                                   for c in a.get("citations") or []]} for a in ordered]

        user = prompts.qa_input(pages, questions, truncated)
        return self.llm.chat_json(self.text_model, prompts.QA_PROMPT, user, prompts.qa_schema(n), "document_qa",
                                  self.s.answer_timeout_s, self.s.answer_max_tokens, parse)
