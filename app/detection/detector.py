"""Combine rules, text LLM and (optional) vision LLM into one ID decision."""

import time
from collections.abc import Callable
from typing import Literal

import numpy as np

from app.config import Settings
from app.detection.llm import DocType, LlmClient, LlmError, Verdict
from app.detection.mrz import parse_fields
from app.detection.rules import RuleResult, evaluate, resolve_rules
from app.schemas import CamelModel

RULES_ONLY_MAX_CONFIDENCE = 0.65  # the rule score isn't calibrated; never sound sure on rules alone
MRZ_CONFIDENCE = 0.99


class IdDetection(CamelModel):
    is_id: bool
    doc_type: DocType
    confidence: float
    decided_by: list[Literal["rules", "llm", "vision"]]
    reasons: list[str]
    # Parsed MRZ fields (number, names, dates, expired, checks) when an MRZ with at most one failed check is found.
    mrz: dict | None = None


def _p_id(v: Verdict) -> float:
    """Verdict -> probability that the document is an ID."""
    return v.confidence if v.isId else 1 - v.confidence


class IdDetector:
    def __init__(self, settings: Settings, llm: LlmClient | None, on_fallback: Callable[[str], None] | None = None):
        self.s = settings
        self.llm = llm
        # Called with "llm" or "vision" when detection goes on without a model that failed or is missing (metrics).
        self.on_fallback = on_fallback
        # Which rules decided (for `processing.rules` and the audit trail): the start of their SHA-256. A file, or a
        # rule set managed in the admin dashboard ("db:<name>").
        self.rules, self.rules_version = resolve_rules(settings.id_patterns_path)

    def _fallback(self, stage: str) -> None:
        if self.on_fallback is not None:
            try:
                self.on_fallback(stage)
            except Exception:  # noqa: BLE001 - metrics must never break detection
                pass

    def _vision_available(self) -> bool:
        return (self.s.vision_enabled and self.llm is not None
                and self.s.llm_vision_model in self.llm.available_models())

    def _skip_reason(self, text: str, pages: int | None, rules: RuleResult) -> str | None:
        """Why the LLM isn't needed (a long document without a single ID signal), or None."""
        if rules.signals:
            return None
        s = self.s
        if s.llm_skip_min_pages > 0 and pages is not None and pages >= s.llm_skip_min_pages:
            return f"{pages} pages and no ID signal; LLM skipped"
        if s.llm_skip_min_chars > 0 and len("".join(text.split())) >= s.llm_skip_min_chars:
            return "long text and no ID signal; LLM skipped"
        return None

    def detect(self, text: str, normalized_text: str, image: np.ndarray | None = None, pages: int | None = None
               ) -> tuple[IdDetection, dict[str, int]]:
        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        rules: RuleResult = evaluate(text, normalized_text, self.rules)
        timings["rules"] = round((time.perf_counter() - t0) * 1000)
        reasons = [f"rules: {r}" for r in rules.reasons]
        m = rules.mrz
        mrz = parse_fields(m, text=text) if m is not None and m.valid_count >= len(m.checks) - 1 else None

        if rules.decisive:
            return IdDetection(is_id=True, doc_type=rules.mrz.doc_type, confidence=MRZ_CONFIDENCE,
                               decided_by=["rules"], reasons=reasons, mrz=mrz), timings

        if skip := self._skip_reason(text, pages, rules):
            return IdDetection(is_id=False, doc_type="not_id", confidence=self.s.llm_skip_confidence,
                               decided_by=["rules"], reasons=reasons + [f"rules: {skip}"]), timings

        decided_by: list = ["rules"]
        p = rules.score
        llm_type = vision_type = None

        if not text.strip():
            reasons.append("no text extracted")
        elif not (self.s.llm_enabled and self.llm):
            reasons.append("llm: disabled")
        else:
            t0 = time.perf_counter()
            try:
                v = self.llm.classify_text(text)
                p = (1 - self.s.rules_weight) * _p_id(v) + self.s.rules_weight * rules.score
                decided_by.append("llm")
                llm_type = v.docType if v.isId else None
                reasons += [f"llm: {r}" for r in v.reasons]
            except LlmError:
                reasons.append("llm: unavailable, using rules only")
                self._fallback("llm")
            timings["llm"] = round((time.perf_counter() - t0) * 1000)

        if "llm" not in decided_by:  # rules only: pull towards 0.5
            margin = RULES_ONLY_MAX_CONFIDENCE - 0.5
            p = min(max(p, 0.5 - margin), 0.5 + margin)

        if max(p, 1 - p) < self.s.vision_threshold and image is not None:
            if self._vision_available():
                t0 = time.perf_counter()
                try:
                    vv = self.llm.classify_image(image)
                    has_text_evidence = "llm" in decided_by or bool(rules.signals)
                    p = ((1 - self.s.vision_weight) * p + self.s.vision_weight * _p_id(vv)
                         if has_text_evidence else _p_id(vv))
                    decided_by.append("vision")
                    vision_type = vv.docType if vv.isId else None
                    reasons += [f"vision: {r}" for r in vv.reasons]
                except LlmError:
                    reasons.append("vision: unavailable")
                    self._fallback("vision")
                timings["vision"] = round((time.perf_counter() - t0) * 1000)
            elif self.s.vision_enabled:
                reasons.append(f"vision: model {self.s.llm_vision_model} not available")
                self._fallback("vision")

        is_id = p >= 0.5
        doc_type = (vision_type or llm_type or rules.doc_type or "other_id") if is_id else "not_id"
        confidence = round(p if is_id else 1 - p, 3)
        return IdDetection(is_id=is_id, doc_type=doc_type, confidence=confidence,
                           decided_by=decided_by, reasons=reasons, mrz=mrz), timings
