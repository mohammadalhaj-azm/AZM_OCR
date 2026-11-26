"""Extract-and-answer: route every page to the right engine, then correct the text and answer questions with the LLM.

Two stages, like the main pipeline (the job worker can run them on different threads, LLM_WORKER_THREADS):

`extract_routed` (stage 1: OCR / GPU)
    1. `extract` as for /v1/documents: text layers taken as they are, everything else OCR'd by PaddleOCR. An
       `inspect` hook keeps PaddleOCR's signals per page (and the page image where another engine may need it).
    2. Per page, a route (app/answer/routing.py): the signals, else the vision model, else the configured fallback
       (marked `uncertain`). Pages routed to handwriting are read by the handwriting model, up to
       HANDWRITING_MAX_CONCURRENT pages at once (vLLM); for handwritten and mixed pages PaddleOCR's reading is kept
       next to it.
    3. Review items that need no model: low-confidence lines, handwriting warnings, uncertain routes, protected
       values that the two readings of a page disagree on, engines that were unavailable.

`finish` (stage 2: LLM)
    4. Correction, per page chunk (ANSWER_LLM_CONCURRENCY chunks at once), with protected values masked
       (app/answer/guards.py). A chunk whose reply breaks a protected value keeps its original text and is flagged
       ("restore + flag"). Parallel or not, the result is assembled in page and chunk order.
    5. Questions, from the corrected text, with citations checked against the pages.

The text model failing never fails the job: the result says what is missing (`correction.status`, answer
`status: unavailable`). Holds document text: never log it.
"""

import contextvars
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, TypeVar

import cv2
import numpy as np

from app.answer import guards
from app.answer.llm import AnswerLlm, LlmError
from app.answer.routing import (
    HANDWRITING,
    PADDLE,
    TEXT_LAYER,
    decide,
    document_kind,
    engines_for,
    needs_image,
    ocr_signals,
)
from app.config import Settings
from app.extractors import extract
from app.handwriting.backends import HandwritingUnavailable
from app.handwriting.guards import ink_ratio
from app.handwriting.service import HandwritingBusy
from app.normalize import detect_languages
from app.schemas import ExtractionResult

TASK = "extract_and_answer"
Mode = Literal["auto", "printed", "handwritten"]
MAX_OCR_REVIEW_ITEMS = 20  # low-confidence lines reported per page
MAX_DISAGREEMENTS = 10
T = TypeVar("T")
R = TypeVar("R")
NOT_FOUND = {"ar": "لا يحتوي المستند على هذه المعلومة.", "en": "The document does not contain this information."}


@dataclass
class AnswerOptions:
    questions: list[str] = field(default_factory=list)
    mode: Mode = "auto"
    correct: bool = True

    @classmethod
    def of(cls, value: "AnswerOptions | dict | None") -> "AnswerOptions":
        if isinstance(value, AnswerOptions):
            return value
        d = value or {}
        return cls(list(d.get("questions") or []), d.get("mode") or "auto", bool(d.get("correct", True)))

    def to_job(self) -> dict:
        return {"questions": self.questions, "mode": self.mode, "correct": self.correct}


@dataclass
class Routed:
    """Output of stage 1. Holds document text: never log it."""
    options: AnswerOptions
    pages: list[dict]
    timings: dict[str, int]
    vision_used: bool = False
    handwriting: dict | None = None  # the handwriting model's info, when it read a page


def shrink(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    f = max_side / max(h, w)
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


def ordered_map(fn: Callable[[T], R], items: Iterable[T], workers: int, name: str) -> list[R]:
    """[fn(x) for x in items], on up to `workers` threads, results in input order. Each task runs in a copy of the
    caller's context, so its log lines keep the job's ids (app/logs.py). An exception is raised as in the loop."""
    items = list(items)
    workers = min(max(1, workers), len(items))
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(workers, thread_name_prefix=name) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, x) for x in items]
        return [f.result() for f in futures]


def make_inspector(s: Settings, mode: Mode):
    def inspect(img: np.ndarray, out) -> dict:
        ink = ink_ratio(img) if out.detected == 0 else None
        sig = ocr_signals(out, ink, s)
        decision, reason = decide(sig, s)
        arabic = (detect_languages(out.text) or [None])[0] == "ar"
        keep = needs_image(decision, s, arabic, mode)
        return {"signals": sig, "decision": decision, "reason": reason, "arabic": arabic,
                "image": shrink(img, s.answer_keep_max_side) if keep else None}
    return inspect


def _fallback(svc, stage: str) -> None:
    fn = getattr(svc, "on_llm_fallback", None)
    if fn is not None:
        try:
            fn(stage)
        except Exception:  # noqa: BLE001 - metrics must never fail a job
            pass


def _item(page: int, reason: str, category: str, source: str, quote: str | None = None,
          suggestion: str | None = None, located: bool | None = None, confidence: float | None = None) -> dict:
    return {"page": page, "quote": quote, "suggestion": suggestion, "reason": reason, "category": category,
            "source": source, "located": located, "confidence": confidence}


# --- stage 1 ----------------------------------------------------------------------------------------------------------

def extract_routed(data: bytes, svc, settings: Settings, options: AnswerOptions | dict | None) -> tuple[
        ExtractionResult, Routed]:
    """Raises FileRejected for unusable uploads."""
    opts = AnswerOptions.of(options)
    s = settings
    ex_settings = s
    if opts.mode == "handwritten":  # also scanned PDFs with a (junk) text layer: render and read every page
        ex_settings = s.model_copy(update={"min_text_chars_per_page": 10 ** 9})
    ex = extract(data, ex_settings, svc.engine, inspect=make_inspector(s, opts.mode))
    alm = AnswerLlm(svc.llm, s)
    hw = getattr(svc, "handwriting", None)
    vision: list[bool] = []  # checked once, on the first unclear page
    t = {"routing": 0.0, "handwriting": 0.0}
    hw_pages = 0
    routed = Routed(opts, [], {})
    to_read = []  # (index in routed.pages, page, route, image)
    for p in ex.pages:
        if p.method == "text_layer":
            routed.pages.append(_text_layer_page(p))
            continue
        t0 = time.perf_counter()
        route, image = _route(p, s, opts, alm, vision, svc, routed)
        t["routing"] += time.perf_counter() - t0
        to_read.append((len(routed.pages), p, route, image))
        routed.pages.append(None)
    # The pages are read in parallel: the handwriting service's slots bound how many reach the model at once, across
    # all jobs. Paddle-only pages cost nothing here (PaddleOCR already ran in `extract`).
    t0 = time.perf_counter()
    workers = s.handwriting_max_concurrent if hw is not None and getattr(hw, "enabled", False) else 1
    read = ordered_map(lambda it: _read(it[1], it[2], it[3], s, hw), to_read, workers, "answer-read")
    for (i, *_), (page, used_hw) in zip(to_read, read):
        routed.pages[i] = page
        if used_hw:
            hw_pages += 1
            routed.handwriting = hw.model_info()
    t["handwriting"] = time.perf_counter() - t0
    routed.timings = {"routing": round(t["routing"] * 1000), "handwriting": round(t["handwriting"] * 1000),
                      "handwritingPages": hw_pages}
    return ex, routed


def _text_layer_page(p) -> dict:
    route = {"kind": "printed", "decidedBy": TEXT_LAYER, "uncertain": False, "engines": [TEXT_LAYER],
             "signals": None, "reasons": ["the file's own text layer"]}
    return {"page": p.page, "method": p.method, "engine": TEXT_LAYER, "route": route, "text": p.text,
            "confidence": None, "readings": [{"engine": TEXT_LAYER, "text": p.text, "confidence": None,
                                              "warnings": []}],
            "warnings": [], "review": [], "lines": None, "width": None, "height": None, "_marked": p.text,
            "_reference": None}


def _route(p, s: Settings, opts: AnswerOptions, alm: AnswerLlm, vision: list[bool], svc, routed: Routed):
    insp = p.inspection or {"signals": None, "decision": "unclear", "reason": "no OCR signals", "arabic": False,
                            "image": None}
    image, reasons = insp["image"], [insp["reason"]]
    uncertain = False
    if opts.mode != "auto":
        kind, decided_by = opts.mode, "override"
        reasons = [f"mode={opts.mode} was requested"]
    elif insp["decision"] != "unclear":
        kind, decided_by = insp["decision"], "ocr_signals"
    else:
        kind, decided_by, uncertain = "unknown", "fallback", True
        if not vision and image is not None:
            vision.append(alm.vision_available())
        if image is None:
            reasons.append("no page image for the vision model")
        elif not vision[0]:
            reasons.append("the vision model is not available")
            if s.routing_vision and s.vision_enabled:
                _fallback(svc, "routing_vision")
        else:
            try:
                kind, reasons = alm.classify_writing(image)
                decided_by, uncertain = "vision", False
                routed.vision_used = True
            except LlmError:
                reasons.append("the vision model failed")
                _fallback(svc, "routing_vision")
        if uncertain:
            reasons.append(f"fallback: {s.routing_uncertain_fallback}")
    engines = engines_for(kind, decided_by, s, insp["arabic"], bool(p.text.strip()))
    route = {"kind": kind, "decidedBy": decided_by, "uncertain": uncertain, "engines": engines,
             "signals": insp["signals"], "reasons": reasons}
    return route, image


def _confident_lines(p, s: Settings) -> list[dict]:
    return [ln for ln in (p.lines or []) if ln.confidence >= s.routing_low_line_conf]


def _read(p, route: dict, image, s: Settings, hw) -> tuple[dict, bool]:
    """Run the route's engines on the page: (page record, whether the handwriting model read it)."""
    n = p.page
    warnings: list[str] = []
    review: list[dict] = []
    readings: dict[str, dict] = {PADDLE: {"engine": PADDLE, "text": p.text, "confidence": p.confidence,
                                          "warnings": []}}
    used_hw = False
    engines = list(route["engines"])
    if HANDWRITING in engines:
        problem = None
        if hw is None or not getattr(hw, "enabled", False):
            problem = "handwriting_unavailable", "the handwriting model is disabled on this server"
        elif image is None:
            problem = "handwriting_unavailable", "no page image was kept for the handwriting model"
        else:
            try:
                out = hw.transcribe_image(image)
                used_hw = True
                hw_warnings = list(out["warnings"]) + (["blank"] if out.get("skipped") == "blank" else [])
                readings[HANDWRITING] = {"engine": HANDWRITING, "text": out["text"], "confidence": None,
                                         "warnings": hw_warnings}
                if out.get("skipped") == "blank" and p.text.strip():
                    review.append(_item(n, "the handwriting model's ink check found the page blank, but PaddleOCR "
                                           "read text on it", "inconsistent", "handwriting"))
                for w in out["warnings"]:
                    review.append(_item(n, {"truncated": "the handwriting model's output was cut off: text may be "
                                                         "missing",
                                            "repetition_loop": "the handwriting model repeated itself: part of "
                                                               "this text may be invented"}.get(w, w),
                                        "model_warning", "handwriting"))
            except HandwritingBusy:
                problem = "handwriting_busy", "the handwriting model stayed busy too long"
            except HandwritingUnavailable as e:  # its message never contains document content
                problem = "handwriting_unavailable", str(e)
        if problem:
            warnings.append(problem[0])
            engines.remove(HANDWRITING)
            if PADDLE not in engines:
                engines.append(PADDLE)
            review.append(_item(n, f"{problem[1]}; the page was read by printed-text OCR only, which handles "
                                   "handwriting poorly", "unavailable", "handwriting"))
    route["engines"] = engines
    primary = readings[engines[0]]
    if route["uncertain"]:
        review.append(_item(n, "it is not clear whether this page is printed or handwritten ("
                               + "; ".join(route["reasons"]) + ")", "routing", "routing"))

    marked, reference = primary["text"], None
    if primary["engine"] == PADDLE:
        low = [ln for ln in (p.lines or []) if ln.confidence < s.routing_low_line_conf]
        for ln in low[:MAX_OCR_REVIEW_ITEMS]:
            review.append(_item(n, f"low OCR confidence ({ln.confidence:.2f})", "low_confidence", "ocr_confidence",
                                quote=ln.text, located=True, confidence=ln.confidence))
        if p.lines:
            marked = "\n".join(f"{guards.LOW_OPEN}{ln.text}{guards.LOW_CLOSE}"
                               if ln.confidence < s.routing_low_line_conf else ln.text for ln in p.lines)
    elif PADDLE in engines:  # handwriting first: PaddleOCR's confident lines are the reference
        confident = _confident_lines(p, s)
        reference = "\n".join(ln.text for ln in confident) or None
        hw_values = guards.protected_values(primary["text"])
        seen = set()
        for ln in confident:
            for m in guards.PROTECTED.finditer(ln.text):
                key = next(iter(guards.protected_values(m.group(0))), "")
                if key and key not in hw_values and key not in seen and len(seen) < MAX_DISAGREEMENTS:
                    seen.add(key)
                    review.append(_item(n, "PaddleOCR read this value on the page, but the handwriting model's "
                                           "reading does not contain it", "inconsistent", "readings",
                                        quote=m.group(0), located=True))
    kept = [readings[e] for e in engines if e in readings]
    page = {"page": n, "method": p.method, "engine": primary["engine"], "route": route, "text": primary["text"],
            "confidence": primary["confidence"], "readings": kept, "warnings": warnings, "review": review,
            "lines": p.lines, "width": p.width, "height": p.height, "_marked": marked, "_reference": reference}
    return page, used_hw


# --- stage 2 ----------------------------------------------------------------------------------------------------------

def finish(ex: ExtractionResult, routed: Routed, svc, settings: Settings, include_boxes: bool = False) -> dict:
    """The answer-specific part of the result (merged into the pipeline result by app/pipeline.py).
    include_boxes: OCR pages keep PaddleOCR's `lines`, `width`, `height` (as in /v1/documents)."""
    s = settings
    opts = routed.options
    alm = AnswerLlm(svc.llm, s)
    timings: dict[str, int] = {}
    review: list[dict] = [it for p in routed.pages for it in p["review"]]
    warnings: list[str] = []

    t0 = time.perf_counter()
    if not opts.correct:
        correction = {"status": "skipped", "model": None}
        for p in routed.pages:
            p["correctedText"], p["changes"] = None, None
    elif not alm.text_available():
        correction = {"status": "unavailable", "model": None}
        review.append(_item(None, "the text model is disabled: the text was not corrected", "unavailable", "llm"))
        for p in routed.pages:
            p["correctedText"], p["changes"] = None, None
    else:
        correction = _correct_all(routed.pages, alm, review, svc)
    timings["correction"] = round((time.perf_counter() - t0) * 1000)

    corrected_full = None
    if correction["status"] in ("done", "partial"):
        corrected_full = "\n\n".join(p["correctedText"] for p in routed.pages if p["correctedText"])

    answers, answered_by_llm = None, False
    if opts.questions:
        t0 = time.perf_counter()
        answers, answered_by_llm = _answer(routed.pages, opts.questions, alm, review, warnings, svc)
        timings["answer"] = round((time.perf_counter() - t0) * 1000)

    boxes = {p["page"]: p for p in ex.to_json_dict(include_lines=True)["pages"]} if include_boxes else {}
    pages_out = []
    for p in routed.pages:
        page = {k: v for k, v in p.items()
                if not k.startswith("_") and k not in ("review", "lines", "width", "height")}
        for key in ("width", "height", "lines"):
            if boxes.get(p["page"], {}).get(key) is not None:
                page[key] = boxes[p["page"]][key]
        pages_out.append(page)
    kinds = [p["route"]["kind"] for p in routed.pages]
    counts: dict[str, int] = {}
    for k in kinds:
        counts[k] = counts.get(k, 0) + 1
    out = {
        "pages": pages_out,
        "correctedText": corrected_full,
        "routing": {"document": document_kind(kinds), "pages": counts,
                    "uncertainPages": [p["page"] for p in routed.pages if p["route"]["uncertain"]]},
        "review": review,
        "correction": correction,
        "warnings": warnings,
        "_timings": timings,
        "_llm": bool(correction["model"]) or answered_by_llm,  # the text model produced part of the result
    }
    if answers is not None:
        out["answers"] = answers
    return out


def _correct_chunk(p: dict, chunk: str, alm: AnswerLlm) -> tuple[str, list[dict], bool]:
    """One chunk of a page: (text, review items, whether the model answered). Thread-safe (no shared state)."""
    s = alm.s
    review: list[dict] = []
    original = guards.strip_low_marks(chunk)
    masked, values = guards.mask(chunk)
    ref = (p["_reference"] or "")[:s.answer_chunk_chars] or None  # not masked: it is never copied
    try:
        c = alm.correct(masked, p["engine"], p["readings"][0]["warnings"], ref)
    except LlmError:
        review.append(_item(p["page"], "the text model failed: this part was not corrected", "unavailable", "llm"))
        return original, review, False
    # the model answered; a reply that breaks a protected value is restored, not a failure
    problem = guards.check_placeholders(c.text, values)
    if problem:
        text = original
        review.append(_item(p["page"], f"{problem}: this part keeps its original text", "protected_value",
                            "protected_value"))
    else:
        text = guards.strip_low_marks(guards.unmask(c.text, values)).strip()
    for it in c.review:
        quote = guards.strip_low_marks(guards.unmask(it["quote"], values)).strip()
        suggestion = it["suggestion"]
        if suggestion is not None:
            suggestion = guards.strip_low_marks(guards.unmask(suggestion, values))
            if guards.PLACEHOLDER.search(suggestion):
                suggestion = None  # it referred to a value we can't map back
        review.append(_item(p["page"], it["reason"] or "flagged by the text model", it["category"], "llm",
                            quote=quote or None, suggestion=suggestion,
                            located=guards.located(quote, original) if quote else None))
    return text, review, True


def _correct_all(pages: list[dict], alm: AnswerLlm, review: list[dict], svc) -> dict:
    s = alm.s
    tasks = []  # (page, chunk) in page and chunk order
    for p in pages:
        if not p["text"].strip():
            p["correctedText"], p["changes"] = p["text"], []
            continue
        tasks += [(p, chunk) for chunk in guards.chunks(p["_marked"], s.answer_chunk_chars)]
    results = ordered_map(lambda t: _correct_chunk(t[0], t[1], alm), tasks, s.answer_llm_concurrency,
                          "answer-correct")
    ok = failed = 0
    by_page: dict[int, list[str]] = {}
    for (p, _), (text, items, answered) in zip(tasks, results):
        ok, failed = ok + answered, failed + (not answered)
        by_page.setdefault(id(p), []).append(text)
        review.extend(items)
    for p in pages:
        if id(p) in by_page:
            p["correctedText"] = "\n".join(c for c in by_page[id(p)] if c)
            p["changes"] = guards.word_changes(p["text"], p["correctedText"])
    if failed:
        _fallback(svc, "answer_llm")
    status = "done" if not failed else ("partial" if ok else "unavailable")
    if status == "unavailable":  # nothing was corrected: no corrected text at all
        for p in pages:
            p["correctedText"], p["changes"] = None, None
    return {"status": status, "model": None if status == "unavailable" else alm.text_model}


def _lang(text: str) -> str:
    langs = detect_languages(text, min_letters=1)
    return langs[0] if langs else "en"


def _answer(pages: list[dict], questions: list[str], alm: AnswerLlm, review: list[dict], warnings: list[str],
            svc) -> tuple[list[dict], bool]:
    """(answers, whether the text model produced them)."""
    s = alm.s
    texts = {p["page"]: (p.get("correctedText") or p["text"]) for p in pages}
    originals = {p["page"]: p["text"] for p in pages}
    given, used, truncated = [], 0, False
    for n, text in texts.items():
        if not text.strip():
            continue
        if used + len(text) > s.answer_context_chars:
            truncated = True
            rest = s.answer_context_chars - used
            if rest > 500 and not given:  # a single huge first page: give its head
                given.append((n, text[:rest]))
            break
        given.append((n, text))
        used += len(text)
    if truncated:
        warnings.append("context_truncated")

    def unavailable() -> list[dict]:
        return [{"question": q, "status": "unavailable", "answer": None, "citations": [], "needsReview": True}
                for q in questions]

    if not alm.text_available():
        return unavailable(), False
    if not given:  # no text at all: nothing to ask the model about
        return [{"question": q, "status": "not_found", "answer": NOT_FOUND[_lang(q)], "citations": [],
                 "needsReview": False} for q in questions], False
    try:
        got = alm.answer(given, questions, truncated)
    except LlmError:
        _fallback(svc, "answer_llm")
        review.append(_item(None, "the text model failed: the questions were not answered", "unavailable", "llm"))
        return unavailable(), False
    uncertain_pages = {p["page"] for p in pages if p["route"]["uncertain"]}
    out = []
    for q, a in zip(questions, got):
        citations = []
        for c in a["citations"]:
            page = c["page"]
            verified = page in texts and guards.located(c["quote"], texts[page], originals[page])
            citations.append({"page": page, "quote": c["quote"], "verified": verified})
        if not a["found"]:
            out.append({"question": q, "status": "not_found", "answer": a["answer"] or NOT_FOUND[_lang(q)],
                        "citations": [], "needsReview": truncated})
            continue
        flagged = any(it["page"] == c["page"] and it["quote"] and guards.overlaps(it["quote"], c["quote"])
                      for c in citations for it in review)
        needs = (not citations or not all(c["verified"] for c in citations) or flagged
                 or any(c["page"] in uncertain_pages for c in citations))
        out.append({"question": q, "status": "answered", "answer": a["answer"], "citations": citations,
                    "needsReview": needs})
    return out, True
