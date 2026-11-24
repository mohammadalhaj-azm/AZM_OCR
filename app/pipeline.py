"""Full pipeline: detect type -> extract -> normalize -> rules -> LLM -> (vision) -> result dict.

Split into two stages so the job worker can overlap them (LLM_WORKER_THREADS > 0): `extract_stage` (CPU: PDF/DOCX,
OCR, normalization) and `detect_stage` (rules + LLM + vision, mostly waiting on the LLM server).

A processing profile (app.profiles) can change settings per client system: rules file, LLM/vision use, page limit,
OCR DPI, and what the result keeps (`withhold` runs before the result is stored or returned).

`task="extract_and_answer"` (POST /v1/extract-and-answer, app/answer) uses the same two stages: stage 1 extracts and
routes pages to PaddleOCR or the handwriting model, stage 2 has the LLM correct the text and answer the questions.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app import __version__
from app.answer import TASK, AnswerOptions, Routed, extract_routed, finish
from app.config import Settings
from app.detection.detector import IdDetection, IdDetector
from app.detection.llm import LlmClient
from app.extractors import extract
from app.normalize import clean_text, detect_languages, normalize_for_match
from app.ocr.engine import OcrPool
from app.profiles import DEFAULT_PROFILE, Profile, withhold
from app.schemas import ExtractionResult


@dataclass
class Services:
    settings: Settings
    engine: OcrPool
    llm: LlmClient
    detector: IdDetector
    _detectors: dict = field(default_factory=dict, repr=False)  # IdDetector per profile rules/LLM setup
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    on_llm_fallback: Callable[[str], None] | None = None  # for every detector (see IdDetector.on_fallback)
    handwriting: Any = None  # HandwritingService (set by create_app): extract-and-answer's handwriting engine

    @classmethod
    def create(cls, settings: Settings) -> "Services":
        llm = LlmClient(settings)
        return cls(settings, OcrPool(settings), llm, IdDetector(settings, llm))

    def report_fallbacks(self, fn: Callable[[str], None]) -> None:
        """Call `fn("llm" | "vision")` whenever ID detection goes on without a model (all profiles)."""
        self.on_llm_fallback = fn
        with self._lock:
            for detector in [self.detector, *self._detectors.values()]:
                if hasattr(detector, "on_fallback"):
                    detector.on_fallback = fn

    def reset_detectors(self) -> None:
        """After a configuration change in the admin dashboard (settings, rule sets): rebuild the detectors."""
        with self._lock:
            self._detectors.clear()
            self.detector = IdDetector(self.settings, self.llm, self.on_llm_fallback)

    def detector_for(self, profile: Profile):
        key = profile.detection_key()
        if key == DEFAULT_PROFILE.detection_key():
            return self.detector
        with self._lock:
            if key not in self._detectors:
                self._detectors[key] = IdDetector(profile.settings_for(self.settings), self.llm, self.on_llm_fallback)
            return self._detectors[key]


@dataclass
class Extracted:
    """Output of `extract_stage`, input of `detect_stage`. Holds document text: never log it."""

    filename: str
    ex: ExtractionResult
    include_boxes: bool
    t0: float = field(default_factory=time.perf_counter)
    detect_id: bool = True
    profile: Profile = DEFAULT_PROFILE
    settings: Settings | None = None  # the effective settings (profile applied)
    answer: Routed | None = None  # task extract_and_answer: the routed pages


def as_profile(profile: Profile | dict | None) -> Profile:
    """A job stores its profile as JSON (Profile.to_job)."""
    if profile is None:
        return DEFAULT_PROFILE
    return profile if isinstance(profile, Profile) else Profile.model_validate(profile)


def extract_stage(data: bytes, filename: str, svc: Services, max_pages: int | None = None,
                  include_boxes: bool = False, detect_id: bool = True,
                  profile: Profile | dict | None = None, task: str | None = None,
                  answer: AnswerOptions | dict | None = None) -> Extracted:
    """Raises FileRejected for unusable uploads. max_pages: the endpoint's page limit instead of MAX_PAGES (the
    profile's max_pages still applies). task: None (text + ID detection) or "extract_and_answer" with `answer`
    options (questions, mode, correct)."""
    t0 = time.perf_counter()
    prof = as_profile(profile)
    settings = prof.settings_for(svc.settings)
    if task == TASK:
        if max_pages is None:
            max_pages = svc.settings.answer_max_pages
    elif task is not None:
        raise ValueError(f"unknown task {task!r}")
    if max_pages is not None:
        limit = max_pages if prof.max_pages is None else min(max_pages, prof.max_pages)
        settings = settings.model_copy(update={"max_pages": limit})
    if task == TASK:
        ex, routed = extract_routed(data, svc, settings, answer)
        return Extracted(filename, ex, include_boxes, t0, False, prof, settings, routed)
    return Extracted(filename, extract(data, settings, svc.engine), include_boxes, t0, detect_id, prof, settings)


def provenance(x: Extracted, detection: IdDetection | None, detector) -> dict:
    """What produced the result: service version, profile, and the OCR models, rules file and LLMs actually used."""
    s = x.settings
    out: dict = {"version": __version__, "profile": x.profile.name}
    if any(p.method == "ocr" for p in x.ex.pages):
        out["ocr"] = {"det": s.ocr_det_model, "recAr": s.ocr_rec_model_ar, "recEn": s.ocr_rec_model_en or None,
                      "preprocess": s.ocr_preprocess, "dpi": s.ocr_dpi}
    if detection is not None:
        out["rules"] = getattr(detector, "rules_version", None)
        if "llm" in detection.decided_by:
            out["llm"] = s.llm_text_model
        if "vision" in detection.decided_by:
            out["vision"] = s.llm_vision_model
    return out


def detect_stage(x: Extracted, svc: Services, detect_id: bool | None = None) -> dict:
    """detect_id=None: as decided in extract_stage. The profile can switch detection off."""
    if x.settings is None:  # an Extracted built by hand
        x.settings = svc.settings
    if x.answer is not None:
        return answer_result(x, svc)
    ex = x.ex
    timings = dict(ex.timings_ms)
    detection, detector = None, None
    if (x.detect_id if detect_id is None else detect_id) and x.profile.detect_id:
        detector = svc.detector_for(x.profile)
        detection, det_timings = detector.detect(ex.full_text, ex.normalized_text, ex.vision_image,
                                                 pages=len(ex.pages))
        timings.update(det_timings)
    timings["total"] = round((time.perf_counter() - x.t0) * 1000)
    out = ex.to_json_dict(include_lines=x.include_boxes)
    result = {
        "file": {"name": x.filename, "mimeType": ex.mime_type, "pages": len(ex.pages)},
        "pages": out["pages"],
        "fullText": out["fullText"],
        "normalizedText": out["normalizedText"],
        "languages": out["languages"],
    }
    if detection is not None:
        result["idDetection"] = detection.model_dump(by_alias=True, exclude={"mrz"} if detection.mrz is None else None)
    result["timingsMs"] = timings
    result["processing"] = provenance(x, detection, detector)
    return withhold(result, x.profile)  # before anything is stored or returned


def answer_result(x: Extracted, svc: Services) -> dict:
    """Stage 2 of extract-and-answer: LLM correction and questions, and the result JSON."""
    ex, routed, s = x.ex, x.answer, x.settings
    part = finish(ex, routed, svc, s, x.include_boxes)
    timings = {**ex.timings_ms, **routed.timings, **part.pop("_timings")}
    timings["total"] = round((time.perf_counter() - x.t0) * 1000)
    full_text = "\n\n".join(p["text"] for p in part["pages"] if p["text"])
    processing = provenance(x, None, None)
    if routed.handwriting:
        processing["handwriting"] = {"backend": routed.handwriting.get("backend"),
                                     "name": routed.handwriting.get("name"),
                                     "device": routed.handwriting.get("device")}
    if part.pop("_llm"):
        processing["llm"] = s.answer_llm_model or s.llm_text_model
    if routed.vision_used:
        processing["vision"] = s.llm_vision_model
    if any(p["method"] == "ocr" for p in part["pages"]):
        processing["routing"] = {"lowLineConf": s.routing_low_line_conf, "printedMinConf": s.routing_printed_min_conf,
                                 "printedMaxLowShare": s.routing_printed_max_low_share,
                                 "handwrittenMaxConf": s.routing_handwritten_max_conf,
                                 "minKeptShare": s.routing_min_kept_share,
                                 "uncertainFallback": s.routing_uncertain_fallback,
                                 "printedArabic": s.route_printed_arabic}
    result = {
        "file": {"name": x.filename, "mimeType": ex.mime_type, "pages": len(ex.pages)},
        "pages": part["pages"],
        "fullText": full_text,
        "normalizedText": normalize_for_match(clean_text(full_text, fix_reversed=False)),
        "languages": detect_languages(full_text),
        "correctedText": part["correctedText"],
        "routing": part["routing"],
        "review": part["review"],
        "correction": part["correction"],
    }
    if "answers" in part:
        result["answers"] = part["answers"]
    if part["warnings"]:
        result["warnings"] = part["warnings"]
    result["timingsMs"] = timings
    result["processing"] = processing
    return withhold(result, x.profile)  # before anything is stored or returned


def analyze(data: bytes, filename: str, svc: Services, detect_id: bool = True, max_pages: int | None = None,
            include_boxes: bool = False, profile: Profile | dict | None = None, task: str | None = None,
            answer: AnswerOptions | dict | None = None) -> dict:
    """Result JSON without jobId/status. Raises FileRejected for unusable uploads.

    detect_id=False: text extraction only (no rules/LLM/vision, no `idDetection` key).
    max_pages: the endpoint's page limit (used by the synchronous endpoint).
    include_boxes: OCR pages also get `lines` [{text, confidence, bbox}], `width`, `height`.
    profile: a processing profile (app.profiles), or its JSON as stored in a job.
    task="extract_and_answer": routed OCR, LLM correction and answers (app/answer); `answer` holds its options.
    """
    return detect_stage(extract_stage(data, filename, svc, max_pages, include_boxes, detect_id, profile, task,
                                      answer), svc)


analyze.stages = (extract_stage, detect_stage)  # lets the worker run the two stages on separate threads
