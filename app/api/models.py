"""Response models of the REST API: the typed contract in the OpenAPI spec (docs/openapi.json).

They describe the JSON exactly as the pipeline produces it (tests/test_api.py compares the two), and are separate from
the pipeline's own models so the contract doesn't change when internals do. Compatibility rules: docs/INTEGRATION.md,
"API versioning and compatibility".

A field in a model's `_omit_if_none` is left out when it has no value (e.g. `lines` without includeBoxes); a field in
`_omit_if_unset` is left out when the result doesn't have it at all, and kept as `null` when the result has it with no
value (e.g. `correctedText`: absent from /v1/documents results, null in an extract-and-answer result whose text was
not corrected). Every other field is always present, `null` when it has no value (e.g. `confidence` of a text-layer
page).
"""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_serializer

from app.detection.llm import DocType

Language = Literal["ar", "en"]


class _Out(BaseModel):
    _omit_if_none: ClassVar[frozenset[str]] = frozenset()
    _omit_if_unset: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        return {k: v for k, v in handler(self).items()
                if (v is not None or k not in self._omit_if_none)
                and (k in self.model_fields_set or k not in self._omit_if_unset)}


class FileInfo(_Out):
    name: str = Field(description="The file name as uploaded")
    mimeType: str = Field(description="Detected from the content (magic bytes), never from the name")
    pages: int


class Line(_Out):
    text: str
    confidence: float = Field(description="0-1")
    bbox: list[float] = Field(min_length=4, max_length=4,
                              description="[x0, y0, x1, y1] as fractions (0-1) of the page, origin top-left")


# Building blocks of extract-and-answer pages (experimental, may change)

Engine = Literal["text_layer", "paddleocr", "handwriting"]
PageKind = Literal["printed", "handwritten", "mixed", "no_text", "unknown"]


class RouteSignals(_Out):
    """PaddleOCR's numbers for a scanned page or image (what the routing decided on)."""

    detected: int = Field(description="Text regions the detector found")
    read: int = Field(description="Regions recognized with a score >= OCR_MIN_SCORE")
    meanConfidence: float | None = Field(description="Mean recognition confidence 0-1; null when nothing was read")
    lowConfidenceShare: float | None = Field(description="Share of read regions below ROUTING_LOW_LINE_CONF")
    inkRatio: float | None = Field(description="Ink share of the page; only measured when no text was detected")


class Route(_Out):
    kind: PageKind = Field(description="unknown: unclear from PaddleOCR's signals and no vision verdict")
    decidedBy: Literal["text_layer", "ocr_signals", "vision", "override", "fallback"] = Field(
        description="override: the request's `mode`; fallback: ROUTING_UNCERTAIN_FALLBACK applied")
    uncertain: bool = Field(description="The kind is not known: review the page; both readings may be kept")
    engines: list[Engine] = Field(description="Engines whose readings were kept, the primary one first")
    signals: RouteSignals | None = Field(description="null for text-layer pages")
    reasons: list[str] = Field(description="Short reasons (from the signals, or the vision model's own)")


class Reading(_Out):
    engine: Engine
    text: str | None = Field(description="As the engine read it")
    confidence: float | None = Field(description="PaddleOCR's average confidence; always null for the handwriting "
                                                 "model and text layers (no calibrated confidence exists)")
    warnings: list[Literal["truncated", "repetition_loop", "blank"]]


class Change(_Out):
    original: str = Field(description="Words of the original text (empty: words were inserted)")
    corrected: str = Field(description="Words of the corrected text (empty: words were removed)")


class Page(_Out):
    """A page of the result. The fields from `engine` on are only in extract-and-answer results (experimental), where
    `text` is the primary engine's reading."""

    _omit_if_none = frozenset({"width", "height", "lines"})
    _omit_if_unset = frozenset({"engine", "route", "readings", "correctedText", "changes", "warnings"})

    page: int = Field(description="1-based")
    method: Literal["text_layer", "ocr"] = Field(description="text_layer: taken from the PDF/DOCX; ocr: recognized")
    text: str
    confidence: float | None = Field(None, description="Average OCR confidence 0-1; null for text_layer pages")
    width: int | None = Field(None, description="OCR pages with includeBoxes: pixel width of the OCR'd image")
    height: int | None = Field(None, description="OCR pages with includeBoxes: pixel height of the OCR'd image")
    lines: list[Line] | None = Field(None, description="OCR pages with includeBoxes: lines in reading order")
    engine: Engine | None = Field(None, description="Extract-and-answer: the engine whose reading is `text`")
    route: Route | None = Field(None, description="Extract-and-answer: how the page was routed, and why")
    readings: list[Reading] | None = Field(None, description="Extract-and-answer: every engine's reading (mixed and "
                                                              "uncertain pages have two)")
    correctedText: str | None = Field(None, description="Extract-and-answer: the LLM's corrected, formatted text; "
                                                        "null when not corrected")
    changes: list[Change] | None = Field(None, description="Extract-and-answer: word-level differences between text "
                                                           "and correctedText, computed by the service (not reported "
                                                           "by the LLM)")
    warnings: list[Literal["handwriting_unavailable", "handwriting_busy"]] | None = Field(
        None, description="Extract-and-answer: the handwriting model could not read the page")


class Mrz(_Out):
    """Parsed machine-readable zone. Personal data. Rely on a field only when its `checks` entry is true; names carry
    no check digit."""

    _omit_if_none = frozenset({"namesSource"})

    format: Literal["TD1", "TD2", "TD3"]
    documentCode: str | None = None
    issuingCountry: str | None = None
    surname: str | None = None
    givenNames: str | None = None
    documentNumber: str | None = None
    nationality: str | None = None
    birthDate: str | None = Field(None, description="ISO date YYYY-MM-DD")
    sex: Literal["M", "F", "X"] | None = None
    expiryDate: str | None = Field(None, description="ISO date YYYY-MM-DD")
    optionalData: str | None = None
    expired: bool | None = Field(None, description="expiryDate is before today")
    valid: bool = Field(description="All check digits passed")
    checks: dict[str, bool] = Field(description="ICAO check digit per field, e.g. documentNumber, birthDate, "
                                                "expiryDate, personalNumber, composite")
    namesSource: Literal["mrz", "mrz+text"] | None = Field(
        None, description="mrz+text: names were completed from the printed text")


class IdDetectionResult(_Out):
    _omit_if_none = frozenset({"mrz"})

    isId: bool
    docType: DocType
    confidence: float = Field(description="0-1, how sure the decision is")
    decidedBy: list[Literal["rules", "llm", "vision"]] = Field(description="The stages that took part, in order")
    reasons: list[str] = Field(description="Human-readable reasons, for review screens")
    mrz: Mrz | None = Field(None, description="Present when an MRZ with at most one failed check digit was found")


class OcrUsed(_Out):
    _omit_if_none = frozenset({"recEn"})

    det: str
    recAr: str
    recEn: str | None = None
    preprocess: bool
    dpi: int


class HandwritingUsed(_Out):
    backend: Literal["transformers", "openai"] | None = None
    name: str | None = Field(None, description="The handwriting model (Hugging Face id, local folder or served name)")
    device: str | None = Field(None, description="cpu / cuda for the transformers backend; null for openai")


class RoutingUsed(_Out):
    """The routing settings of extract-and-answer (ROUTING_*). Starting values, not calibrated: see
    eval/run_routing_eval.py."""

    lowLineConf: float
    printedMinConf: float
    printedMaxLowShare: float
    handwrittenMaxConf: float
    minKeptShare: float
    uncertainFallback: Literal["both", "handwriting", "printed"]
    printedArabic: Literal["paddle", "handwriting"]


class Processing(_Out):
    """What produced this result, for audits: the service version, the processing profile, and the OCR models, rules
    file version (start of its SHA-256) and LLMs that were actually used. A part that wasn't used is left out."""

    _omit_if_none = frozenset({"profile", "ocr", "rules", "llm", "vision", "withheld", "handwriting", "routing"})

    version: str
    profile: str | None = Field(None, description="The processing profile (`default`: the global settings)")
    ocr: OcrUsed | None = None
    rules: str | None = None
    llm: str | None = None
    vision: str | None = None
    handwriting: HandwritingUsed | None = Field(None, description="Extract-and-answer: the handwriting model, when "
                                                                  "it read a page")
    routing: RoutingUsed | None = Field(None, description="Extract-and-answer: the routing settings, when a page was "
                                                          "routed")
    withheld: list[Literal["text", "mrz"]] | None = Field(
        None, description="Left out by the profile (data minimization): `text` = pages[].text, fullText, "
                          "normalizedText and lines are empty and the AI models' reasons are removed; `mrz` = no "
                          "idDetection.mrz. Withheld data is not stored either.")


class JobCreated(BaseModel):
    jobId: str


class ExtractResult(_Out):
    """Text-only result. `languages` lists `ar`/`en` by amount of text."""

    file: FileInfo
    pages: list[Page]
    fullText: str = Field(description="All pages' text, as extracted")
    normalizedText: str = Field(description="Arabic-normalized text, for search and matching")
    languages: list[Language]
    timingsMs: dict[str, int] = Field(description="Milliseconds per stage, e.g. extract, ocr, llm, vision, total "
                                                  "(ocrPages: number of OCR'd pages); keys depend on the stages run")
    processing: Processing


class JobError(_Out):
    code: str = Field(description="Stable error code (see the error table); program against it")
    message: str = Field(description="For humans; may change")


class WebhookStatus(_Out):
    state: Literal["pending", "delivered", "failed"]
    attempts: int


# --- Experimental: POST /v1/extract-and-answer (additive fields of the job result; may change) ----------------------

ReviewCategory = Literal["illegible", "ambiguous", "possibly_misread", "inconsistent", "name", "low_confidence",
                         "model_warning", "routing", "protected_value", "unavailable"]


class ReviewItem(_Out):
    """Something a person should check. Never a guess presented as fact: `suggestion` is only a proposal."""

    page: int | None = Field(description="null: about the whole document")
    quote: str | None = Field(description="The text concerned, as in the original; null for page-level items")
    suggestion: str | None = Field(description="A proposed reading (from the LLM), or null")
    reason: str
    category: ReviewCategory
    source: Literal["llm", "ocr_confidence", "handwriting", "routing", "readings", "protected_value"] = Field(
        description="llm: flagged by the text model; the others are checks of the service itself")
    located: bool | None = Field(description="The quote was found in the original page text (false: the LLM's "
                                             "quote doesn't match it); null without a quote")
    confidence: float | None = Field(description="PaddleOCR confidence of the line (ocr_confidence items)")


class Citation(_Out):
    page: int
    quote: str = Field(description="Supporting text, as quoted by the LLM")
    verified: bool = Field(description="The quote was found on that page")


class Answer(_Out):
    question: str
    status: Literal["answered", "not_found", "unavailable"] = Field(
        description="not_found: the document doesn't contain it; unavailable: the text model could not be used")
    answer: str | None = Field(description="From the document only, in the question's language")
    citations: list[Citation]
    needsReview: bool = Field(description="A citation could not be verified, overlaps a review item or an uncertain "
                                          "page, the answer has no citation, or only part of the document was read")


class Correction(_Out):
    status: Literal["done", "partial", "skipped", "unavailable"] = Field(
        description="partial: some parts kept their original text (see review); skipped: correct=false")
    model: str | None


class RoutingSummary(_Out):
    document: PageKind
    pages: dict[str, int] = Field(description="Pages per kind")
    uncertainPages: list[int]


class JobView(_Out):
    """`status` queued/processing/done/failed. When done, the result fields are included; when failed, `error`.
    `webhook` is present when a callbackUrl was given. Jobs of POST /v1/extract-and-answer (experimental) also have
    `correctedText`, `routing`, `review`, `correction`, and `answers` when questions were asked."""

    _omit_if_none = frozenset({"externalId", "metadata", "file", "pages", "fullText", "normalizedText", "languages",
                               "idDetection", "timingsMs", "processing", "error", "webhook", "routing", "review",
                               "correction", "answers", "warnings"})
    _omit_if_unset = frozenset({"correctedText"})

    jobId: str
    status: Literal["queued", "processing", "done", "failed"]
    externalId: str | None = None
    metadata: dict[str, Any] | None = None
    file: FileInfo | None = None
    pages: list[Page] | None = None
    fullText: str | None = None
    normalizedText: str | None = None
    languages: list[Language] | None = None
    idDetection: IdDetectionResult | None = None
    correctedText: str | None = Field(None, description="Extract-and-answer: all pages' corrected text; null when "
                                                        "the text was not corrected")
    routing: RoutingSummary | None = None
    review: list[ReviewItem] | None = None
    correction: Correction | None = None
    answers: list[Answer] | None = None
    warnings: list[Literal["context_truncated"]] | None = Field(
        None, description="context_truncated: the questions were answered from the first pages only")
    timingsMs: dict[str, int] | None = None
    processing: Processing | None = None
    error: JobError | None = None
    webhook: WebhookStatus | None = None


# --- Experimental: POST /v1/handwriting (separate from the models above; may change) ------------------------------

class HandwritingPage(_Out):
    _omit_if_none = frozenset({"skipped"})

    page: int = Field(description="1-based")
    text: str = Field(description="As the model read it: only line endings and surrounding whitespace are tidied")
    confidence: None = Field(None, description="Always null: the model gives no calibrated confidence, and none is "
                                               "made up")
    ms: int = Field(description="Milliseconds spent on this page")
    skipped: Literal["blank"] | None = Field(
        None, description="blank: the page has (almost) no ink, so it was not sent to the model")
    warnings: list[Literal["truncated", "repetition_loop"]] = Field(
        description="truncated: the output hit HANDWRITING_MAX_NEW_TOKENS; repetition_loop: the output repeats "
                    "itself (a decoding loop). Review such pages")


class HandwritingModel(_Out):
    backend: Literal["transformers", "openai"]
    name: str = Field(description="The model (Hugging Face id, local folder or served name)")
    device: str | None = Field(None, description="cpu / cuda for the transformers backend; null for openai")
    maxPixels: int = Field(description="Largest image size the model saw (pixels)")
    segmenter: str = Field(description="How pages were split for the model (whole_page in this version)")


class HandwritingResult(_Out):
    """Experimental handwriting OCR result. `languages` lists `ar`/`en` by amount of text."""

    file: FileInfo
    pages: list[HandwritingPage]
    fullText: str = Field(description="All pages' text, blank pages left out")
    languages: list[Language]
    timingsMs: dict[str, int] = Field(description="Milliseconds: decode, model, total")
    model: HandwritingModel
