"""Response models of the REST API: the typed contract in the OpenAPI spec (docs/openapi.json).

They describe the JSON exactly as the pipeline produces it (tests/test_api.py compares the two), and are separate from
the pipeline's own models so the contract doesn't change when internals do. Compatibility rules: docs/INTEGRATION.md,
"API versioning and compatibility".

A field in a model's `_omit_if_none` is left out when it has no value (e.g. `lines` without includeBoxes); every
other field is always present, `null` when it has no value (e.g. `confidence` of a text-layer page).
"""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_serializer

from app.detection.llm import DocType

Language = Literal["ar", "en"]


class _Out(BaseModel):
    _omit_if_none: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _omit_absent(self, handler):
        return {k: v for k, v in handler(self).items() if v is not None or k not in self._omit_if_none}


class FileInfo(_Out):
    name: str = Field(description="The file name as uploaded")
    mimeType: str = Field(description="Detected from the content (magic bytes), never from the name")
    pages: int


class Line(_Out):
    text: str
    confidence: float = Field(description="0-1")
    bbox: list[float] = Field(min_length=4, max_length=4,
                              description="[x0, y0, x1, y1] as fractions (0-1) of the page, origin top-left")


class Page(_Out):
    _omit_if_none = frozenset({"width", "height", "lines"})

    page: int = Field(description="1-based")
    method: Literal["text_layer", "ocr"] = Field(description="text_layer: taken from the PDF/DOCX; ocr: recognized")
    text: str
    confidence: float | None = Field(None, description="Average OCR confidence 0-1; null for text_layer pages")
    width: int | None = Field(None, description="OCR pages with includeBoxes: pixel width of the OCR'd image")
    height: int | None = Field(None, description="OCR pages with includeBoxes: pixel height of the OCR'd image")
    lines: list[Line] | None = Field(None, description="OCR pages with includeBoxes: lines in reading order")


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


class Processing(_Out):
    """What produced this result, for audits: the service version, the processing profile, and the OCR models, rules
    file version (start of its SHA-256) and LLMs that were actually used. A part that wasn't used is left out."""

    _omit_if_none = frozenset({"profile", "ocr", "rules", "llm", "vision", "withheld"})

    version: str
    profile: str | None = Field(None, description="The processing profile (`default`: the global settings)")
    ocr: OcrUsed | None = None
    rules: str | None = None
    llm: str | None = None
    vision: str | None = None
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


class JobView(_Out):
    """`status` queued/processing/done/failed. When done, the result fields are included; when failed, `error`.
    `webhook` is present when a callbackUrl was given."""

    _omit_if_none = frozenset({"externalId", "metadata", "file", "pages", "fullText", "normalizedText", "languages",
                               "idDetection", "timingsMs", "processing", "error", "webhook"})

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
