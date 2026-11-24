"""Result models. JSON uses camelCase (e.g. `mimeType`, `fullText`)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, PrivateAttr
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class OcrLine(CamelModel):
    text: str
    confidence: float
    bbox: list[float]  # [x0, y0, x1, y1], normalized 0..1 of the page (origin top-left)


class PageResult(CamelModel):
    page: int
    method: Literal["text_layer", "ocr"]
    text: str
    confidence: float | None = None  # average OCR confidence; None for text-layer pages
    # OCR pages only; serialized only when asked for (includeBoxes)
    lines: list[OcrLine] | None = None
    width: int | None = None  # pixel size of the OCR'd image
    height: int | None = None
    # Set by an `inspect` hook of `extract` (extract-and-answer routing): what the hook returned for this page's
    # image and OCR output. Never serialized.
    _inspection: Any = PrivateAttr(default=None)

    @property
    def inspection(self):
        return self._inspection


class ExtractionResult(CamelModel):
    mime_type: str
    pages: list[PageResult]  # page text is cleaned (see app.normalize.clean_text)
    normalized_text: str = ""  # for matching only
    languages: list[str] = []
    timings_ms: dict[str, int] = {}
    # First OCR'd page image (BGR ndarray), kept in memory for the vision fallback. Never serialized.
    _vision_image: Any = PrivateAttr(default=None)

    @property
    def vision_image(self):
        return self._vision_image

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    def to_json_dict(self, include_lines: bool = False) -> dict:
        """`pages`, `fullText`, `timingsMs`; the caller adds the `file` block (name, mimeType, pages).
        include_lines: add `lines`, `width`, `height` to OCR pages."""
        d = self.model_dump(by_alias=True, exclude={"mime_type"})
        for page in d["pages"]:
            for key in ("lines", "width", "height"):
                if not include_lines or page.get(key) is None:
                    page.pop(key, None)
        d["fullText"] = self.full_text
        return d
