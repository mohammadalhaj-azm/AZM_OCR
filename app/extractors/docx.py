"""DOCX: paragraphs and tables (document order) plus OCR of embedded images.

Output pages: page 1 = document text (text_layer); pages 2..n = embedded images (ocr).
"""

import io
from collections.abc import Callable

import numpy as np
from docx import Document
from docx.table import Table

from app.config import Settings
from app.extractors.filetype import FileRejected
from app.extractors.image import decode_images
from app.schemas import PageResult

MIN_IMAGE_SIDE = 32  # skip icons / bullets


def _table_lines(table: Table) -> list[str]:
    lines = []
    for row in table.rows:
        cells: list[str] = []
        for cell in row.cells:
            t = cell.text.strip()
            if t and (not cells or cells[-1] != t):  # merged cells repeat their text
                cells.append(t)
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _paragraph_lines(paragraphs) -> list[str]:
    return [p.text.strip() for p in paragraphs if p.text.strip()]


def extract_docx(data: bytes, s: Settings, ocr_page: Callable[[np.ndarray], PageResult]) -> list[PageResult]:
    try:
        doc = Document(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001
        raise FileRejected("corrupt", "DOCX could not be opened") from e

    lines: list[str] = []
    seen_headers: set[str] = set()
    for section in doc.sections:
        for hdr in (section.header, section.footer):
            for line in _paragraph_lines(hdr.paragraphs):
                if line not in seen_headers:
                    seen_headers.add(line)
                    lines.append(line)
    for block in doc.iter_inner_content():
        if isinstance(block, Table):
            lines.extend(_table_lines(block))
        elif block.text.strip():
            lines.append(block.text.strip())
    pages = [PageResult(page=1, method="text_layer", text="\n".join(lines))]

    # Only the document's own media (word/media/...), not e.g. docProps/thumbnail.jpeg.
    images = [p.blob for p in doc.part.package.iter_parts()
              if p.content_type.startswith("image/") and str(p.partname).startswith("/word/")]
    for blob in images:
        if len(pages) > s.max_pages:
            raise FileRejected("too_many_pages", f"more than {s.max_pages} embedded images")
        try:
            frames = decode_images(blob, max_frames=1)
        except FileRejected:
            continue  # EMF/WMF/SVG or broken image: nothing to OCR
        img = frames[0]
        if min(img.shape[:2]) < MIN_IMAGE_SIDE:
            continue
        pages.append(ocr_page(img).model_copy(update={"page": len(pages) + 1}))
    return pages
