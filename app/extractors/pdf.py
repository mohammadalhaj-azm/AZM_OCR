"""PDF: use the text layer per page; render + OCR pages with too little text. (pypdfium2, not PyMuPDF.)

With an OCR pool (OCR_ENGINES > 1) scanned pages are OCR'd in parallel. Rendering stays on the calling thread
(pdfium isn't thread-safe), and at most `parallel` rendered pages wait at a time, which bounds memory.
"""

import re
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pypdfium2 as pdfium

from app.config import Settings
from app.extractors.filetype import FileRejected
from app.extractors.image import pil_to_bgr
from app.schemas import PageResult

_NON_SPACE = re.compile(r"\S")


def render_pdf_pages(data: bytes, dpi: int, max_pages: int) -> list[np.ndarray]:
    """Every page rendered to a BGR image, text layer or not (handwriting scans often carry a junk text layer).
    Used by the handwriting endpoint; `extract_pdf` is unchanged."""
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise FileRejected("corrupt", "PDF could not be opened (corrupt or encrypted)") from e
    try:
        if len(pdf) > max_pages:
            raise FileRejected("too_many_pages", f"more than {max_pages} pages")
        images = []
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                bitmap = page.render(scale=dpi / 72)
                images.append(pil_to_bgr(bitmap.to_pil()))
                bitmap.close()
            finally:
                page.close()
        return images
    finally:
        pdf.close()


def extract_pdf(data: bytes, s: Settings, ocr_page: Callable[[np.ndarray], PageResult], parallel: int = 1,
                on_render: Callable[[np.ndarray], None] | None = None) -> list[PageResult]:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise FileRejected("corrupt", "PDF could not be opened (corrupt or encrypted)") from e
    try:
        if len(pdf) > s.max_pages:
            raise FileRejected("too_many_pages", f"more than {s.max_pages} pages")
        slots: list[PageResult | Future] = []
        in_flight: deque[Future] = deque()
        with ThreadPoolExecutor(parallel, thread_name_prefix="ocr-page") if parallel > 1 else nullcontext() as pool:
            for i in range(len(pdf)):
                page = pdf[i]
                try:
                    textpage = page.get_textpage()
                    text = textpage.get_text_bounded().replace("\r\n", "\n").replace("\r", "\n").strip()
                    textpage.close()
                    if len(_NON_SPACE.findall(text)) >= s.min_text_chars_per_page:
                        slots.append(PageResult(page=i + 1, method="text_layer", text=text))
                        continue
                    while len(in_flight) >= parallel:  # backpressure: don't render far ahead of OCR
                        in_flight.popleft().result()
                    bitmap = page.render(scale=s.ocr_dpi / 72)
                    img = pil_to_bgr(bitmap.to_pil())
                    bitmap.close()
                finally:
                    page.close()
                if on_render:
                    on_render(img)
                if pool is None:
                    slots.append(ocr_page(img))
                else:
                    fut = pool.submit(ocr_page, img)
                    in_flight.append(fut)
                    slots.append(fut)
            done = [x.result() if isinstance(x, Future) else x for x in slots]
        return [p.model_copy(update={"page": i + 1}) for i, p in enumerate(done)]
    finally:
        pdf.close()
