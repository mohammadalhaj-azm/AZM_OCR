"""Where on a page the handwriting model looks.

- `WholePageSegmenter` (`whole_page`): the whole page in one model call.
- `LinesSegmenter` (`lines`): the text lines found by the PaddleOCR detector (OCR_DET_MODEL, already used by the main
  OCR), one region per line, top to bottom. For models that read a single line per image.

The service transcribes each region and joins the texts with a newline, so neither the backends nor the API change
with the segmenter.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.config import Settings

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in page pixels


@dataclass(frozen=True)
class Region:
    image: np.ndarray  # BGR
    bbox: BBox | None = None  # None = the whole page


class Segmenter(Protocol):
    name: str

    def split(self, page: np.ndarray) -> list[Region]:
        """The regions of a page (BGR) in reading order."""
        ...


class WholePageSegmenter:
    name = "whole_page"

    def split(self, page: np.ndarray) -> list[Region]:
        return [Region(page)]


# --- lines -----------------------------------------------------------------------------------------------------------

LINE_OVERLAP = 0.5  # boxes on one line overlap vertically by at least this share of the smaller box's height
PAD_Y, PAD_X = 0.15, 0.3  # padding around a line, as a share of its height (handwriting strokes reach past the box)
MIN_PAD = 4  # pixels


def _vertical_overlap(a: BBox, b: BBox) -> float:
    inter = min(a[3], b[3]) - max(a[1], b[1])
    smaller = min(a[3] - a[1], b[3] - b[1])
    return inter / smaller if smaller > 0 else 0.0


def group_lines(boxes: list[BBox], min_overlap: float = LINE_OVERLAP) -> list[BBox]:
    """Merge text boxes into lines (a detector often splits a handwritten line into words or pieces): boxes that
    overlap vertically by at least `min_overlap` of the smaller height join the same line. Lines are returned top to
    bottom. Two columns side by side at the same height become one line."""
    lines: list[list[int]] = []
    for box in sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0])):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        best, best_overlap = None, min_overlap
        for line in lines:
            overlap = _vertical_overlap(box, tuple(line))
            if overlap >= best_overlap:
                best, best_overlap = line, overlap
        if best is None:
            lines.append(list(box))
        else:
            best[0], best[1] = min(best[0], box[0]), min(best[1], box[1])
            best[2], best[3] = max(best[2], box[2]), max(best[3], box[3])
    return [tuple(line) for line in sorted(lines, key=lambda b: (b[1], b[0]))]


def pad_box(box: BBox, width: int, height: int) -> BBox:
    h = box[3] - box[1]
    py, px = max(MIN_PAD, round(PAD_Y * h)), max(MIN_PAD, round(PAD_X * h))
    return max(0, box[0] - px), max(0, box[1] - py), min(width, box[2] + px), min(height, box[3] + py)


def poly_bbox(poly) -> BBox:
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return (int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min())),
            int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max())))


Detector = Callable[[np.ndarray], list]  # BGR image -> text polygons (each a list of (x, y) points)


def paddle_detector(s: Settings) -> Detector:
    """The PaddleOCR text detector the main OCR uses (its model is baked into the Docker image)."""
    from paddleocr import TextDetection

    from app.handwriting.backends import HandwritingUnavailable
    from app.ocr.engine import model_cache_dir

    path = model_cache_dir() / s.ocr_det_model
    if not (path / "inference.yml").exists():
        raise HandwritingUnavailable(f"the line detector ({s.ocr_det_model}) is not downloaded: "
                                     "uv run python -m app.cli download-models")
    common = {} if s.ocr_cpu_threads is None else {"cpu_threads": s.ocr_cpu_threads}
    det = TextDetection(model_name=s.ocr_det_model, model_dir=str(path), limit_side_len=s.ocr_det_limit_side_len,
                        limit_type=s.ocr_det_limit_type, **common)
    return lambda img: [p.tolist() for p in det.predict(img)[0]["dt_polys"]]


class LinesSegmenter:
    """One region per text line. The detector loads on first use; one page at a time (Paddle predictors aren't
    guaranteed thread-safe). Returns every line found: the service applies HANDWRITING_MAX_LINES."""
    name = "lines"

    def __init__(self, settings: Settings, detector: Detector | None = None):
        self.s = settings
        self._detect = detector
        self._lock = threading.Lock()

    def split(self, page: np.ndarray) -> list[Region]:
        import cv2

        height, width = page.shape[:2]
        scale = min(1.0, self.s.ocr_max_side / max(height, width))
        small = page if scale == 1.0 else cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        with self._lock:
            if self._detect is None:
                self._detect = paddle_detector(self.s)
            polys = self._detect(small)
        boxes = []
        for poly in polys:
            x0, y0, x1, y1 = poly_bbox(poly)
            boxes.append((int(x0 / scale), int(y0 / scale), int(np.ceil(x1 / scale)), int(np.ceil(y1 / scale))))
        regions = []
        for line in group_lines(boxes):
            x0, y0, x1, y1 = pad_box(line, width, height)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                regions.append(Region(page[y0:y1, x0:x1], (x0, y0, x1, y1)))
        return regions


SEGMENTERS: dict[str, Callable[[Settings], Segmenter]] = {
    WholePageSegmenter.name: lambda s: WholePageSegmenter(),
    LinesSegmenter.name: LinesSegmenter,
}


def make_segmenter(name: str, settings: Settings | None = None) -> Segmenter:
    try:
        factory = SEGMENTERS[name]
    except KeyError:
        raise ValueError(f"unknown handwriting segmenter {name!r}") from None
    return factory(settings or Settings(_env_file=None))
