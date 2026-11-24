"""PaddleOCR (PP-OCRv5) engine: detect once, recognize each box with the Arabic and English models.

`OcrPool` holds OCR_ENGINES independent engines, so that several pages (of one PDF, or of different requests)
are OCR'd at the same time. Each engine is used by one thread at a time.
"""

import logging
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.config import Settings
from app.ocr.merge import Box, Rec, group_lines, merge_recognitions, page_text_and_confidence
from app.ocr.preprocess import preprocess_with_transform

# Runtime must not contact model hosters. Must be set before paddlex is imported.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

log = logging.getLogger(__name__)


class ModelsMissing(RuntimeError):
    pass


@dataclass
class OcrOutput:
    text: str
    confidence: float | None
    boxes: list[Box]
    # Reading-order lines: {"text", "confidence", "bbox": [x0, y0, x1, y1]}, bbox normalized to 0..1 of the input
    # image (independent of OCR downscaling / deskew), so it maps onto the page at any resolution.
    lines: list[dict] = field(default_factory=list)
    width: int = 0  # input image size in pixels
    height: int = 0


def available_cpus() -> int:
    """CPUs this process may use (respects CPU affinity / cpusets; set OCR_CPU_THREADS under a CPU quota)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows, macOS
        return os.cpu_count() or 1


def model_cache_dir() -> Path:
    return Path(os.environ.get("PADDLE_PDX_CACHE_HOME", Path.home() / ".paddlex")) / "official_models"


def _crop(img: np.ndarray, poly) -> np.ndarray | None:
    """Perspective-crop the minimum-area rectangle around a detected polygon."""
    pts = cv2.boxPoints(cv2.minAreaRect(np.asarray(poly, np.float32)))
    pts = sorted(pts.tolist(), key=lambda p: p[0])
    left, right = sorted(pts[:2], key=lambda p: p[1]), sorted(pts[2:], key=lambda p: p[1])
    box = np.float32([left[0], right[0], right[1], left[1]])  # tl, tr, br, bl
    w = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    h = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))
    if w < 4 or h < 4:
        return None
    m = cv2.getPerspectiveTransform(box, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    crop = cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if h / w >= 1.5:  # vertical text box
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return crop


def line_items(boxes: list[Box], scale: float, affine: np.ndarray | None, width: int, height: int) -> list[dict]:
    """Lines with bboxes mapped back to input-image coordinates (undo deskew, then downscale), normalized."""
    inv = cv2.invertAffineTransform(affine) if affine is not None else None
    out = []
    for line in group_lines(boxes):
        pts = np.asarray([p for b in line for p in b.poly], np.float64)
        if inv is not None:
            pts = pts @ inv[:, :2].T + inv[:, 2]
        pts /= scale
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        n = sum(len(b.text) for b in line)
        conf = sum(b.score * len(b.text) for b in line) / max(n, 1)
        out.append({"text": " ".join(b.text for b in line), "confidence": round(conf, 4),
                    "bbox": [round(min(max(v / d, 0.0), 1.0), 4)
                             for v, d in ((x0, width), (y0, height), (x1, width), (y1, height))]})
    return out


class OcrEngine:
    """Loads models lazily on first use (PDFs with a text layer never need them)."""

    def __init__(self, settings: Settings, allow_download: bool | None = None):
        self.s = settings
        self.allow_download = settings.ocr_allow_download if allow_download is None else allow_download
        self._det = self._rec_ar = self._rec_en = None
        self._rec_pool: ThreadPoolExecutor | None = None
        self.device: str | None = None  # e.g. "cpu", "gpu:0"; set when the models load
        # Paddle predictors aren't guaranteed thread-safe: one image at a time per engine (use OcrPool for more).
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._det is not None

    def _model_dir(self, name: str) -> str | None:
        if self.allow_download:
            return None  # PaddleX uses its cache, downloading if needed
        path = model_cache_dir() / name
        if not (path / "inference.yml").exists():
            raise ModelsMissing(f"OCR model {name} not found in {path.parent}; "
                                "run `uv run python -m app.cli download-models` once (needs internet)")
        return str(path)

    def models_present(self) -> bool:
        names = [self.s.ocr_det_model, self.s.ocr_rec_model_ar, self.s.ocr_rec_model_en]
        return all((model_cache_dir() / n / "inference.yml").exists() for n in names if n)

    def load(self) -> "OcrEngine":
        with self._lock:
            return self._load()

    def _load(self) -> "OcrEngine":
        if self._det is not None:
            return self
        from paddleocr import TextDetection, TextRecognition
        from paddlex.utils.device import get_default_device

        # OCR_DEVICE unset: GPU 0 if PaddlePaddle is the CUDA build and sees a GPU, else the CPU (PaddleOCR's own
        # default, resolved here so that it can be reported).
        device = self.s.ocr_device or get_default_device()
        common = {"device": device}
        if self.s.ocr_cpu_threads is not None:
            common["cpu_threads"] = self.s.ocr_cpu_threads
        self._det = TextDetection(
            model_name=self.s.ocr_det_model, model_dir=self._model_dir(self.s.ocr_det_model),
            limit_side_len=self.s.ocr_det_limit_side_len, limit_type=self.s.ocr_det_limit_type, **common)
        self._rec_ar = TextRecognition(
            model_name=self.s.ocr_rec_model_ar, model_dir=self._model_dir(self.s.ocr_rec_model_ar), **common)
        if self.s.ocr_rec_model_en:
            self._rec_en = TextRecognition(
                model_name=self.s.ocr_rec_model_en, model_dir=self._model_dir(self.s.ocr_rec_model_en), **common)
            if self.s.ocr_parallel_rec:
                self._rec_pool = ThreadPoolExecutor(1, thread_name_prefix="ocr-rec-en")
        self.device = device
        log.info("OCR models loaded on %s", device)
        return self

    def _recognize(self, rec, crops: list[np.ndarray]) -> list[Rec]:
        return [Rec(r["rec_text"], float(r["rec_score"]))
                for r in rec.predict(crops, batch_size=self.s.ocr_rec_batch_size)]

    def recognize(self, img: np.ndarray) -> OcrOutput:
        """OCR one BGR image."""
        with self._lock:
            return self._recognize_locked(img)

    def _recognize_locked(self, img: np.ndarray) -> OcrOutput:
        self._load()
        height, width = img.shape[:2]
        scale = 1.0
        if max(height, width) > self.s.ocr_max_side:
            scale = self.s.ocr_max_side / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        affine = None
        if self.s.ocr_preprocess:
            img, affine = preprocess_with_transform(img)

        polys = [p.tolist() for p in self._det.predict(img)[0]["dt_polys"]]
        crops, kept = [], []
        for poly in polys:
            c = _crop(img, poly)
            if c is not None:
                crops.append(c)
                kept.append(poly)
        if not crops:
            return OcrOutput("", None, [], [], width, height)

        if self._rec_pool is not None:  # OCR_PARALLEL_REC: both recognizers at the same time
            en_future = self._rec_pool.submit(self._recognize, self._rec_en, crops)
            ar = self._recognize(self._rec_ar, crops)
            en = en_future.result()
        else:
            ar = self._recognize(self._rec_ar, crops)
            en = self._recognize(self._rec_en, crops) if self._rec_en else [None] * len(crops)
        boxes = []
        for poly, a, e in zip(kept, ar, en):
            merged = merge_recognitions(a, e, self.s.ocr_min_score)
            if merged is not None:
                boxes.append(Box(poly, merged.text, merged.score))
        text, conf = page_text_and_confidence(boxes)
        return OcrOutput(text, conf, boxes, line_items(boxes, scale, affine, width, height), width, height)


class OcrPool:
    """OCR_ENGINES independent engines behind the `OcrEngine` interface. `size` = how many images at once."""

    def __init__(self, settings: Settings, size: int | None = None):
        self.s = settings
        self.size = max(1, size if size is not None else settings.ocr_engines)
        per_engine = settings
        if settings.ocr_cpu_threads is None and self.size > 1:  # split the cores instead of oversubscribing them
            per_engine = settings.model_copy(update={"ocr_cpu_threads": max(1, available_cpus() // self.size)})
        self.engines = [OcrEngine(per_engine) for _ in range(self.size)]
        self._free: queue.Queue[OcrEngine] = queue.Queue()
        for e in self.engines:
            self._free.put(e)

    def load(self) -> "OcrPool":
        for e in self.engines:
            e.load()
        return self

    def models_present(self) -> bool:
        return self.engines[0].models_present()

    def stats(self) -> dict:
        """Engines in the pool, loaded (models in RAM) and busy right now (admin dashboard)."""
        return {"engines": self.size, "loaded": sum(e.loaded for e in self.engines),
                "busy": max(0, self.size - self._free.qsize()), "cpuThreads": self.engines[0].s.ocr_cpu_threads,
                "device": next((e.device for e in self.engines if e.device), None),
                "parallelRec": self.s.ocr_parallel_rec}

    def recognize(self, img: np.ndarray) -> OcrOutput:
        engine = self._free.get()
        try:
            return engine.recognize(img)
        finally:
            self._free.put(engine)
