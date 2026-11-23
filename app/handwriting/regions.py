"""Where on a page the handwriting model looks: the seam for line / region cropping.

V1 sends the whole page in one call (`WholePageSegmenter`). A later segmenter (e.g. PaddleOCR detection boxes grouped
into lines, or a layout model) returns several regions in reading order; the service transcribes each one and joins
the texts, so neither the backends nor the API change.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Region:
    image: np.ndarray  # BGR
    bbox: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1) in page pixels; None = the whole page


class Segmenter(Protocol):
    name: str

    def split(self, page: np.ndarray) -> list[Region]:
        """The regions of a page (BGR) in reading order."""
        ...


class WholePageSegmenter:
    name = "whole_page"

    def split(self, page: np.ndarray) -> list[Region]:
        return [Region(page)]


SEGMENTERS: dict[str, type] = {WholePageSegmenter.name: WholePageSegmenter}


def make_segmenter(name: str) -> Segmenter:
    try:
        return SEGMENTERS[name]()
    except KeyError:
        raise ValueError(f"unknown handwriting segmenter {name!r}") from None
