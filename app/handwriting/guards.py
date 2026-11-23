"""Guards against invented text. They only skip or flag, never rewrite what the model read.

- `ink_ratio` / `is_blank`: a page with (almost) no ink is not sent to the model at all. The threshold is low on
  purpose: a signature, a stamp or a single short word still goes to the model (the eval measures what it does
  with them), because a stricter heuristic would also drop real, short handwriting.
- `repetition_loop`: greedy decoding can get stuck repeating a phrase; the page is flagged so a reviewer sees it.
"""

import re

import cv2
import numpy as np

_ANALYSIS_SIDE = 1000  # pages are measured at this size, so the thresholds don't depend on the DPI
_MARGIN = 0.03  # scan edges (shadows, punch holes, borders) are ignored
_MIN_CONTRAST = 50  # ink must be this much darker (0-255) than the paper
_MIN_BLOB_PX = 6  # connected ink smaller than this (at _ANALYSIS_SIDE) is scanner noise


def ink_ratio(img_bgr: np.ndarray) -> float:
    """Share of the page (without margins) covered by ink: dark-on-paper pixels in blobs larger than specks."""
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if max(h, w) > _ANALYSIS_SIDE:
        f = _ANALYSIS_SIDE / max(h, w)
        gray = cv2.resize(gray, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]
    mh, mw = int(h * _MARGIN), int(w * _MARGIN)
    gray = gray[mh:h - mh or None, mw:w - mw or None]
    if gray.size == 0:
        return 0.0
    paper = float(np.median(gray))
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu alone splits even a blank page's noise in two; the contrast floor keeps paper texture out
    ink = (gray < min(otsu, paper - _MIN_CONTRAST)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.empty(0)
    return float(areas[areas >= _MIN_BLOB_PX].sum()) / gray.size


def is_blank(img_bgr: np.ndarray, min_ink_ratio: float) -> bool:
    return ink_ratio(img_bgr) < min_ink_ratio


_PUNCT_ONLY = re.compile(r"^[\W_]+$")


def _tail_repeats(tokens: list[str], n: int) -> int:
    """How many times the last n-gram repeats back to back at the end (allowing a cut-off last copy)."""
    best = 0
    for cut in range(n):  # the final copy may be incomplete when generation stopped mid-loop
        end = len(tokens) - cut
        unit = tokens[end - n:end]
        if len(unit) < n:
            break
        reps, i = 1, end - n
        while i - n >= 0 and tokens[i - n:i] == unit:
            reps += 1
            i -= n
        best = max(best, reps)
    return best


def repetition_loop(text: str) -> bool:
    """True when the text looks like a decoding loop: a phrase repeated many times back to back at the end, the
    same line 4+ times in a row, or one letter 30+ times. Dotted form lines (".....") don't count."""
    tokens = text.split()
    for n in range(1, 11):
        if len(tokens) >= 2 * n and _tail_repeats(tokens, n) >= max(4, -(-12 // n)):
            if not all(_PUNCT_ONLY.match(t) for t in tokens[-n:]):
                return True
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    run = 1
    for prev, cur in zip(lines, lines[1:]):
        run = run + 1 if cur == prev and not _PUNCT_ONLY.match(cur) else 1
        if run >= 4:
            return True
    return re.search(r"([^\W\d_])\1{29,}", text) is not None
