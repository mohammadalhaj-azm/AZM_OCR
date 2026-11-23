"""Pure helpers: merge Arabic/English recognitions per box, and order boxes into lines.

Measured behaviour of the PP-OCRv5 mobile recognizers on the same crop:
- pure Arabic line: Arabic model reads it; English model returns "" (score 0).
- pure Latin line: both read it well.
- mixed line (e.g. "الاسم سامي  Name SAMPLE"): each model reads ONLY its own script.
So a plain max-confidence pick would drop the Arabic part; we combine instead.
"""

import re
from dataclasses import dataclass

_ARABIC_LETTER = re.compile(r"[ؠ-يٮ-ۓۺ-ۿݐ-ݿﭐ-﷿ﹰ-ﻼ]")
_LATIN_ALNUM = re.compile(r"[A-Za-z0-9]")
_LATIN_RUN = re.compile(r"[A-Za-z0-9<][A-Za-z0-9<:/.,\-]*")
_SPACES = re.compile(r"\s+")


@dataclass
class Rec:
    text: str
    score: float


@dataclass
class Box:
    poly: list[list[float]]  # 4 points [x, y]
    text: str
    score: float

    @property
    def y_range(self) -> tuple[float, float]:
        ys = [p[1] for p in self.poly]
        return min(ys), max(ys)

    @property
    def x_center(self) -> float:
        return sum(p[0] for p in self.poly) / len(self.poly)


def count_arabic(text: str) -> int:
    return len(_ARABIC_LETTER.findall(text))


def count_latin(text: str) -> int:
    return len(_LATIN_ALNUM.findall(text))


def _compact(text: str) -> str:
    return _SPACES.sub("", text)


def merge_recognitions(ar: Rec, en: Rec | None, min_score: float) -> Rec | None:
    ar_ok = bool(ar.text.strip()) and ar.score >= min_score
    en_ok = en is not None and count_latin(en.text) >= 2 and en.score >= min_score
    if not en_ok:
        return ar if ar_ok else None
    if not ar_ok:
        return en
    if count_arabic(ar.text) == 0:  # Latin-only box: both models read it; take the more confident
        return ar if ar.score > en.score else en
    if _compact(en.text) in _compact(ar.text):  # Arabic model already has the Latin part
        return ar
    # Mixed box: Arabic part from the Arabic model, Latin part from the English model.
    # Arabic first = logical (reading) order for an RTL line with a trailing LTR run.
    arabic_part = _SPACES.sub(" ", _LATIN_RUN.sub(" ", ar.text)).strip()
    text = f"{arabic_part} {en.text.strip()}".strip()
    la, le = len(arabic_part), len(en.text.strip())
    score = (ar.score * la + en.score * le) / max(la + le, 1)
    return Rec(text, score)


def group_lines(boxes: list[Box]) -> list[list[Box]]:
    """Group boxes into lines (top to bottom). RTL lines are ordered right-to-left."""
    lines: list[list[Box]] = []
    for box in sorted(boxes, key=lambda b: sum(b.y_range) / 2):
        top, bottom = box.y_range
        if lines:
            ltop = min(b.y_range[0] for b in lines[-1])
            lbottom = max(b.y_range[1] for b in lines[-1])
            overlap = min(bottom, lbottom) - max(top, ltop)
            if overlap >= 0.5 * min(bottom - top, lbottom - ltop):
                lines[-1].append(box)
                continue
        lines.append([box])
    for line in lines:
        joined = " ".join(b.text for b in line)
        rtl = count_arabic(joined) >= count_latin(joined)
        line.sort(key=lambda b: b.x_center, reverse=rtl)
    return lines


def page_text_and_confidence(boxes: list[Box]) -> tuple[str, float | None]:
    if not boxes:
        return "", None
    text = "\n".join(" ".join(b.text for b in line) for line in group_lines(boxes))
    total = sum(len(b.text) for b in boxes)
    conf = sum(b.score * len(b.text) for b in boxes) / max(total, 1)
    return text, round(conf, 4)
