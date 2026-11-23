"""Handwriting OCR metrics (pure functions): raw and normalized CER/WER, the gap between them, and invented text.

Two text views, always reported side by side, never one instead of the other:

- raw: Unicode NFC and whitespace only (every run of spaces / newlines becomes one space; line breaks are layout,
  not recognition). Everything else counts as an error.
- normalized: raw plus the fixed rules in NORMALIZATION_RULES, which remove only marks a transcriber may or may
  not write (tatweel, tashkeel) and unify digit / punctuation code points. Letters are NEVER folded: أ/إ/آ/ا,
  ى/ي, ة/ه and the hamza carriers stay distinct, because confusing them is a real recognition (spelling) error.

A large raw - normalized gap means the normalized score hides differences; such samples are flagged for review.
"""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

CATEGORIES = ("ar_handwritten", "en_handwritten", "mixed", "blank", "signature_only", "stamp_only", "sparse",
              "printed_control")
NO_TEXT_CATEGORIES = ("blank", "signature_only", "stamp_only")  # invented text is measured here
GAP_FLAG = 0.05  # raw CER - normalized CER above this (5 points) is flagged

_WS = re.compile(r"\s+")
_TATWEEL = "ـ"
# Tashkeel: fathatan..sukun (U+064B-0652), maddah/hamza above/below marks (U+0653-0655), superscript alef (U+0670).
# Not the hamza letters (ء أ إ ؤ ئ): those are spelling.
_TASHKEEL = re.compile("[ً-ٰٕ]")
_DIGITS = {**{0x0660 + i: str(i) for i in range(10)}, **{0x06F0 + i: str(i) for i in range(10)}}
_PUNCT = {"،": ",", "؛": ";", "؟": "?", "٪": "%", "٫": ".", "٬": ",",
          "«": '"', "»": '"', "“": '"', "”": '"', "‘": "'", "’": "'"}
_PUNCT_MAP = str.maketrans({**{chr(k): v for k, v in _DIGITS.items()}, **_PUNCT})

NORMALIZATION_RULES = (
    "Unicode NFC; whitespace runs -> one space",
    "remove tatweel (U+0640)",
    "remove tashkeel U+064B-U+0655 and superscript alef U+0670",
    "Arabic-Indic and Persian digits -> 0-9",
    "Arabic comma/semicolon/question mark/percent/decimal/thousands separators and curly/angle quotes -> ASCII",
    "NOT folded (counted as errors): alef forms, alef maqsura/yeh, ta marbuta/heh, hamza carriers",
)


def raw_view(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text or "")).strip()


def normalized_view(text: str) -> str:
    text = raw_view(text).replace(_TATWEEL, "")
    text = _TASHKEEL.sub("", text).translate(_PUNCT_MAP)
    return _WS.sub(" ", text).strip()


@dataclass(frozen=True)
class Edits:
    substitutions: int = 0
    deletions: int = 0  # reference units the hypothesis left out
    insertions: int = 0  # hypothesis units not in the reference (invented / extra)

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def edit_ops(ref: Sequence, hyp: Sequence) -> Edits:
    """Levenshtein alignment of two sequences (characters or words), with its operation counts."""
    # each cell: (cost, subs, dels, ins); two rows only
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        r = ref[i - 1]
        for j in range(1, len(hyp) + 1):
            d = prev[j - 1]
            if r == hyp[j - 1]:
                best = d
            else:
                best = (d[0] + 1, d[1] + 1, d[2], d[3])
            u = prev[j]  # ref unit deleted
            if u[0] + 1 < best[0]:
                best = (u[0] + 1, u[1], u[2] + 1, u[3])
            left = cur[j - 1]  # hyp unit inserted
            if left[0] + 1 < best[0]:
                best = (left[0] + 1, left[1], left[2], left[3] + 1)
            cur.append(best)
        prev = cur
    _, s, d, n = prev[-1]
    return Edits(s, d, n)


def rate(edits: int, ref_len: int) -> float | None:
    """Error rate; None when the reference is empty (undefined: see the invented-text figures instead)."""
    return edits / ref_len if ref_len else None


@dataclass
class Score:
    ref_chars: int
    hyp_chars: int
    raw_char: Edits
    raw_word: Edits
    norm_char: Edits
    norm_word: Edits
    ref_words: int
    norm_ref_chars: int
    norm_ref_words: int

    @property
    def raw_cer(self) -> float | None:
        return rate(self.raw_char.total, self.ref_chars)

    @property
    def raw_wer(self) -> float | None:
        return rate(self.raw_word.total, self.ref_words)

    @property
    def norm_cer(self) -> float | None:
        return rate(self.norm_char.total, self.norm_ref_chars)

    @property
    def norm_wer(self) -> float | None:
        return rate(self.norm_word.total, self.norm_ref_words)

    @property
    def gap(self) -> float | None:
        """raw CER - normalized CER: how much the normalization forgives."""
        return None if self.raw_cer is None or self.norm_cer is None else self.raw_cer - self.norm_cer

    @property
    def gap_flag(self) -> bool:
        return self.gap is not None and self.gap > GAP_FLAG

    @property
    def inserted_share(self) -> float | None:
        """Share of the output's characters that are not in the reference (raw view)."""
        return self.raw_char.insertions / self.hyp_chars if self.hyp_chars else None


def score(reference: str, hypothesis: str) -> Score:
    ref, hyp = raw_view(reference), raw_view(hypothesis)
    nref, nhyp = normalized_view(reference), normalized_view(hypothesis)
    return Score(
        ref_chars=len(ref), hyp_chars=len(hyp),
        raw_char=edit_ops(ref, hyp), raw_word=edit_ops(ref.split(), hyp.split()),
        norm_char=edit_ops(nref, nhyp), norm_word=edit_ops(nref.split(), nhyp.split()),
        ref_words=len(ref.split()), norm_ref_chars=len(nref), norm_ref_words=len(nref.split()),
    )


@dataclass
class Sample:
    file: str
    category: str
    score: Score | None = None  # None when the file failed
    pages: int = 0
    skipped_pages: int = 0  # pages the blank guard did not send to the model
    warnings: list[str] = field(default_factory=list)
    ms: int = 0
    error: str = ""


def _pooled(samples: list[Sample], edits_of, ref_len_of) -> float | None:
    """Corpus-level rate: all edits / all reference units (long samples weigh more, as they should)."""
    scored = [s.score for s in samples if s.score is not None]
    total_ref = sum(ref_len_of(sc) for sc in scored)
    return sum(edits_of(sc).total for sc in scored) / total_ref if total_ref else None


def category_summary(samples: list[Sample]) -> dict:
    ok = [s for s in samples if s.score is not None]
    raw_cer = _pooled(ok, lambda sc: sc.raw_char, lambda sc: sc.ref_chars)
    norm_cer = _pooled(ok, lambda sc: sc.norm_char, lambda sc: sc.norm_ref_chars)
    hyp_chars = sum(s.score.hyp_chars for s in ok)
    inserted = sum(s.score.raw_char.insertions for s in ok)
    return {
        "n": len(samples), "errors": len(samples) - len(ok),
        "rawCer": raw_cer, "rawWer": _pooled(ok, lambda sc: sc.raw_word, lambda sc: sc.ref_words),
        "normCer": norm_cer, "normWer": _pooled(ok, lambda sc: sc.norm_word, lambda sc: sc.norm_ref_words),
        "gap": None if raw_cer is None or norm_cer is None else raw_cer - norm_cer,
        "gapFlagged": sum(1 for s in ok if s.score.gap_flag),
        "outputChars": hyp_chars,
        "insertedShare": inserted / hyp_chars if hyp_chars else None,
        "samplesWithOutput": sum(1 for s in ok if s.score.hyp_chars > 0),
        "pages": sum(s.pages for s in samples), "skippedPages": sum(s.skipped_pages for s in samples),
        "warnings": sum(1 for s in samples if s.warnings),
        "msPerPage": (sum(s.ms for s in samples) / sum(s.pages for s in samples)
                      if sum(s.pages for s in samples) else None),
    }


def invented_text(samples: list[Sample]) -> dict:
    """For pages that should yield no (or almost no) text: how much the model produced anyway."""
    out = {}
    for cat in NO_TEXT_CATEGORIES:
        rows = [s for s in samples if s.category == cat and s.score is not None]
        if not rows:
            continue
        excess = [max(0, s.score.hyp_chars - s.score.ref_chars) for s in rows]
        out[cat] = {"n": len(rows), "withOutput": sum(1 for s in rows if s.score.hyp_chars > 0),
                    "excessChars": sum(excess), "maxExcessChars": max(excess),
                    "skippedByGuard": sum(1 for s in rows if s.pages and s.skipped_pages == s.pages)}
    return out
