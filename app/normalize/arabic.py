"""Arabic text normalization (pure functions, no I/O).

clean_text(text)      -> `text`: the original, cleaned. NFKC, digits -> 0-9, no tatweel or bidi
                         control chars, reversed PDF lines fixed, tidy whitespace. Tashkeel is kept.
normalize_for_match() -> `normalizedText`: additionally strips tashkeel and unifies letter variants.
"""

import re
import unicodedata

# Arabic-Indic (U+0660-0669) and Extended Arabic-Indic / Persian (U+06F0-06F9) digits, plus Arabic
# decimal/thousands separators and percent sign.
_DIGITS = {**{0x0660 + i: str(i) for i in range(10)}, **{0x06F0 + i: str(i) for i in range(10)},
           0x066B: ".", 0x066C: ",", 0x066A: "%"}
_TATWEEL = "ـ"
# Invisible bidi/format controls: LRM, RLM, ALM, LRE..RLO, LRI..PDI, ZWSP, BOM.
_CONTROLS = re.compile("[​‎‏؜‪-‮⁦-⁩﻿]")
# Tashkeel: harakat/tanwin/shadda/sukun etc., superscript alef, Quranic annotation marks.
_TASHKEEL = re.compile("[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭ]")
_MATCH_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",  # alef variants
    "ى": "ي",  # alef maqsura
    "ة": "ه",  # ta marbuta
    "ی": "ي", "ک": "ك",  # Persian yeh / keheh, common in OCR and Persian keyboards
    "،": ",", "؛": ";", "؟": "?",
    "‌": "", "‍": "",  # ZWNJ / ZWJ
})

_ARABIC_LETTER = re.compile("[ؠ-يٮ-ۓۺ-ۿݐ-ݿﭐ-﷿ﹰ-ﻼ]")
_LATIN_LETTER = re.compile("[A-Za-z]")
_ARABIC_WORD = re.compile("[ؠ-يٱ-ۓۺ-ۿݐ-ݿﭐ-﷿ﹰ-ﻼ]+")
_LTR_RUN = re.compile(r"[A-Za-z0-9٠-٩۰-۹][A-Za-z0-9٠-٩۰-۹.,:/%+\-]*"
                      r"[A-Za-z0-9٠-٩۰-۹]|[A-Za-z0-9٠-٩۰-۹]")
_MIRROR = str.maketrans("()[]{}<>«»", ")(][}{><»«")
_H_SPACE = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


# --- reversed-line detection -------------------------------------------------------------

def _form(ch: str) -> str | None:
    """'initial' / 'medial' / 'final' / 'isolated' for Arabic presentation-form characters."""
    name = unicodedata.name(ch, "")
    for form in ("INITIAL", "MEDIAL", "FINAL", "ISOLATED"):
        if name.endswith(f"{form} FORM"):
            return form.lower()
    return None


def _direction_scores(line: str) -> tuple[int, int]:
    """(forward, reversed) evidence for Arabic words in one line (raw text, before NFKC)."""
    fwd = rev = 0
    for word in _ARABIC_WORD.findall(line):
        first, last = _form(word[0]), _form(word[-1])
        if len(word) > 1 and (first or last):  # presentation forms: positional shapes are decisive
            fwd += 2 * (first == "initial") + 2 * (last == "final")
            rev += 2 * (first == "final") + 2 * (last == "initial")
        w = unicodedata.normalize("NFKC", word)
        # ة and ى only end words; "ال" (definite article) starts many words.
        fwd += 2 * (w[-1] in "ةى") + (len(w) > 2 and w.startswith("ال"))
        rev += 2 * (w[0] in "ةى") + (len(w) > 2 and w.endswith("لا"))
    return fwd, rev


def _says_reversed(fwd: int, rev: int) -> bool:
    return rev >= 2 and rev > fwd


def reverse_visual_line(line: str) -> str:
    """Visual (left-to-right glyph order) -> logical order for an RTL line.

    Reverse everything, then restore left-to-right runs (Latin words, numbers, dates) and mirror brackets.
    """
    rev = line[::-1].translate(_MIRROR)
    return _LTR_RUN.sub(lambda m: m.group()[::-1], rev)


def is_reversed_line(line: str) -> bool:
    return _says_reversed(*_direction_scores(line))


def fix_reversed_lines(text: str) -> str:
    """Fix lines stored in visual order.

    A PDF that stores text reversed usually does so for the whole page, so page-level evidence
    decides for short lines with no cues of their own. A line's own clear evidence always wins.
    """
    lines = text.split("\n")
    scores = [_direction_scores(ln) for ln in lines]
    page_reversed = _says_reversed(sum(f for f, _ in scores), sum(r for _, r in scores))
    out = []
    for ln, (fwd, rev) in zip(lines, scores):
        if _says_reversed(fwd, rev):
            flip = True
        elif page_reversed:
            flip = fwd <= rev and bool(_ARABIC_WORD.search(ln))
        else:
            flip = False
        out.append(reverse_visual_line(ln) if flip else ln)
    return "\n".join(out)


# --- public API ----------------------------------------------------------------------------

def clean_text(text: str, fix_reversed: bool = True) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROLS.sub("", text)
    if fix_reversed:  # before NFKC: needs presentation forms, and keeps lam-alef ligatures intact
        text = fix_reversed_lines(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DIGITS).replace(_TATWEEL, "")
    lines = [_H_SPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def normalize_for_match(text: str) -> str:
    """For matching only: strip tashkeel, unify alef/yeh/ta-marbuta variants. Expects clean_text output."""
    text = _TASHKEEL.sub("", text).translate(_MATCH_MAP)
    lines = [_H_SPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    return _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()


def detect_languages(text: str, min_letters: int = 3) -> list[str]:
    """'ar' and/or 'en' by letter counts (Latin script is reported as 'en')."""
    counts = {"ar": len(_ARABIC_LETTER.findall(text)), "en": len(_LATIN_LETTER.findall(text))}
    return [lang for lang, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= min_letters]
