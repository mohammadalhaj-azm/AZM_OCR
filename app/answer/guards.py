"""Checks that never trust the LLM (pure functions, no I/O).

- `mask` / `unmask`: protected values (numbers, dates, amounts, IDs, e-mails) are replaced by placeholders such as
  ⟦P1⟧ before the text goes to the LLM, and put back afterwards, so the LLM cannot change a single digit.
  `check_placeholders` finds the replies where that went wrong (a placeholder dropped, duplicated or made up, or
  digits written by the model itself); such a chunk keeps its original text and is flagged.
- `located`: is a quote (from a review item or an answer's citation) really in the text?
- `word_changes`: what the correction changed, word by word, computed here (not reported by the model).
"""

import difflib
import re

from app.normalize import clean_text, normalize_for_match

_D = "0-9٠-٩۰-۹"  # Western, Arabic-Indic and Extended Arabic-Indic digits
PROTECTED = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"  # e-mail
    r"|(?<![\w])(?=[A-Za-z0-9-]*[0-9])(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9](?![\w])"  # A1234567
    rf"|[{_D}](?:[{_D}.,:/\-٫٬]*[{_D}])?"  # 1,250.00  12/05/2024  050-123-4567  ١٢٣
)
PLACEHOLDER = re.compile(r"⟦P(\d+)⟧")
_ANY_DIGIT = re.compile(f"[{_D}]")
LOW_OPEN, LOW_CLOSE = "⟪", "⟫"  # marks low-confidence text for the LLM


def mask(text: str) -> tuple[str, list[str]]:
    """(text with ⟦P1⟧, ⟦P2⟧, ... for the protected values, the values in order)."""
    values: list[str] = []

    def repl(m: re.Match) -> str:
        values.append(m.group(0))
        return f"⟦P{len(values)}⟧"

    return PROTECTED.sub(repl, text), values


def unmask(text: str, values: list[str]) -> str:
    """Put the values back. Unknown placeholders are left as they are (check_placeholders reports them)."""
    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        return values[i - 1] if 1 <= i <= len(values) else m.group(0)

    return PLACEHOLDER.sub(repl, text)


def check_placeholders(reply: str, values: list[str]) -> str | None:
    """Why the model's (still masked) reply can't be used safely, or None when every protected value comes back
    exactly once and the model wrote no digits of its own."""
    found = [int(i) for i in PLACEHOLDER.findall(reply)]
    expected = set(range(1, len(values) + 1))
    if any(i not in expected for i in found):
        return "the correction referred to a protected value that does not exist"
    if len(found) != len(set(found)):
        return "the correction repeated a protected value"
    if missing := expected - set(found):
        return f"the correction dropped {len(missing)} protected value(s) (numbers, dates, IDs)"
    if _ANY_DIGIT.search(PLACEHOLDER.sub("", reply)):
        return "the correction added digits that are not in the original text"
    return None


def strip_low_marks(text: str) -> str:
    return text.replace(LOW_OPEN, "").replace(LOW_CLOSE, "")


def _key(text: str) -> str:
    """For comparing only: NFKC, digits unified (١٢٣ -> 123), Arabic letter variants and tashkeel, case, spacing."""
    return " ".join(normalize_for_match(clean_text(strip_low_marks(text), fix_reversed=False)).split()).casefold()


def located(quote: str, *texts: str) -> bool:
    """The quote occurs in one of the texts (Arabic letter variants, tashkeel, case and whitespace ignored)."""
    q = _key(quote)
    return bool(q) and any(q in _key(t) for t in texts if t)


def overlaps(a: str, b: str) -> bool:
    """One quote contains the other (normalized like `located`)."""
    ka, kb = _key(a), _key(b)
    return bool(ka and kb) and (ka in kb or kb in ka)


def protected_values(text: str) -> set[str]:
    """The protected values of a text, digits unified (١٢٣ = 123), for comparing two readings."""
    return {_key(m.group(0)) for m in PROTECTED.finditer(text)}


def word_changes(original: str, corrected: str, limit: int = 200) -> list[dict]:
    """[{original, corrected}] for every changed run of words (an empty side = inserted / deleted words)."""
    a, b = original.split(), corrected.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op != "equal":
            out.append({"original": " ".join(a[i1:i2]), "corrected": " ".join(b[j1:j2])})
            if len(out) >= limit:
                break
    return out


def chunks(text: str, max_chars: int) -> list[str]:
    """Split at line breaks into pieces of at most max_chars (a single longer line is split at spaces, then hard)."""
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    cur = ""
    for line in text.split("\n"):
        pieces = [line]
        if len(line) > max_chars:
            pieces, rest = [], line
            while len(rest) > max_chars:
                cut = rest.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                pieces.append(rest[:cut])
                rest = rest[cut:].lstrip(" ")
            pieces.append(rest)
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > max_chars:
                out.append(cur)
                cur = piece
            else:
                cur = f"{cur}\n{piece}" if cur else piece
    if cur:
        out.append(cur)
    return out
