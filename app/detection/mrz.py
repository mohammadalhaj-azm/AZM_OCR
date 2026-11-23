"""Machine-readable zone (ICAO 9303) detection and check-digit validation: TD1, TD2, TD3.

Tolerates common OCR noise: spaces, lower case, '«' for '<<', O/I/S/B in numeric fields,
and a missing or extra '<' filler.
"""

import re
from dataclasses import dataclass, field
from datetime import date

FORMATS = {"TD1": (3, 30), "TD2": (2, 36), "TD3": (2, 44)}
_MRZ_CHARS = re.compile(r"^[A-Z0-9<]+$")
_DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})


def check_digit(field_: str) -> str:
    total = 0
    for i, ch in enumerate(field_):
        if ch.isdigit():
            v = int(ch)
        elif "A" <= ch <= "Z":
            v = ord(ch) - 55
        else:  # '<'
            v = 0
        total += v * (7, 3, 1)[i % 3]
    return str(total % 10)


@dataclass
class MrzResult:
    format: str
    doc_code: str
    country: str
    lines: list[str]
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def valid_count(self) -> int:
        return sum(self.checks.values())

    @property
    def all_valid(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def doc_type(self) -> str:
        code = self.doc_code.replace("<", "")
        if code.startswith("P"):
            return "passport"
        if code.startswith("V"):
            return "other_id"  # visa
        if code in ("IR", "AR", "CR", "RP"):  # residence-permit codes used by e.g. EU/UK TD1 cards
            return "residence_permit"
        return "national_id"


def clean_line(line: str) -> str:
    line = line.upper().replace("«", "<<").replace("‹", "<").replace("〈", "<")
    return re.sub(r"\s+", "", line)


def _digits(s: str) -> str:
    return s.translate(_DIGIT_FIX)


def _validate(fmt: str, lines: list[str]) -> dict[str, bool]:
    def ok(value: str, digit: str, numeric: bool = False) -> bool:
        v = _digits(value) if numeric else value
        return check_digit(v) == _digits(digit)

    if fmt == "TD3":
        l2 = lines[1]
        num = _digits(l2[13:20]) + _digits(l2[21:28])
        personal, personal_cd = l2[28:42], l2[42]
        return {
            "document_number": ok(l2[0:9], l2[9]),
            "birth_date": ok(l2[13:19], l2[19], numeric=True),
            "expiry_date": ok(l2[21:27], l2[27], numeric=True),
            # An empty personal number may have '<' or '0' as its check digit.
            "personal_number": personal_cd in "<0" if not personal.strip("<") else ok(personal, personal_cd),
            "composite": ok(l2[0:10] + num + l2[28:43], l2[43]),
        }
    if fmt == "TD2":
        l2 = lines[1]
        num = _digits(l2[13:20]) + _digits(l2[21:28])
        return {
            "document_number": ok(l2[0:9], l2[9]),
            "birth_date": ok(l2[13:19], l2[19], numeric=True),
            "expiry_date": ok(l2[21:27], l2[27], numeric=True),
            "composite": ok(l2[0:10] + num + l2[28:35], l2[35]),
        }
    l1, l2 = lines[0], lines[1]  # TD1
    num = _digits(l2[0:7]) + _digits(l2[8:15])
    return {
        "document_number": ok(l1[5:14], l1[14]),
        "birth_date": ok(l2[0:6], l2[6], numeric=True),
        "expiry_date": ok(l2[8:14], l2[14], numeric=True),
        "composite": ok(l1[5:30] + num + l2[18:29], l2[29]),
    }


def _length_variants(line: str, target: int) -> list[str]:
    """The line itself if it fits, else versions with one '<' run grown/shrunk to the target length."""
    if len(line) == target:
        return [line]
    diff = target - len(line)
    if abs(diff) > 3:
        return []
    out = []
    for m in re.finditer(r"<+", line):
        run = m.group()
        if len(run) + diff < 1:
            continue
        out.append(line[: m.start()] + "<" * (len(run) + diff) + line[m.end():])
    return out


def _name_line_variants(line: str, target: int) -> list[str]:
    """Name lines (TD3/TD2 line 1, TD1 line 3) carry no check digits: accept any OCR length, pad/trim."""
    return [line[:target].ljust(target, "<")]


# Which lines of each format are name-only (no check digits).
_NAME_LINES = {"TD3": {0}, "TD2": {0}, "TD1": {2}}


def _candidates(text: str) -> list[str]:
    lines = []
    for raw in text.split("\n"):
        ln = clean_line(raw)
        if 20 <= len(ln) <= 47 and _MRZ_CHARS.match(ln) and "<" in ln:
            lines.append(ln)
        else:
            lines.append("")  # keep positions so only consecutive lines form an MRZ
    return lines


def find_mrz(text: str) -> MrzResult | None:
    """Best MRZ found in the text (most valid check digits), or None."""
    lines = _candidates(text)
    best: MrzResult | None = None
    for fmt, (n, length) in FORMATS.items():
        for start in range(len(lines) - n + 1):
            group = lines[start:start + n]
            if not all(group):
                continue
            variants: list[list[str]] = [[]]
            for i, ln in enumerate(group):
                options = (_name_line_variants(ln, length) if i in _NAME_LINES[fmt]
                           else _length_variants(ln, length))
                variants = [v + [x] for v in variants for x in options][:64]
            for v in variants:
                if fmt in ("TD2", "TD3") and v[0][0] not in "PIACV":
                    continue
                if fmt == "TD1" and v[0][0] not in "IACV":
                    continue
                res = MrzResult(fmt, v[0][0:2], v[0][2:5], v, _validate(fmt, v))
                if best is None or res.valid_count > best.valid_count:
                    best = res
    return best


def has_mrz_like_line(text: str) -> bool:
    return any(len(ln) >= 20 and "<<<" in ln for ln in _candidates(text))


# --- Field parsing (ICAO 9303 layouts) -----------------------------------------------------------------------

_CHECK_NAMES = {"document_number": "documentNumber", "birth_date": "birthDate", "expiry_date": "expiryDate",
                "personal_number": "personalNumber", "composite": "composite"}


_LETTER_FIX = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"})
_WORD = re.compile(r"[A-Z]{2,}")


def _text(s: str) -> str | None:
    """'<' is the filler: strip it, use it as the word separator."""
    out = re.sub(r"<+", " ", s).strip()
    return out or None


def _letters(s: str) -> str | None:
    """Fields that only hold letters (country codes, names): undo OCR digit look-alikes."""
    return _text(s.translate(_LETTER_FIX))


def _names(s: str) -> tuple[str | None, str | None]:
    surname, _, given = s.partition("<<")
    return _letters(surname), _letters(given)


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


def reconcile_names(fields: dict, text: str) -> dict:
    """The names line has no check digit, and OCR drops letters in it (TESTER -> TEER). Replace each MRZ name
    word by the one printed word (visual zone) that contains it as a subsequence and is at most 3 letters longer.
    Sets `namesSource` to "mrz" or "mrz+text"."""
    viz = "\n".join(ln for ln in text.upper().splitlines() if "<" not in ln)  # printed text, not the MRZ itself
    words = set(_WORD.findall(viz))
    changed = False
    for key in ("surname", "givenNames"):
        if not fields.get(key):
            continue
        out = []
        for token in fields[key].split():
            if token in words:
                out.append(token)
                continue
            cands = {w for w in words if len(token) < len(w) <= len(token) + 3 and len(token) >= 0.6 * len(w)
                     and w[0] == token[0] and _is_subsequence(token, w)}
            if len(cands) == 1:
                out.append(cands.pop())
                changed = True
            else:
                out.append(token)
        fields[key] = " ".join(out)
    fields["namesSource"] = "mrz+text" if changed else "mrz"
    return fields


def _date(yymmdd: str, kind: str, today: date) -> str | None:
    """YYMMDD -> ISO date. Birth dates are in the past (century chosen so); expiry dates are 20YY."""
    d = _digits(yymmdd)
    if not d.isdigit():
        return None
    yy, mm, dd = int(d[:2]), int(d[2:4]), int(d[4:6])
    year = 2000 + yy
    if kind == "birth" and year > today.year:
        year -= 100
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _sex(c: str) -> str | None:
    return c if c in ("M", "F", "X") else None


def parse_fields(res: MrzResult, today: date | None = None, text: str | None = None) -> dict:
    """Holder and document fields of a found MRZ, as camelCase JSON. Check-digit results are under `checks`:
    a field whose check failed is still returned (it may hold a single OCR error), so callers decide.
    Names carry no check digit; with `text` (the whole document) they are cross-checked, see reconcile_names."""
    today = today or date.today()
    ln = res.lines
    if res.format in ("TD3", "TD2"):
        width = FORMATS[res.format][1]
        surname, given = _names(ln[0][5:width])
        l2 = ln[1]
        fields = {"documentNumber": _text(l2[0:9]), "nationality": _letters(l2[10:13]),
                  "birthDate": _date(l2[13:19], "birth", today), "sex": _sex(l2[20]),
                  "expiryDate": _date(l2[21:27], "expiry", today),
                  "optionalData": _text(l2[28:42] if res.format == "TD3" else l2[28:35])}
    else:  # TD1
        surname, given = _names(ln[2][0:30])
        l1, l2 = ln[0], ln[1]
        fields = {"documentNumber": _text(l1[5:14]), "nationality": _letters(l2[15:18]),
                  "birthDate": _date(l2[0:6], "birth", today), "sex": _sex(l2[7]),
                  "expiryDate": _date(l2[8:14], "expiry", today),
                  "optionalData": _text(l1[15:30] + "<" + l2[18:29])}
    expiry = fields["expiryDate"]
    out = {
        "format": res.format,
        "documentCode": _text(res.doc_code),
        "issuingCountry": _letters(res.country),
        "surname": surname,
        "givenNames": given,
        **fields,
        "expired": (date.fromisoformat(expiry) < today) if expiry else None,
        "valid": res.all_valid,
        "checks": {_CHECK_NAMES.get(k, k): v for k, v in res.checks.items()},
    }
    return reconcile_names(out, text) if text is not None else out
