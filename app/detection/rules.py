"""Rules engine: fast, explainable ID signals. Every hit carries a weight and a reason."""

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from app.detection.mrz import MrzResult, find_mrz, has_mrz_like_line
from app.normalize import clean_text, normalize_for_match

MRZ_VALID_WEIGHT = 0.99
MRZ_PARTIAL_WEIGHT = 0.6  # most check digits valid
MRZ_LIKE_WEIGHT = 0.35  # "<<<" lines without valid check digits


@dataclass
class Signal:
    weight: float
    reason: str
    doc_type: str | None = None


@dataclass
class RuleResult:
    score: float  # 0..1
    doc_type: str | None  # best hint, or None
    signals: list[Signal] = field(default_factory=list)
    negatives: list[Signal] = field(default_factory=list)
    mrz: MrzResult | None = None

    @property
    def reasons(self) -> list[str]:
        return [s.reason for s in self.signals + self.negatives]

    @property
    def decisive(self) -> bool:
        """A fully valid MRZ decides without the LLM."""
        return self.mrz is not None and self.mrz.all_valid


@dataclass
class _Keyword:
    text: str
    key: str  # normalized, compact, upper-case
    weight: float
    doc_type: str | None


@dataclass
class _IdPattern:
    country: str
    name: str
    regex: re.Pattern
    weight: float
    doc_type: str | None


@dataclass
class RuleSet:
    keywords: list[_Keyword]
    negatives: list[_Keyword]
    patterns: list[_IdPattern]


def _key(text: str) -> str:
    return "".join(normalize_for_match(clean_text(text)).split()).upper()


def _keywords(items) -> list[_Keyword]:
    return [_Keyword(i["text"], _key(i["text"]), float(i["weight"]), i.get("doc_type")) for i in items or []]


_WORD_CHAR = r"A-Z0-9؀-ۿ"
SHORT_KEYWORD = 4  # keys this short must match a whole word (e.g. "VAT" must not hit "PRIVATE")


def _matches(kw: _Keyword, compact: str, spaced: str) -> bool:
    if not kw.key:
        return False
    if len(kw.key) <= SHORT_KEYWORD:
        return re.search(rf"(?<![{_WORD_CHAR}]){re.escape(kw.key)}(?![{_WORD_CHAR}])", spaced) is not None
    return kw.key in compact


def parse_rules(cfg: dict) -> RuleSet:
    patterns = [
        _IdPattern(country, p["name"], re.compile(p["pattern"]), float(p["weight"]), p.get("doc_type"))
        for country, items in (cfg.get("countries") or {}).items()
        for p in items or []
    ]
    return RuleSet(_keywords(cfg.get("keywords")), _keywords(cfg.get("negative_keywords")), patterns)


@lru_cache
def load_rules(path: Path) -> RuleSet:
    return parse_rules(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


@lru_cache(maxsize=64)
def load_rules_text(text: str) -> RuleSet:
    """A rules file's YAML text -> RuleSet. ValueError with the reason when it is invalid."""
    try:
        cfg = yaml.safe_load(text) or {}
        if not isinstance(cfg, dict):
            raise ValueError("the rules must be a YAML mapping (keywords, negative_keywords, countries)")
        unknown = set(cfg) - {"keywords", "negative_keywords", "countries"}
        if unknown:
            raise ValueError(f"unknown top-level keys: {', '.join(sorted(map(str, unknown)))}")
        rules = parse_rules(cfg)
    except ValueError:
        raise
    except yaml.YAMLError as e:
        raise ValueError(f"not valid YAML: {e}") from None
    except re.error as e:
        raise ValueError(f"invalid regular expression: {e}") from None
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed entry ({type(e).__name__}: {e})") from None
    for kw in rules.keywords + rules.negatives:
        if not 0 <= kw.weight <= 1:
            raise ValueError(f"keyword {kw.text!r}: weight must be between 0 and 1")
    for p in rules.patterns:
        if not 0 <= p.weight <= 1:
            raise ValueError(f"pattern {p.name!r}: weight must be between 0 and 1")
    return rules


# Rule sets managed in the admin dashboard, referenced as "db:<name>" (ID_PATTERNS_PATH or a profile's id_patterns)
MANAGED_PREFIX = "db:"
_managed: dict[str, str] = {}


def set_managed_rules(sets: dict[str, str]) -> None:
    """Replace the dashboard-managed rule sets ({name: YAML text}) of this process."""
    global _managed
    _managed = dict(sets)


def managed_rules() -> dict[str, str]:
    return dict(_managed)


def resolve_rules(ref: str | Path) -> tuple[RuleSet, str]:
    """A rules file path, or "db:<name>" for a dashboard-managed rule set -> (rules, version: start of its SHA-256).
    FileNotFoundError for an unknown managed rule set."""
    ref = str(ref)
    if ref.startswith(MANAGED_PREFIX):
        name = ref[len(MANAGED_PREFIX):]
        text = _managed.get(name)
        if text is None:
            raise FileNotFoundError(f"rule set {name!r} is not defined in the admin dashboard")
        return load_rules_text(text), hashlib.sha256(text.encode()).hexdigest()[:12]
    path = Path(ref)
    return load_rules(path), hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _noisy_or(weights: list[float]) -> float:
    p = 1.0
    for w in weights:
        p *= 1 - min(max(w, 0.0), 1.0)
    return 1 - p


def evaluate(text: str, normalized_text: str, rules: RuleSet) -> RuleResult:
    """`text` = cleaned text (for MRZ and ID numbers), `normalized_text` = for keyword matching."""
    spaced = normalized_text.upper()
    compact = "".join(spaced.split())
    signals: list[Signal] = []

    mrz = find_mrz(text)
    if mrz is not None and mrz.all_valid:
        signals.append(Signal(MRZ_VALID_WEIGHT, f"valid {mrz.format} MRZ ({mrz.doc_type}, "
                                                f"{mrz.valid_count}/{len(mrz.checks)} check digits)", mrz.doc_type))
    elif mrz is not None and mrz.valid_count >= len(mrz.checks) - 1:
        signals.append(Signal(MRZ_PARTIAL_WEIGHT, f"{mrz.format} MRZ with {mrz.valid_count}/{len(mrz.checks)} "
                                                  "valid check digits", mrz.doc_type))
    elif has_mrz_like_line(text):
        signals.append(Signal(MRZ_LIKE_WEIGHT, "MRZ-like lines ('<<<') without valid check digits"))

    for kw in rules.keywords:
        if _matches(kw, compact, spaced):
            signals.append(Signal(kw.weight, f"keyword '{kw.text}'", kw.doc_type))
    for p in rules.patterns:
        if p.regex.search(text):
            signals.append(Signal(p.weight, f"{p.country} ID-number pattern: {p.name}", p.doc_type))

    negatives = [Signal(-kw.weight, f"non-ID keyword '{kw.text}'")
                 for kw in rules.negatives if _matches(kw, compact, spaced)]

    score = _noisy_or([s.weight for s in signals])
    for n in negatives:
        score *= 1 + n.weight  # weight is negative
    by_type: dict[str, float] = {}
    for s in signals:
        if s.doc_type:
            by_type[s.doc_type] = by_type.get(s.doc_type, 0.0) + s.weight
    doc_type = max(by_type, key=by_type.get) if by_type else None
    return RuleResult(round(score, 4), doc_type, signals, negatives, mrz)
