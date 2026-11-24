"""Routing decisions of extract-and-answer (app/answer/routing.py): pure functions over PaddleOCR's signals."""

import pytest

from app.answer.routing import HANDWRITING, PADDLE, decide, document_kind, engines_for, needs_image, ocr_signals
from app.config import Settings
from tests.fakes_answer import HANDWRITTEN, PRINTED, UNCLEAR, ocr

S = Settings(_env_file=None)


def sig(lines, detected=None, ink=None):
    return ocr_signals(ocr(lines, detected), ink, S)


def test_signals_are_numbers_only():
    s = sig(UNCLEAR)
    assert s == {"detected": 4, "read": 4, "meanConfidence": s["meanConfidence"], "lowConfidenceShare": 0.5,
                 "inkRatio": None}
    assert 0.7 < s["meanConfidence"] < 0.9


@pytest.mark.parametrize("signals,decision", [
    (sig(PRINTED), "printed"),
    (sig(HANDWRITTEN), "handwritten"),  # low confidence
    (sig(PRINTED, detected=10), "handwritten"),  # most detected regions could not be read
    (sig([], detected=0, ink=0.0), "no_text"),
    (sig([], detected=0, ink=0.01), "unclear"),  # ink but no text found: let the vision model look
    (sig(UNCLEAR), "unclear"),  # confident printed lines next to weak ones: possibly mixed
])
def test_decide(signals, decision):
    assert decide(signals, S)[0] == decision


def test_thresholds_are_settings():
    strict = S.model_copy(update={"routing_printed_min_conf": 0.999})
    assert decide(sig(PRINTED), strict)[0] == "unclear"


def test_engines_per_kind():
    assert engines_for("printed", "ocr_signals", S, arabic=True, paddle_has_text=True) == [PADDLE]
    switched = S.model_copy(update={"route_printed_arabic": "handwriting"})
    assert engines_for("printed", "ocr_signals", switched, True, True) == [HANDWRITING, PADDLE]
    assert engines_for("printed", "ocr_signals", switched, False, True) == [PADDLE]  # English stays on PaddleOCR
    assert engines_for("mixed", "vision", S, False, True) == [HANDWRITING, PADDLE]  # both readings kept
    assert engines_for("handwritten", "ocr_signals", S, False, False) == [HANDWRITING]
    for fallback, engines in (("both", [HANDWRITING, PADDLE]), ("handwriting", [HANDWRITING]),
                              ("printed", [PADDLE])):
        s = S.model_copy(update={"routing_uncertain_fallback": fallback})
        assert engines_for("unknown", "fallback", s, False, True) == engines


def test_images_are_kept_only_where_needed():
    assert not needs_image("printed", S, arabic=True, mode="auto")
    assert needs_image("unclear", S, False, "auto") and needs_image("handwritten", S, False, "auto")
    assert needs_image("printed", S, False, "handwritten") and not needs_image("handwritten", S, False, "printed")
    assert needs_image("printed", S.model_copy(update={"route_printed_arabic": "handwriting"}), True, "auto")


@pytest.mark.parametrize("kinds,doc", [
    (["printed", "printed"], "printed"), (["printed", "handwritten"], "mixed"), (["mixed"], "mixed"),
    (["no_text"], "no_text"), (["handwritten", "no_text"], "handwritten"), (["unknown"], "unknown"),
    (["printed", "unknown"], "printed"), ([], "no_text"),
])
def test_document_kind(kinds, doc):
    assert document_kind(kinds) == doc
