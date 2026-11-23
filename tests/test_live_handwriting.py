"""The real handwriting model on synthetic pages: it must not invent text. Opt-in (slow, needs the model):

    RUN_HANDWRITING_TESTS=1 uv run pytest tests/test_live_handwriting.py -v

Uses HANDWRITING_* from the environment / .env (backend, model, device); HANDWRITING_ENABLED is forced on. On CPU
each page takes a minute or more. Printed text stands in for handwriting here; the real accuracy question is answered
by eval/run_handwriting_eval.py on hand-verified scans.
"""

import os

import pytest

from app.config import Settings
from app.extractors.filetype import PNG
from app.handwriting import HandwritingService
from app.handwriting.backends import deps_installed, model_cached
from eval.hw_metrics import normalized_view, score
from tests import synth_handwriting as hw


def _settings(**kw) -> Settings:
    return Settings(handwriting_enabled=True, **kw)


def _ready() -> str | None:
    if os.environ.get("RUN_HANDWRITING_TESTS") != "1":
        return "set RUN_HANDWRITING_TESTS=1 to run the handwriting model"
    s = _settings()
    if s.handwriting_backend == "transformers" and not (deps_installed() and model_cached(s.handwriting_model)):
        return "handwriting model or the `handwriting` extra not installed"
    return None


pytestmark = pytest.mark.skipif(_ready() is not None, reason=_ready() or "")


@pytest.fixture(scope="module")
def unguarded():
    """The model sees every page: what it does on empty pages without the blank guard."""
    from app.privacy import apply_offline_env

    apply_offline_env()
    svc = HandwritingService(_settings(handwriting_blank_check=False))
    yield svc
    svc.close()


def _read(svc, img) -> str:
    return svc.transcribe(hw.png(img), PNG)["fullText"]


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def test_guard_skips_a_blank_page_without_calling_the_model():
    out = HandwritingService(_settings()).transcribe(hw.png(hw.blank()), PNG)
    assert out["pages"][0]["skipped"] == "blank" and out["fullText"] == ""


@pytest.mark.parametrize("page", [hw.blank, hw.noisy_blank], ids=["white", "scanned-noise"])
def test_model_invents_nothing_on_an_empty_page(unguarded, page):
    text = _read(unguarded, page())
    assert _letters(text) <= 5, f"invented {_letters(text)} letters on an empty page"


def test_signature_only_yields_no_sentence(unguarded):
    text = _read(unguarded, hw.signature())
    assert len(text.split()) <= 3, f"{len(text.split())} words for a page with only a signature"


def test_stamp_only_yields_at_most_the_stamp_text(unguarded):
    text = _read(unguarded, hw.stamp())
    assert len(normalized_view(text)) <= len("مختوم") + 10, "more text than the stamp holds"


def test_few_words_are_read_without_additions(unguarded):
    sc = score("نعم موافق", _read(unguarded, hw.few_words("نعم موافق")))
    assert sc.hyp_chars <= sc.ref_chars + 6, "the model added text to a two-word page"
    assert sc.norm_cer is not None and sc.norm_cer <= 0.5


def test_arabic_page_loose_accuracy(unguarded):
    lines = ["بسم الله الرحمن الرحيم", "السيد مدير المدرسة المحترم", "أرجو الموافقة على طلبي"]
    sc = score(" ".join(lines), _read(unguarded, hw.text_page(lines)))
    assert sc.raw_cer is not None and sc.raw_cer < 0.3, f"raw CER {sc.raw_cer:.2f}"
