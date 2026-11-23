import unicodedata

import pytest

from app.normalize.arabic import (
    clean_text,
    detect_languages,
    fix_reversed_lines,
    is_reversed_line,
    normalize_for_match,
    reverse_visual_line,
)


def shaped(text: str) -> str:
    """Logical text -> presentation forms (what many PDFs store)."""
    import arabic_reshaper

    return arabic_reshaper.reshape(text)


# --- NFKC / presentation forms ----------------------------------------------------------------

def test_presentation_forms_become_normal_letters():
    raw = shaped("الرقم الوطني")
    assert any(0xFE70 <= ord(c) <= 0xFEFF for c in raw)  # really presentation forms
    assert clean_text(raw) == "الرقم الوطني"


def test_lam_alef_ligature_decomposes():
    assert clean_text("ﻻ") == "لا"  # ﻻ


def test_nfkc_on_latin_fullwidth():
    assert clean_text("ＰＡＳＳＰＯＲＴ") == "PASSPORT"


# --- digits ---------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("٠١٢٣٤٥٦٧٨٩", "0123456789"),  # Arabic-Indic
    ("۰۱۲۳۴۵۶۷۸۹", "0123456789"),  # Persian / Extended Arabic-Indic
    ("الرقم الوطني: ٠١٢٣٤٥٦٧٨٩٠", "الرقم الوطني: 01234567890"),
    ("١٢٫٥٪", "12.5%"),
    ("١٬٥٠٠", "1,500"),
])
def test_digits_converted(raw, expected):
    assert clean_text(raw) == expected


# --- tatweel / tashkeel / controls ------------------------------------------------------------

def test_tatweel_removed_from_text():
    assert clean_text("بطـــاقة") == "بطاقة"


def test_tashkeel_kept_in_text_but_stripped_for_matching():
    raw = "مُحَمَّد"
    assert clean_text(raw) == raw
    assert normalize_for_match(clean_text(raw)) == "محمد"


def test_bidi_controls_and_zero_width_removed():
    assert clean_text("‏جواز‎ ‫سفر‬​") == "جواز سفر"


def test_whitespace_tidied():
    assert clean_text("  a \t b  \r\n\r\n\r\n\r\nc  ") == "a b\n\nc"


# --- letter unification (matching only) ---------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("أحمد", "احمد"), ("إقامة", "اقامه"), ("آمنة", "امنه"), ("ٱلله", "الله"),
    ("مصطفى", "مصطفي"), ("بطاقة شخصية", "بطاقه شخصيه"),
    ("کیف", "كيف"),  # Persian keheh / yeh
    ("نعم، لا؟", "نعم, لا?"),
])
def test_normalize_for_match(raw, expected):
    assert normalize_for_match(clean_text(raw)) == expected


def test_original_text_is_not_unified():
    assert clean_text("إقامة") == "إقامة"


def test_latin_untouched_by_match_normalization():
    assert normalize_for_match("P<XXXTESTER<<SAMPLE<<<<") == "P<XXXTESTER<<SAMPLE<<<<"


# --- reversed (visual-order) PDF text ---------------------------------------------------------

def visual(text: str) -> str:
    """Simulate a PDF that stores glyphs in left-to-right visual order."""
    from bidi.algorithm import get_display

    return get_display(text, base_dir="R")


@pytest.mark.parametrize("logical", [
    "الجمهورية العربية السورية",
    "بطاقة شخصية",
    "شركة",
    "رقم الفاتورة: 2024/115",
])
def test_reversed_line_with_clear_cues_fixed_on_its_own(logical):
    raw = visual(logical)
    assert raw != logical
    assert is_reversed_line(raw)
    assert clean_text(raw) == logical


REVERSED_PAGE = [
    "الجمهورية العربية التجريبية",
    "بطاقة شخصية",
    "جواز سفر PASSPORT",  # no cues of its own: page evidence decides
    "المبلغ (1500) ريال",
    "تاريخ الميلاد 01/02/1990",
    "Date of Birth: 01/02/1990",  # Latin-only line: never touched
]


def test_reversed_page_fixes_lines_without_own_cues():
    raw = "\n".join(visual(ln) for ln in REVERSED_PAGE)
    assert clean_text(raw) == "\n".join(REVERSED_PAGE)


def test_correct_page_never_flipped():
    from tests import synth

    for page in (REVERSED_PAGE, synth.ID_CARD_LINES, synth.INVOICE_LINES):
        text = "\n".join(page)
        assert clean_text(text) == text
        assert clean_text(shaped(text)) == text  # logical order + presentation forms


def test_line_without_cues_left_alone_when_page_is_not_reversed():
    raw = visual("جواز سفر")  # can't tell on its own: be conservative
    assert clean_text(raw) == raw


@pytest.mark.parametrize("logical", ["الجمهورية العربية", "وزارة الداخلية - بطاقة شخصية", "سامي"])
def test_reversed_presentation_forms_fixed(logical):
    raw = visual(shaped(logical))  # shaped + visual order: the classic broken-PDF case
    assert clean_text(raw) == logical


@pytest.mark.parametrize("logical", [
    "الجمهورية العربية السورية", "بطاقة شخصية", "إلا", "لا", "هلا والله", "الرقم الوطني: 01234567890",
    "Date of Birth: 01/02/1990", "P<XXXTESTER<<SAMPLE<<<<<<<<", "",
])
def test_correct_text_not_flipped(logical):
    assert not is_reversed_line(logical)
    assert not is_reversed_line(shaped(logical))
    assert clean_text(logical) == clean_text(logical, fix_reversed=False)


def test_only_reversed_lines_are_fixed():
    text = "INVOICE\n" + visual("فاتورة ضريبية") + "\nالتاريخ: 2024/03/15"
    assert fix_reversed_lines(text) == "INVOICE\nفاتورة ضريبية\nالتاريخ: 2024/03/15"


def test_reverse_visual_line_keeps_numbers_and_mirrors_brackets():
    raw = visual("الريال (1500)")
    assert raw == "(1500) لايرلا"
    assert reverse_visual_line(raw) == "الريال (1500)"


def test_fix_can_be_disabled():
    raw = visual("بطاقة شخصية")
    assert clean_text(raw, fix_reversed=False) == unicodedata.normalize("NFKC", raw)


# --- languages ----------------------------------------------------------------------------------

@pytest.mark.parametrize("text,langs", [
    ("بطاقة شخصية NATIONAL ID", ["ar", "en"]),
    ("PASSPORT Republic of Example and more", ["en"]),
    ("الجمهورية العربية", ["ar"]),
    ("12345", []),
])
def test_detect_languages(text, langs):
    assert detect_languages(text) == langs
