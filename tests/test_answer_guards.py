"""Checks of extract-and-answer that never trust the LLM (app/answer/guards.py)."""

import pytest

from app.answer import guards

TEXT = ("الاسم: أحمد علي\nرقم الهوية: 1234567890 تاريخ ١٢/٠٥/٢٠٢٤ المبلغ 1,250.00 ريال\n"
        "Passport A1234567, mail a.b@example.com, tel 050-123-4567")


def test_mask_hides_every_protected_value_and_unmask_restores_it():
    masked, values = guards.mask(TEXT)
    assert values == ["1234567890", "١٢/٠٥/٢٠٢٤", "1,250.00", "A1234567", "a.b@example.com", "050-123-4567"]
    assert not any(ch.isdigit() for ch in guards.PLACEHOLDER.sub("", masked))
    assert "أحمد علي" in masked and "Passport" in masked  # words stay for the LLM to correct
    assert guards.unmask(masked, values) == TEXT


def test_placeholders_intact_and_reordered_are_fine():
    masked, values = guards.mask("a 1 b 2")
    assert guards.check_placeholders(masked, values) is None
    assert guards.check_placeholders("b ⟦P2⟧\na ⟦P1⟧", values) is None  # layout may move values around


@pytest.mark.parametrize("reply,problem", [
    ("a ⟦P1⟧ b", "dropped 1"),
    ("a ⟦P1⟧ b ⟦P2⟧ ⟦P2⟧", "repeated"),
    ("a ⟦P1⟧ b ⟦P2⟧ ⟦P3⟧", "does not exist"),
    ("a ⟦P1⟧ b ⟦P2⟧ 2024", "added digits"),
    ("a ⟦P1⟧ b ⟦P2⟧ ٣", "added digits"),
])
def test_broken_placeholders_are_caught(reply, problem):
    _, values = guards.mask("a 1 b 2")
    assert problem in guards.check_placeholders(reply, values)


def test_located_ignores_letter_variants_tashkeel_case_and_spacing():
    page = "رقم  الهوية: 123\nCustomer: Sample Tester"
    assert guards.located("رقم الهويه", page)
    assert guards.located("customer:   SAMPLE", page)
    assert guards.located("⟪Sample⟫ Tester", page)
    assert not guards.located("Sample Toster", page)
    assert not guards.located("", page) and not guards.located("x", "", None)


def test_protected_values_unify_digits():
    assert guards.protected_values("رقم ١٢٣٤") == guards.protected_values("No 1234") == {"1234"}


def test_word_changes_are_computed_not_reported():
    assert guards.word_changes("the qick brown fox", "the quick brown fox") == [{"original": "qick",
                                                                                  "corrected": "quick"}]
    assert guards.word_changes("a b", "a b c") == [{"original": "", "corrected": "c"}]
    assert guards.word_changes("same text", "same\ntext") == []  # layout only
    assert len(guards.word_changes(" ".join("a" * 300), " ".join("b" * 300), limit=5)) <= 5


def test_chunks_split_at_lines_and_never_exceed_the_limit():
    text = "\n".join(f"line {i} " + "w" * 40 for i in range(50))
    parts = guards.chunks(text, 300)
    assert all(len(p) <= 300 for p in parts) and "\n".join(parts) == text
    long_line = "word " * 200
    assert all(len(p) <= 100 for p in guards.chunks(long_line, 100))
    assert guards.chunks("short", 100) == ["short"]
