"""Handwriting evaluation: raw vs normalized CER/WER, letters never folded, gap flags, manifest rules."""

import json

import pytest

from app.handwriting import HandwritingService
from app.handwriting.backends import HwOutput
from eval import run_handwriting_eval
from eval.hw_metrics import Edits, Sample, category_summary, edit_ops, invented_text, normalized_view, raw_view, score
from tests import synth_handwriting as hw
from tests.test_api import make_settings


def test_edit_ops_counts():
    assert edit_ops("abc", "abc") == Edits(0, 0, 0)
    assert edit_ops("abc", "abd") == Edits(1, 0, 0)
    assert edit_ops("abc", "ab") == Edits(0, 1, 0)
    assert edit_ops("abc", "abcxy") == Edits(0, 0, 2)
    assert edit_ops("", "xyz") == Edits(0, 0, 3)
    assert edit_ops("kitten", "sitting").total == 3
    assert edit_ops("one two three".split(), "one 2 three four".split()) == Edits(1, 0, 1)


def test_raw_view_changes_only_nfc_and_whitespace():
    assert raw_view("  أحمد \n\n  علي\t") == "أحمد علي"
    assert raw_view("أ") == "أ"  # NFC composes alef + combining hamza
    assert raw_view("مُحَمَّد ـ ١٢") == "مُحَمَّد ـ ١٢"  # tashkeel, tatweel and digits untouched


def test_normalization_removes_marks_but_never_folds_letters():
    assert normalized_view("مُحَمَّد") == "محمد"
    assert normalized_view("مـــحمد") == "محمد"
    assert normalized_view("رقم ١٢٣ و ۴۵") == "رقم 123 و 45"
    assert normalized_view("نعم، لا؟") == "نعم, لا?"
    for a, b in [("أحمد", "احمد"), ("إسلام", "اسلام"), ("آمال", "امال"), ("على", "علي"), ("مدرسة", "مدرسه"),
                 ("مسؤول", "مسوول"), ("شئ", "شيء")]:
        assert normalized_view(a) != normalized_view(b), (a, b)


def test_diacritics_only_difference_is_a_raw_error_and_flagged():
    sc = score("مُحَمَّدٌ رَسُولُ", "محمد رسول")
    assert sc.raw_cer > 0.4 and sc.norm_cer == 0 and sc.gap_flag
    assert sc.raw_wer == 1.0 and sc.norm_wer == 0


def test_real_letter_errors_stay_errors_after_normalization():
    sc = score("أحمد في المدرسة", "احمد في المدرسه")
    assert sc.norm_cer == sc.raw_cer == pytest.approx(2 / 15) and not sc.gap_flag
    assert sc.norm_wer == sc.raw_wer == pytest.approx(2 / 3)


def test_empty_reference_rates_are_undefined_and_inserted_text_counted():
    sc = score("", "نص مخترع")
    assert sc.raw_cer is None and sc.norm_cer is None and sc.gap is None
    assert sc.hyp_chars == 8 and sc.raw_char.insertions == 8 and sc.inserted_share == 1
    assert score("", "").inserted_share is None


def test_category_summary_pools_edits():
    a, b = Sample("a", "mixed", score("abcd", "abcx"), pages=1, ms=1000), Sample("b", "mixed", score("ab", "ab"),
                                                                                 pages=1, ms=3000)
    c = category_summary([a, b, Sample("c", "mixed", error="corrupt")])
    assert c["rawCer"] == pytest.approx(1 / 6) and c["n"] == 3 and c["errors"] == 1 and c["msPerPage"] == 2000


def test_invented_text_summary():
    rows = [Sample("b1", "blank", score("", ""), pages=1, skipped_pages=1),
            Sample("b2", "blank", score("", "نص"), pages=1),
            Sample("s1", "signature_only", score("محمد", "محمد علي حسن"), pages=1)]
    out = invented_text(rows)
    assert out["blank"] == {"n": 2, "withOutput": 1, "excessChars": 2, "maxExcessChars": 2, "skippedByGuard": 1}
    assert out["signature_only"]["excessChars"] == 8 and "stamp_only" not in out


class ReplayBackend:
    """Answers per image size, so each sample gets a known output."""
    name = "transformers"

    def __init__(self, answers):
        self.answers = answers

    def transcribe(self, img):
        return HwOutput(self.answers.get(img.shape[1], ""), 1, 1, "stop")

    def info(self):
        return {"backend": "transformers", "name": "stub", "device": "cpu"}

    def status(self):
        return {}

    def load(self):
        return None


def write_samples(folder):
    from PIL import Image

    folder.mkdir()
    pages = {"a01.png": hw.few_words("نعم موافق"), "blank.png": hw.blank(), "sig.png": hw.signature(),
             "todo.png": hw.few_words("لاحقا")}
    for width, (name, img) in enumerate(pages.items(), start=1001):  # the width tells ReplayBackend the sample
        Image.fromarray(img[:, :, ::-1]).resize((width, 1400)).save(folder / name)
    (folder / "a01.txt").write_text("نعم موافق", encoding="utf-8")
    (folder / "sig.txt").write_text("", encoding="utf-8")
    (folder / "manifest.csv").write_text(
        "file,category,gt_file,verified_by,verified_at,notes\n"
        "a01.png,ar_handwritten,a01.txt,Reviewer,2026-11-20,\n"
        "blank.png,blank,,Reviewer,2026-11-20,empty\n"
        "sig.png,signature_only,sig.txt,Reviewer,2026-11-20,\n"
        "todo.png,sparse,todo.txt,,,not checked yet\n", encoding="utf-8")


def test_run_scores_verified_rows_only(tmp_path):
    write_samples(tmp_path / "samples")
    s = make_settings(tmp_path, handwriting_enabled=True)
    answers = {1001: "نعم موافق", 1003: "توقيع مخترع"}  # a01.png, sig.png
    svc = HandwritingService(s, backend=ReplayBackend(answers))
    out_dir = tmp_path / "out"
    summary = run_handwriting_eval.run(tmp_path / "samples", out_dir, service=svc, echo=lambda *_: None)
    assert summary["scored"] == 3 and summary["unverified"] == ["todo.png"]
    assert summary["categories"]["ar_handwritten"]["rawCer"] == 0
    assert summary["inventedText"]["blank"]["skippedByGuard"] == 1
    assert summary["inventedText"]["signature_only"]["withOutput"] == 1
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Unverified" in report and "todo.png" in report and "NOT folded" in report
    assert "نعم" not in (out_dir / "results.csv").read_text(encoding="utf-8")  # scores only, no text
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["config"]["blankCheck"] is True
    assert not (out_dir / "texts").exists()


def test_manifest_rules(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "manifest.csv").write_text("file,category,gt_file,verified_by\na.png,poetry,a.txt,X\n",
                                         encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown category"):
        run_handwriting_eval.load_manifest(folder / "manifest.csv")
    (folder / "manifest.csv").write_text("file,category,gt_file,verified_by\na.png,ar_handwritten,,X\n",
                                         encoding="utf-8")
    with pytest.raises(SystemExit, match="gt_file is required"):
        run_handwriting_eval.load_manifest(folder / "manifest.csv")


def test_refuses_a_folder_inside_the_repo(tmp_path):
    with pytest.raises(SystemExit, match="OUTSIDE"):
        run_handwriting_eval.run(run_handwriting_eval.REPO_ROOT / "tests", tmp_path / "out", service=object())
