import json

import pytest

from app.detection.detector import IdDetector
from app.pipeline import Services
from eval.metrics import Row, binary_metrics, confusion_matrix, doc_type_accuracy, ocr_ms_per_page, timing_stats
from eval.run_eval import REPO_ROOT, load_labels, run
from tests.test_detector import StubLlm, V


def rows():
    return [
        Row("a", "passport", "passport"),
        Row("b", "national_id", "passport"),  # ID, wrong type
        Row("c", "not_id", "not_id"),
        Row("d", "not_id", "national_id"),  # false positive
        Row("e", "national_id", "not_id"),  # false negative
        Row("f", "passport", "error", error="corrupt"),
        Row("g", None, "not_id"),  # unlabeled
    ]


def test_binary_metrics():
    b = binary_metrics(rows())
    assert (b["tp"], b["fp"], b["fn"], b["tn"], b["n"]) == (2, 1, 1, 1, 5)
    assert b["accuracy"] == 3 / 5 and b["precision"] == 2 / 3 and b["recall"] == 2 / 3


def test_doc_type_accuracy_and_confusion():
    assert doc_type_accuracy(rows()) == {"n": 5, "correct": 2, "accuracy": 0.4}
    cm = confusion_matrix(rows())
    assert cm["national_id"]["passport"] == 1 and cm["passport"]["error"] == 1 and cm["not_id"]["not_id"] == 1
    assert sum(sum(r.values()) for r in cm.values()) == 6  # labeled rows only


def test_timing():
    assert ocr_ms_per_page([Row("a", None, "x", ocr_pages=2, ocr_ms=3000), Row("b", None, "x")]) == [1500, 1500]
    t = timing_stats([100, 200, 300, 400])
    assert t["mean"] == 250 and t["median"] == 250 and t["max"] == 400


def test_labels_csv(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text("﻿file,label\nscans\\a.jpg,Passport\nb.pdf,not_id\n", encoding="utf-8")  # BOM, backslash
    assert load_labels(p) == {"scans/a.jpg": "passport", "b.pdf": "not_id"}
    p.write_text("file,label\na.jpg,invoice\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown label"):
        load_labels(p)


def test_refuses_folder_inside_repo(tmp_path):
    with pytest.raises(SystemExit, match="OUTSIDE"):
        run(REPO_ROOT / "tests", None, tmp_path / "out", services=object())


def test_run_on_synthetic_folder(tmp_path, settings, engine, fixtures):
    docs = tmp_path / "docs"
    (docs / "scans").mkdir(parents=True)
    (docs / "scans" / "passport.png").write_bytes(fixtures["passport.png"])
    (docs / "invoice.png").write_bytes(fixtures["invoice.png"])
    (docs / "broken.pdf").write_bytes(b"%PDF-1.7 broken")
    (docs / "labels.csv").write_text("file,label\nscans/passport.png,passport\ninvoice.png,not_id\n"
                                     "broken.pdf,not_id\n", encoding="utf-8")
    s = settings.model_copy(update={"vision_enabled": False})
    llm = StubLlm(V("not_id", 0.95))
    svc = Services(s, engine, llm, IdDetector(s, llm))
    printed = []
    summary = run(docs, docs / "labels.csv", tmp_path / "out", services=svc, echo=printed.append)

    assert summary["documents"] == 3 and summary["errors"] == 1  # labels.csv itself is not a document
    assert summary["docType"] == {"n": 2, "correct": 2, "accuracy": 1.0}
    assert summary["ocrMsPerPage"]["n"] == 2
    out = tmp_path / "out"
    assert {"results.csv", "report.md", "summary.json"} <= {p.name for p in out.iterdir()}
    assert "Confusion matrix" in (out / "report.md").read_text(encoding="utf-8")
    assert json.loads((out / "summary.json").read_text())["binary"]["accuracy"] == 1.0
    assert not any("passport.png" in line or "invoice.png" in line for line in printed)  # no names on screen
