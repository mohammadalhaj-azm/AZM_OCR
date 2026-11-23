"""Evaluation metrics (pure functions)."""

import statistics
from dataclasses import dataclass, field

DOC_TYPES = ["national_id", "passport", "driver_license", "residence_permit", "other_id", "not_id"]
ERROR = "error"


@dataclass
class Row:
    file: str
    label: str | None  # doc type, or None if unlabeled
    pred: str  # doc type, or "error"
    confidence: float | None = None
    decided_by: list[str] = field(default_factory=list)
    pages: int = 0
    ocr_pages: int = 0
    ocr_ms: int = 0
    llm_ms: int = 0
    vision_ms: int = 0
    total_ms: int = 0
    error: str = ""

    @property
    def correct(self) -> bool | None:
        return None if self.label is None else self.pred == self.label


def labeled(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.label is not None]


def binary_metrics(rows: list[Row]) -> dict:
    """ID vs not-ID on labeled rows that were processed (errors are reported separately)."""
    tp = fp = fn = tn = 0
    for r in labeled(rows):
        if r.pred == ERROR:
            continue
        actual, predicted = r.label != "not_id", r.pred != "not_id"
        tp += actual and predicted
        fp += (not actual) and predicted
        fn += actual and not predicted
        tn += (not actual) and not predicted
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / n if n else None,
            "precision": precision, "recall": recall, "f1": f1}


def doc_type_accuracy(rows: list[Row]) -> dict:
    lab = [r for r in labeled(rows) if r.pred != ERROR]
    correct = sum(r.pred == r.label for r in lab)
    return {"n": len(lab), "correct": correct, "accuracy": correct / len(lab) if lab else None}


def confusion_matrix(rows: list[Row]) -> dict[str, dict[str, int]]:
    """matrix[label][prediction] = count. Predictions include 'error'."""
    m = {lab: {p: 0 for p in DOC_TYPES + [ERROR]} for lab in DOC_TYPES}
    for r in labeled(rows):
        m[r.label][r.pred] += 1
    return m


def timing_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    s = sorted(values)
    p95 = s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]
    return {"n": len(s), "mean": round(statistics.fmean(s)), "median": round(statistics.median(s)),
            "p95": round(p95), "max": round(s[-1])}


def ocr_ms_per_page(rows: list[Row]) -> list[float]:
    """One value per OCR'd page (each document's OCR time spread over its OCR'd pages)."""
    out: list[float] = []
    for r in rows:
        if r.ocr_pages:
            out += [r.ocr_ms / r.ocr_pages] * r.ocr_pages
    return out


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def render_report(rows: list[Row], settings_info: dict) -> str:
    b, d, cm = binary_metrics(rows), doc_type_accuracy(rows), confusion_matrix(rows)
    errors = [r for r in rows if r.pred == ERROR]
    ocr, total = timing_stats(ocr_ms_per_page(rows)), timing_stats([r.total_ms for r in rows if r.pred != ERROR])
    llm = timing_stats([r.llm_ms for r in rows if r.llm_ms])
    vision = timing_stats([r.vision_ms for r in rows if r.vision_ms])
    decided: dict[str, int] = {}
    for r in rows:
        if r.decided_by:
            key = "+".join(r.decided_by)
            decided[key] = decided.get(key, 0) + 1

    lines = ["# Evaluation report", "",
             f"Documents: {len(rows)} ({len(labeled(rows))} labeled, {len(errors)} failed to process)", "",
             "Settings: " + ", ".join(f"{k}={v}" for k, v in settings_info.items()), "",
             "## Timing (ms)", "", "| | n | mean | median | p95 | max |", "|---|---|---|---|---|---|"]
    for name, t in [("OCR per page", ocr), ("LLM per document", llm), ("Vision per document", vision),
                    ("Total per document", total)]:
        lines.append(f"| {name} | {t['n']} | " + " | ".join(str(t.get(k, "-")) for k in
                                                          ("mean", "median", "p95", "max")) + " |")
    lines += ["", "## ID detection", "",
              f"- ID vs not-ID accuracy: **{_pct(b['accuracy'])}** (n={b['n']}; TP={b['tp']} FP={b['fp']} "
              f"FN={b['fn']} TN={b['tn']})",
              f"- Precision {_pct(b['precision'])}, recall {_pct(b['recall'])}, F1 {_pct(b['f1'])}",
              f"- Document-type accuracy: **{_pct(d['accuracy'])}** ({d['correct']}/{d['n']})",
              "- Decided by: " + (", ".join(f"{k}: {v}" for k, v in sorted(decided.items())) or "n/a"), "",
              "## Confusion matrix (rows = label, columns = prediction)", ""]
    cols = DOC_TYPES + [ERROR]
    lines += ["| label \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for lab in DOC_TYPES:
        if any(cm[lab].values()):
            lines.append(f"| **{lab}** | " + " | ".join(str(cm[lab][c] or "·") for c in cols) + " |")
    wrong = [r for r in labeled(rows) if not r.correct]
    if wrong:
        lines += ["", "## Misclassified (local file names; keep this report private)", "",
                  "| file | label | prediction | confidence | decided by | error |", "|---|---|---|---|---|---|"]
        for r in wrong:
            lines.append(f"| {r.file} | {r.label} | {r.pred} | {r.confidence if r.confidence is not None else ''} | "
                         f"{'+'.join(r.decided_by)} | {r.error} |")
    return "\n".join(lines) + "\n"
