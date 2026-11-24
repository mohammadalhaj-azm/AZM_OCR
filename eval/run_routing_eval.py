"""Measure extract-and-answer's page routing (printed / handwritten / mixed) on labeled pages, to set ROUTING_*.

    uv run python -m eval.run_routing_eval D:\\routing-samples [--out DIR] [--vision] [--limit N]

The samples folder must be OUTSIDE this repository and contain manifest.csv (UTF-8, header required):

    file,label,verified_by,notes
    invoices/inv01.png,printed,reviewer-1,
    letters/hw03.jpg,handwritten,reviewer-1,
    forms/f12.pdf,mixed,reviewer-1,printed form filled in by hand (label applies to every page)
    misc/blank1.png,no_text,reviewer-1,

- label: printed, handwritten, mixed or no_text, decided by a person looking at the page. For a PDF or multi-page
  TIFF the label applies to every page (split documents whose pages differ).
- Only rows with verified_by filled in are run.

Every page is OCR'd by PaddleOCR (settings from the environment / .env, as the service uses them) and routed with
the current ROUTING_* thresholds. --vision also asks LLM_VISION_MODEL about the pages the signals leave unclear,
exactly as the service does.

Outputs (default eval/out/routing-<timestamp>/, git-ignored): results.csv (one row per page: file, page, label,
decision, final kind, signals), summary.json and report.md (confusion matrix, and the distribution of PaddleOCR's
mean confidence and low-confidence share per label, which is what the thresholds should be set from). No document
text is written. Nothing here is a claim about accuracy on other documents: it measures these pages only.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = ("printed", "handwritten", "mixed", "no_text")
COLUMNS = ("file", "label", "verified_by", "notes")


def read_manifest(folder: Path) -> list[dict]:
    path = folder / "manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ("file", "label", "verified_by") if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing} (expected {', '.join(COLUMNS)})")
        rows = []
        for i, row in enumerate(reader, start=2):
            label = (row.get("label") or "").strip()
            if label not in LABELS:
                raise SystemExit(f"{path}:{i}: label {label!r} is not one of {', '.join(LABELS)}")
            rows.append({"line": i, "file": row["file"].strip(), "label": label,
                         "verified": bool((row.get("verified_by") or "").strip())})
    return rows


def page_images(data: bytes, s) -> list:
    from app.extractors.filetype import IMAGE_TYPES, PDF, validate_upload
    from app.extractors.image import decode_images
    from app.extractors.pdf import render_pdf_pages

    mime = validate_upload(data, s.max_upload_bytes)
    if mime == PDF:
        return render_pdf_pages(data, s.ocr_dpi, s.max_pages)
    if mime in IMAGE_TYPES:
        return decode_images(data, s.max_pages)
    raise SystemExit(f"unsupported file type {mime} (images and PDFs only)")


def spread(values: list[float]) -> dict | None:
    if not values:
        return None
    v = sorted(values)

    def q(p: float) -> float:
        return round(v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))], 4)

    return {"n": len(v), "min": v[0], "p10": q(0.1), "median": round(statistics.median(v), 4), "p90": q(0.9),
            "max": v[-1]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_routing_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="samples folder with manifest.csv (outside the repository)")
    parser.add_argument("--out", type=Path, help="output folder (default eval/out/routing-<timestamp>)")
    parser.add_argument("--vision", action="store_true", help="classify unclear pages with LLM_VISION_MODEL")
    parser.add_argument("--limit", type=int, help="only the first N verified files")
    args = parser.parse_args(argv)

    folder = args.folder.resolve()
    if folder == REPO_ROOT or REPO_ROOT in folder.parents:
        raise SystemExit("keep the samples outside the repository (they are real documents)")
    sys.path.insert(0, str(REPO_ROOT))
    from app.answer.llm import AnswerLlm, LlmError
    from app.answer.routing import decide, ocr_signals
    from app.config import Settings
    from app.detection.llm import LlmClient
    from app.handwriting.guards import ink_ratio
    from app.ocr.engine import OcrPool

    s = Settings()
    rows = [r for r in read_manifest(folder) if r["verified"]]
    if args.limit:
        rows = rows[:args.limit]
    engine = OcrPool(s).load()
    alm = AnswerLlm(LlmClient(s), s) if args.vision else None
    vision_ok = bool(alm and alm.vision_available())
    if args.vision and not vision_ok:
        print(f"warning: {s.llm_vision_model} is not available: unclear pages stay unclear", file=sys.stderr)

    results = []
    for row in rows:
        data = (folder / row["file"]).read_bytes()
        for n, img in enumerate(page_images(data, s), start=1):
            out = engine.recognize(img)
            sig = ocr_signals(out, ink_ratio(img) if out.detected == 0 else None, s)
            decision, _ = decide(sig, s)
            final = decision
            if decision == "unclear" and vision_ok:
                try:
                    final = alm.classify_writing(img)[0]
                except LlmError:
                    final = "unclear"
            results.append({"file": row["file"], "page": n, "label": row["label"], "decision": decision,
                            "final": final, **sig})
            print(f"{row['file']} p{n}: label={row['label']} signals={decision} final={final}", file=sys.stderr)

    out_dir = args.out or REPO_ROOT / "eval" / "out" / f"routing-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]) if results else ["file"])
        w.writeheader()
        w.writerows(results)

    finals = sorted({r["final"] for r in results} | {"unclear"})
    confusion = {lab: {fin: sum(1 for r in results if r["label"] == lab and r["final"] == fin) for fin in finals}
                 for lab in LABELS}
    per_label = {lab: {"meanConfidence": spread([r["meanConfidence"] for r in results
                                                 if r["label"] == lab and r["meanConfidence"] is not None]),
                       "lowConfidenceShare": spread([r["lowConfidenceShare"] for r in results
                                                     if r["label"] == lab and r["lowConfidenceShare"] is not None])}
                 for lab in LABELS}
    thresholds = {k: getattr(s, k) for k in ("routing_low_line_conf", "routing_printed_min_conf",
                                             "routing_printed_max_low_share", "routing_handwritten_max_conf",
                                             "routing_min_kept_share")}
    summary = {"pages": len(results), "files": len(rows), "vision": vision_ok, "thresholds": thresholds,
               "confusion": confusion, "signalsByLabel": per_label}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Routing evaluation", "", f"{len(results)} pages from {len(rows)} verified files; vision: "
             f"{'on' if vision_ok else 'off'}.", "", "Thresholds: " + ", ".join(f"{k}={v}" for k, v in
                                                                                 thresholds.items()), "",
             "## Confusion (rows: label, columns: final route)", "",
             "| label | " + " | ".join(finals) + " |", "|---|" + "---|" * len(finals)]
    lines += [f"| {lab} | " + " | ".join(str(confusion[lab][f]) for f in finals) + " |" for lab in LABELS]
    lines += ["", "## PaddleOCR signals per label (set the thresholds between the labels' ranges)", "",
              "| label | signal | n | min | p10 | median | p90 | max |", "|---|---|---|---|---|---|---|---|"]
    for lab in LABELS:
        for name, d in per_label[lab].items():
            if d:
                lines.append(f"| {lab} | {name} | {d['n']} | {d['min']} | {d['p10']} | {d['median']} | {d['p90']} "
                             f"| {d['max']} |")
    lines += ["", "Mixed pages are expected to be `unclear` from the signals (the vision model decides them); a "
              "handwritten page routed `printed` is the costly mistake."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
