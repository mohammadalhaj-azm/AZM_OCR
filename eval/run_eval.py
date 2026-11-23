"""Run the pipeline over a folder of real documents and report timing + ID-detection accuracy.

    uv run python -m eval.run_eval D:\\docs --labels D:\\docs\\labels.csv [--out DIR] [--limit N] [--no-llm] [--no-vision]

The documents folder must be OUTSIDE this repository. labels.csv (UTF-8, header required):
    file,label
    scans/id_001.jpg,national_id
    letters/offer.pdf,not_id
`file` is relative to the documents folder; `label` is one of: national_id, passport, driver_license,
residence_permit, other_id, not_id. Files without a label are processed (timing) but not scored.

Outputs (default eval/out/<timestamp>/, git-ignored): results.csv, report.md, summary.json.
They contain your file names and predictions, but no document text. Keep them private.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from eval.metrics import DOC_TYPES, ERROR, Row, binary_metrics, confusion_matrix, doc_type_accuracy, \
    ocr_ms_per_page, render_report, timing_stats

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"file", "label"} <= {h.strip().lower() for h in reader.fieldnames}:
            raise SystemExit("labels CSV needs a header with columns: file,label")
        for i, row in enumerate(reader, start=2):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            label = row["label"].lower()
            if label not in DOC_TYPES:
                raise SystemExit(f"labels CSV line {i}: unknown label {label!r}; use one of {', '.join(DOC_TYPES)}")
            labels[row["file"].replace("\\", "/")] = label
    return labels


def list_documents(folder: Path, exclude: set[Path]) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and not p.name.startswith(".") and p.resolve() not in exclude)


def run(docs_dir: Path, labels_path: Path | None, out_dir: Path, services=None, limit: int | None = None,
        echo=print) -> dict:
    docs_dir = docs_dir.resolve()
    if docs_dir.is_relative_to(REPO_ROOT):
        raise SystemExit("refusing to run: the documents folder must be OUTSIDE the repository")
    if services is None:
        from app.config import get_settings
        from app.pipeline import Services

        services = Services.create(get_settings())
    from app.extractors import FileRejected
    from app.pipeline import analyze

    labels = load_labels(labels_path) if labels_path else {}
    exclude = {labels_path.resolve()} if labels_path else set()
    files = list_documents(docs_dir, exclude)[:limit]
    services.engine.load()  # don't count model loading as OCR time

    rows: list[Row] = []
    for i, path in enumerate(files, 1):
        rel = path.relative_to(docs_dir).as_posix()
        label = labels.get(rel)
        t0 = time.perf_counter()
        try:
            out = analyze(path.read_bytes(), path.name, services)
            det, t = out["idDetection"], out["timingsMs"]
            row = Row(rel, label, det["docType"], det["confidence"], det["decidedBy"], out["file"]["pages"],
                      t.get("ocrPages", 0), t.get("ocr", 0), t.get("llm", 0), t.get("vision", 0), t["total"])
        except FileRejected as e:
            row = Row(rel, label, ERROR, error=e.code, total_ms=round((time.perf_counter() - t0) * 1000))
        except Exception as e:  # noqa: BLE001 - keep going; report the type only
            row = Row(rel, label, ERROR, error=type(e).__name__, total_ms=round((time.perf_counter() - t0) * 1000))
        rows.append(row)
        mark = "" if row.correct is None else (" ok" if row.correct else " WRONG")
        echo(f"[{i}/{len(files)}] pages={row.pages} {row.total_ms / 1000:.1f}s pred={row.pred} "
             f"label={label or '-'}{mark}")  # no file names on the console

    s = services.settings
    info = {"ocr_det": s.ocr_det_model, "text_model": s.llm_text_model if s.llm_enabled else "off",
            "vision_model": s.llm_vision_model if s.vision_enabled else "off", "rules_weight": s.rules_weight}
    summary = {"documents": len(rows), "labeled": sum(r.label is not None for r in rows),
               "errors": sum(r.pred == ERROR for r in rows), "binary": binary_metrics(rows),
               "docType": doc_type_accuracy(rows), "confusionMatrix": confusion_matrix(rows),
               "ocrMsPerPage": timing_stats(ocr_ms_per_page(rows)),
               "totalMsPerDocument": timing_stats([r.total_ms for r in rows if r.pred != ERROR]), "settings": info}

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) + ["correct"] if rows else ["file"])
        w.writeheader()
        for r in rows:
            w.writerow({**asdict(r), "decided_by": "+".join(r.decided_by), "correct": r.correct})
    report = render_report(rows, info)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    echo("\n" + report.split("## Misclassified")[0])  # names stay in the files, not on screen
    echo(f"Full report: {out_dir / 'report.md'}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.run_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("docs_dir", type=Path)
    p.add_argument("--labels", type=Path)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "out" / time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--limit", type=int)
    p.add_argument("--no-llm", action="store_true", help="rules only (fast)")
    p.add_argument("--no-vision", action="store_true")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from app.config import get_settings
    from app.pipeline import Services
    from app.privacy import apply_offline_env

    apply_offline_env()
    settings = get_settings().model_copy(update={"llm_enabled": not args.no_llm,
                                                 "vision_enabled": not args.no_vision})
    run(args.docs_dir, args.labels, args.out, Services.create(settings), args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
