"""Evaluate the experimental handwriting OCR on hand-verified samples: raw and normalized CER/WER, invented text.

    uv run python -m eval.run_handwriting_eval D:\\hw-samples [--out DIR] [--limit N] [--no-blank-check] [--save-text]

The samples folder must be OUTSIDE this repository and contain manifest.csv (UTF-8, header required):

    file,category,gt_file,verified_by,verified_at,notes
    letters/a01.jpg,ar_handwritten,letters/a01.txt,reviewer-1,2026-11-20,
    misc/blank1.png,blank,,reviewer-1,2026-11-20,empty page
    misc/sig1.jpg,signature_only,misc/sig1.txt,reviewer-1,2026-11-21,only a name is readable

- file / gt_file are relative to the folder. gt_file is a UTF-8 text file with the exact text on the page, typed
  and checked by a person; never copy it from a model's output, or the score measures nothing. It may be left empty
  for blank / signature_only / stamp_only (= no text expected).
- category: ar_handwritten, en_handwritten, mixed, blank, signature_only, stamp_only, sparse, printed_control.
- Only rows with verified_by filled in are run and scored; the others are listed as unverified.

The model settings come from the environment / .env (HANDWRITING_*), with HANDWRITING_ENABLED forced on.
--no-blank-check runs every page through the model, to see what it invents for empty pages without the guard.

Outputs (default eval/out/handwriting-<timestamp>/, git-ignored): results.csv, report.md, summary.json. They contain
file names and scores but no document text, unless --save-text writes each output to texts/ for review. Keep them
private.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from eval.hw_metrics import (
    CATEGORIES,
    GAP_FLAG,
    NO_TEXT_CATEGORIES,
    NORMALIZATION_RULES,
    Sample,
    category_summary,
    invented_text,
    score,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COLUMNS = ("file", "category", "gt_file", "verified_by", "verified_at", "notes")


@dataclass(frozen=True)
class Row:
    line: int
    file: str
    category: str
    gt_file: str
    verified_by: str
    verified_at: str

    @property
    def verified(self) -> bool:
        return bool(self.verified_by)


def load_manifest(path: Path) -> list[Row]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = {h.strip().lower() for h in reader.fieldnames or []}
        if not {"file", "category", "gt_file", "verified_by"} <= header:
            raise SystemExit(f"manifest needs a header with columns: {','.join(COLUMNS)}")
        for i, raw in enumerate(reader, start=2):
            r = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            if not r.get("file"):
                continue
            if r["category"] not in CATEGORIES:
                raise SystemExit(f"manifest line {i}: unknown category {r['category']!r}; use one of "
                                 f"{', '.join(CATEGORIES)}")
            if not r["gt_file"] and r["category"] not in NO_TEXT_CATEGORIES:
                raise SystemExit(f"manifest line {i}: gt_file is required for category {r['category']}")
            rows.append(Row(i, r["file"].replace("\\", "/"), r["category"], r["gt_file"].replace("\\", "/"),
                            r.get("verified_by", ""), r.get("verified_at", "")))
    return rows


def read_reference(folder: Path, row: Row) -> str:
    return (folder / row.gt_file).read_text(encoding="utf-8-sig") if row.gt_file else ""


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.0f}"


def render_report(summary: dict) -> str:
    cfg = summary["config"]
    lines = [
        "# Handwriting OCR evaluation", "",
        f"- Model: `{cfg['model']}` via `{cfg['backend']}`, max pixels {cfg['maxPixels']}, prompt: "
        f"{'custom' if cfg['customPrompt'] else 'model card default'}, blank guard: "
        f"{'on' if cfg['blankCheck'] else 'OFF'}",
        f"- Samples scored: {summary['scored']}; unverified (not run): {len(summary['unverified'])}; "
        f"failed: {summary['failed']}",
        "", "Raw = NFC + whitespace only. Normalized applies only:", "",
        *[f"- {r}" for r in NORMALIZATION_RULES], "",
        "CER/WER are pooled per category (all edits / all reference characters or words).", "",
        "| category | n | raw CER | raw WER | norm CER | norm WER | gap | gap-flagged | inserted share | s/page |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cat, c in summary["categories"].items():
        lines.append(f"| {cat} | {c['n']} | {_fmt(c['rawCer'])} | {_fmt(c['rawWer'])} | {_fmt(c['normCer'])} | "
                     f"{_fmt(c['normWer'])} | {_fmt(c['gap'])} | {c['gapFlagged']} | {_fmt(c['insertedShare'])} | "
                     f"{_fmt(c['msPerPage'] / 1000 if c['msPerPage'] else None, pct=False)} |")
    if summary["inventedText"]:
        lines += ["", "## Invented text on pages that should have (almost) none", "",
                  "| category | n | with any output | excess chars (total) | excess chars (max) | skipped by guard |",
                  "|---|---|---|---|---|---|"]
        for cat, c in summary["inventedText"].items():
            lines.append(f"| {cat} | {c['n']} | {c['withOutput']} | {c['excessChars']} | {c['maxExcessChars']} | "
                         f"{c['skippedByGuard']} |")
    flagged = [r for r in summary["samples"] if r["gapFlag"]]
    if flagged:
        lines += ["", f"## Normalization gap over {100 * GAP_FLAG:.0f} points (review: the normalized score forgives "
                      "a lot here)", ""]
        lines += [f"- {r['file']}: raw CER {_fmt(r['rawCer'])}, normalized {_fmt(r['normCer'])}" for r in flagged]
    warned = [r for r in summary["samples"] if r["warnings"]]
    if warned:
        lines += ["", "## Pages with warnings", ""] + [f"- {r['file']}: {r['warnings']}" for r in warned]
    if summary["unverified"]:
        lines += ["", "## Unverified (not run, not scored)", ""] + [f"- {f}" for f in summary["unverified"]]
    errors = [r for r in summary["samples"] if r["error"]]
    if errors:
        lines += ["", "## Failed", ""] + [f"- {r['file']}: {r['error']}" for r in errors]
    return "\n".join(lines) + "\n"


def run(folder: Path, out_dir: Path, service=None, limit: int | None = None, blank_check: bool = True,
        save_text: bool = False, echo=print, model: str | None = None, segmenter: str | None = None) -> dict:
    folder = folder.resolve()
    if folder.is_relative_to(REPO_ROOT):
        raise SystemExit("refusing to run: the samples folder must be OUTSIDE the repository")
    manifest = folder / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(f"no manifest.csv in {folder}")
    rows = load_manifest(manifest)
    verified = [r for r in rows if r.verified][:limit] if limit else [r for r in rows if r.verified]
    unverified = [r.file for r in rows if not r.verified]

    from app.config import Settings
    from app.extractors.filetype import FileRejected, validate_upload

    if service is None:
        from app.handwriting import HandwritingService
        from app.privacy import apply_offline_env

        apply_offline_env()
        service = HandwritingService(Settings(handwriting_enabled=True, handwriting_blank_check=blank_check))
    chosen, seg = service.resolve(model, segmenter)  # a wrong --model raises UnknownChoice, naming the configured ones
    s = chosen.s
    out_dir.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for n, row in enumerate(verified, 1):
        sample = Sample(row.file, row.category)
        try:
            reference = read_reference(folder, row)
            data = (folder / row.file).read_bytes()
            result = service.transcribe(data, validate_upload(data, s.max_upload_bytes), model, segmenter)
            sample.score = score(reference, result["fullText"])
            sample.pages = len(result["pages"])
            sample.skipped_pages = sum(1 for p in result["pages"] if p.get("skipped"))
            sample.warnings = sorted({w for p in result["pages"] for w in p["warnings"]})
            sample.ms = result["timingsMs"]["total"]
            if save_text:
                dest = out_dir / "texts" / (row.file + ".hyp.txt")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(result["fullText"], encoding="utf-8")
        except FileRejected as e:
            sample.error = e.code
        except (OSError, UnicodeDecodeError) as e:
            sample.error = f"{type(e).__name__}"
        except Exception as e:  # noqa: BLE001 - keep going; report the type only
            sample.error = type(e).__name__
        samples.append(sample)
        sc = sample.score
        echo(f"[{n}/{len(verified)}] {row.file}: " + (f"error {sample.error}" if sample.error else
             f"raw CER {_fmt(sc.raw_cer)}, norm CER {_fmt(sc.norm_cer)}, {sample.ms / 1000:.1f}s"))

    info = chosen.backend.info() if chosen.backend else {}
    summary = {
        "config": {"modelId": chosen.id, "backend": info.get("backend", s.handwriting_backend),
                   "model": s.handwriting_model, "adapter": s.handwriting_adapter or None,
                   "quantization": s.handwriting_quantization, "device": info.get("device"),
                   "maxPixels": s.handwriting_max_pixels, "maxNewTokens": s.handwriting_max_new_tokens,
                   "customPrompt": bool(s.handwriting_prompt), "blankCheck": s.handwriting_blank_check,
                   "segmenter": seg.name},
        "scored": sum(1 for x in samples if x.score is not None),
        "failed": sum(1 for x in samples if x.error),
        "unverified": unverified,
        "categories": {cat: category_summary([x for x in samples if x.category == cat])
                       for cat in CATEGORIES if any(x.category == cat for x in samples)},
        "inventedText": invented_text(samples),
        "samples": [{"file": x.file, "category": x.category, "pages": x.pages, "skippedPages": x.skipped_pages,
                     "warnings": x.warnings, "ms": x.ms, "error": x.error,
                     **({"refChars": x.score.ref_chars, "hypChars": x.score.hyp_chars,
                         "rawCer": x.score.raw_cer, "rawWer": x.score.raw_wer, "normCer": x.score.norm_cer,
                         "normWer": x.score.norm_wer, "gap": x.score.gap, "gapFlag": x.score.gap_flag,
                         "insertions": x.score.raw_char.insertions, "insertedShare": x.score.inserted_share}
                        if x.score else {"gapFlag": False})} for x in samples],
    }
    fields = ["file", "category", "pages", "skippedPages", "warnings", "ms", "error", "refChars", "hypChars",
              "rawCer", "rawWer", "normCer", "normWer", "gap", "gapFlag", "insertions", "insertedShare"]
    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in summary["samples"]:
            w.writerow({**r, "warnings": " ".join(r["warnings"])})
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    echo(f"wrote {out_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval.run_handwriting_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="samples folder with manifest.csv (outside the repository)")
    parser.add_argument("--out", type=Path, help="output folder (default eval/out/handwriting-<timestamp>)")
    parser.add_argument("--limit", type=int, help="score only the first N verified samples")
    parser.add_argument("--no-blank-check", action="store_true",
                        help="send every page to the model (measures invented text without the guard)")
    parser.add_argument("--save-text", action="store_true", help="also write each output to texts/ for review")
    parser.add_argument("--model", help="handwriting model id from HANDWRITING_MODELS_FILE (default: its default)")
    parser.add_argument("--segmenter", choices=["whole_page", "lines"],
                        help="how pages are split for the model (default: the model's segmenter)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{args.model}" if args.model else "")
    out = args.out or REPO_ROOT / "eval" / "out" / f"handwriting-{stamp}"
    run(args.folder, out, limit=args.limit, blank_check=not args.no_blank_check, save_text=args.save_text,
        model=args.model, segmenter=args.segmenter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
