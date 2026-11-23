"""OCR throughput benchmark: compares OCR_ENGINES / OCR_CPU_THREADS / OCR_PARALLEL_REC / batch size / DPI.

    uv run python scripts/bench_ocr.py                    # synthetic 8-page scanned A4 PDF (no real documents)
    uv run python scripts/bench_ocr.py D:\\private\\scan.pdf  # your own file (only timings are printed, never text)

Each configuration loads its own engines, warms up on one page, then extracts the whole document. Text quality is
reported as similarity to the baseline configuration's text (1.000 = identical), so a faster setting that reads
worse is visible. Run it on the machine you deploy to: the best setting depends on its cores and memory.
"""

import argparse
import difflib
import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from app.config import Settings  # noqa: E402
from app.extractors import extract  # noqa: E402
from app.extractors.image import decode_images  # noqa: E402
from app.ocr.engine import OcrPool  # noqa: E402
from app.privacy import apply_offline_env  # noqa: E402


def synthetic_pdf(pages: int) -> bytes:
    from tests import synth

    fx = synth.all_fixtures()
    tiles = [Image.open(io.BytesIO(fx[n])).convert("RGB") for n in ("id_card.png", "invoice.png", "letter_mentions_id.png",
                                                                    "passport.png")]
    sheets = []
    for p in range(pages):  # A4 at 300 dpi, three documents stacked per page
        sheet = Image.new("RGB", (2480, 3508), "white")
        y = 150
        for k in range(3):
            tile = tiles[(p + k) % len(tiles)]
            scale = 2000 / tile.width
            tile = tile.resize((2000, int(tile.height * scale)))
            sheet.paste(tile, (240, y))
            y += tile.height + 150
        sheets.append(sheet)
    buf = io.BytesIO()
    sheets[0].save(buf, format="PDF", save_all=True, append_images=sheets[1:], resolution=300)
    return buf.getvalue()


def run(name: str, data: bytes, base: Settings, **over) -> tuple[str, float, int, str]:
    s = base.model_copy(update=over)
    pool = OcrPool(s).load()
    first = decode_images(data, 1)[0] if not data.startswith(b"%PDF") else None
    if first is not None:
        pool.recognize(first)
    else:  # warm up on the first PDF page
        extract(data, s.model_copy(update={"max_pages": 10_000}), pool)
    t0 = time.perf_counter()
    res = extract(data, s, pool)
    secs = time.perf_counter() - t0
    return name, secs, res.timings_ms.get("ocrPages", 0), res.full_text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help="document to benchmark (default: synthetic scanned PDF)")
    ap.add_argument("--pages", type=int, default=8, help="pages of the synthetic PDF")
    ap.add_argument("--quick", action="store_true", help="only baseline vs the main candidates")
    args = ap.parse_args()
    apply_offline_env()

    data = Path(args.file).read_bytes() if args.file else synthetic_pdf(args.pages)
    cores = os.cpu_count() or 4
    base = Settings(_env_file=None, max_pages=10_000)
    configs = [
        ("baseline (1 engine, defaults)", {}),
        ("1 engine, parallel AR/EN rec", {"ocr_parallel_rec": True}),
        (f"2 engines x {cores // 2} threads", {"ocr_engines": 2, "ocr_cpu_threads": cores // 2}),
        (f"3 engines x {max(1, cores // 3)} threads", {"ocr_engines": 3, "ocr_cpu_threads": max(1, cores // 3)}),
        (f"4 engines x {max(1, cores // 4)} threads", {"ocr_engines": 4, "ocr_cpu_threads": max(1, cores // 4)}),
        ("1 engine, rec batch 16", {"ocr_rec_batch_size": 16}),
        ("1 engine, 200 DPI", {"ocr_dpi": 200}),
    ]
    if args.quick:
        configs = configs[:3]

    print(f"cores={cores}  document={'given file' if args.file else f'synthetic {args.pages}-page scanned A4 PDF'}")
    print(f"{'configuration':34} {'seconds':>8} {'s/page':>7} {'speed-up':>8} {'text vs baseline':>17}")
    baseline_secs = baseline_text = None
    for name, over in configs:
        name, secs, pages, text = run(name, data, base, **over)
        baseline_secs = baseline_secs or secs
        baseline_text = text if baseline_text is None else baseline_text
        sim = difflib.SequenceMatcher(None, baseline_text, text, autojunk=False).ratio()
        print(f"{name:34} {secs:8.1f} {secs / max(pages, 1):7.2f} {baseline_secs / secs:7.2f}x {sim:17.3f}",
              flush=True)


if __name__ == "__main__":
    main()
