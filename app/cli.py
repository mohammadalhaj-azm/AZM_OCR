"""Command-line entry point.

    uv run python -m app.cli extract <file> [--boxes]   # print extraction JSON (--boxes: line bboxes)
    uv run python -m app.cli analyze <file>     # extraction + ID detection (needs Ollama for the LLM step)
    uv run python -m app.cli download-models    # one-time OCR model download (needs internet)
    uv run python -m app.cli download-handwriting-model [--model ID]   # experimental handwriting model(s)
    uv run python -m app.cli audit-ref <externalId>   # the externalIdHash to search the audit trail for
"""

import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    extract_p = sub.add_parser("extract", help="extract text from a file and print JSON")
    extract_p.add_argument("file", type=Path)
    extract_p.add_argument("--boxes", action="store_true", help="add lines with bounding boxes to OCR pages")
    analyze_p = sub.add_parser("analyze", help="extract text and detect identity documents; print JSON")
    analyze_p.add_argument("file", type=Path)
    analyze_p.add_argument("--boxes", action="store_true", help="add lines with bounding boxes to OCR pages")
    sub.add_parser("download-models", help="download the configured PaddleOCR models")
    hw_p = sub.add_parser("download-handwriting-model",
                          help="download the handwriting model(s) into the Hugging Face cache: HANDWRITING_MODEL (and "
                               "HANDWRITING_ADAPTER), or every model and adapter in HANDWRITING_MODELS_FILE")
    hw_p.add_argument("--model", help="one Hugging Face repo id (a model or an adapter) instead")
    ref_p = sub.add_parser("audit-ref", help="print the externalIdHash of a reference, to search the audit trail "
                                             "(uses RESULT_ENCRYPTION_KEY like the service)")
    ref_p.add_argument("external_id")
    args = parser.parse_args(argv)

    if args.command == "audit-ref":
        from app.api.audit import ref_hasher

        print(ref_hasher(get_settings())(args.external_id))
        return 0

    if args.command == "download-handwriting-model":
        return download_handwriting_model(args.model) if args.model else download_handwriting_models()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic on Windows consoles

    from app.privacy import apply_offline_env

    if args.command != "download-models":
        apply_offline_env()

    from app.extractors import FileRejected, extract
    from app.ocr.engine import OcrEngine, OcrPool

    settings = get_settings()
    if args.command == "download-models":
        OcrEngine(settings, allow_download=True).load()
        print("OCR models ready")
        return 0

    try:
        if args.command == "analyze":
            from app.pipeline import Services, analyze

            out = analyze(args.file.read_bytes(), args.file.name, Services.create(settings), include_boxes=args.boxes)
        else:
            result = extract(args.file.read_bytes(), settings, OcrPool(settings))
            out = {"file": {"name": args.file.name, "mimeType": result.mime_type, "pages": len(result.pages)},
                   **result.to_json_dict(include_lines=args.boxes)}
    except FileRejected as e:
        print(json.dumps({"error": e.code, "message": str(e)}), file=sys.stderr)
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def handwriting_repos(settings) -> list[str]:
    """Every repo the configured handwriting models need (base models and adapters), each once, in order. Models
    on the openai backend are served by another server and are left out."""
    from app.handwriting.registry import load_models

    _, models = load_models(settings)
    repos = []
    for s in models.values():
        if s.handwriting_backend == "openai":
            continue
        for repo in (s.handwriting_model, s.handwriting_adapter):
            if repo and repo not in repos:
                repos.append(repo)
    return repos


def download_handwriting_models() -> int:
    for repo in handwriting_repos(get_settings()):
        download_handwriting_model(repo)
    return 0


def download_handwriting_model(model: str) -> int:
    """One-time download (needs internet); the service itself loads it offline (local_files_only)."""
    if Path(model).is_dir():
        print(f"{model} is a local folder: nothing to download")
        return 0
    from huggingface_hub import snapshot_download

    # this command only: the service keeps HF_HUB_OFFLINE=1 (app/privacy.py)
    # model: config, processor, tokenizer, chat template, weights; adapter: adapter_config.json + its safetensors
    path = snapshot_download(model, allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors"])
    print(f"handwriting model ready: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
