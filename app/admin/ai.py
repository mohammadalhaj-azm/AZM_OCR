"""AI monitoring for the admin dashboard: the OCR engines and the text / vision LLMs, measured.

- Live state of this process: OCR engines loaded / busy, LLM calls in flight, errors and recent latency
  (`LlmClient.stats`), and what the LLM server holds in RAM / VRAM right now (Ollama `/api/ps`).
- History across replicas: inference times, throughput, decisions and fallbacks from the audit trail (`stats`).
- Accuracy: the latest `eval.run_eval` summary in EVAL_OUT_DIR (aggregates only; it has no file names).
"""

import json
import time
from pathlib import Path
from urllib.parse import urlsplit

from app.admin.store import percentiles
from app.config import Settings
from app.ocr.engine import OcrEngine

EVAL_KEYS = ("documents", "labeled", "errors", "binary", "docType", "confusionMatrix", "ocrMsPerPage",
             "totalMsPerDocument", "settings")


def endpoint_host(url: str) -> str:
    """scheme://host:port only (never a path or credentials)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    return f"{parts.scheme}://{host}{f':{parts.port}' if parts.port else ''}"


def ocr_status(settings: Settings, services) -> dict:
    engine = getattr(services, "engine", None)
    pool = engine.stats() if hasattr(engine, "stats") else None
    return {"models": {"detection": settings.ocr_det_model, "recognitionAr": settings.ocr_rec_model_ar,
                       "recognitionEn": settings.ocr_rec_model_en or None},
            "languages": ["ar"] + (["en"] if settings.ocr_rec_model_en else []),
            "modelsPresent": OcrEngine(settings).models_present(), "pool": pool,
            "dpi": settings.ocr_dpi, "preprocess": settings.ocr_preprocess, "minScore": settings.ocr_min_score,
            "warmup": settings.ocr_warmup}


def _call_view(stats: dict | None) -> dict | None:
    if not stats:
        return None
    lat = percentiles(stats.pop("latencyMs", []))
    return {**stats, "latency": lat}


def llm_status(settings: Settings, services) -> dict:
    llm = getattr(services, "llm", None)
    models = llm.available_models(max_age_s=5) if llm is not None else set()
    loaded = llm.loaded_models() if hasattr(llm, "loaded_models") else {}
    calls = llm.stats.snapshot() if hasattr(llm, "stats") else {}
    vision_url = settings.llm_vision_base_url or settings.llm_base_url

    def model(role: str, name: str, url: str, enabled: bool) -> dict:
        base = url.rstrip("/")
        held = loaded.get(base)
        mem = next((m for m in held or [] if m["name"] == name), None)
        vram_share = None
        if mem and mem.get("sizeBytes"):
            vram_share = round((mem.get("vramBytes") or 0) / mem["sizeBytes"] * 100, 1)
        return {"role": role, "model": name, "endpoint": endpoint_host(url), "enabled": enabled,
                "reachable": bool(models), "installed": name in models,
                "backend": "ollama" if held is not None else ("unknown" if not models else "openai-compatible"),
                "loaded": None if held is None else mem is not None, "memory": mem, "vramPercent": vram_share,
                "calls": _call_view(calls.get(name))}

    out = [model("text", settings.llm_text_model, settings.llm_base_url, settings.llm_enabled),
           model("vision", settings.llm_vision_model, vision_url, settings.llm_enabled and settings.vision_enabled)]
    others = [m for held in loaded.values() for m in held or []
              if m["name"] not in (settings.llm_text_model, settings.llm_vision_model)]
    return {"models": out, "otherLoaded": others, "timeoutS": {"text": settings.llm_timeout_s,
                                                               "vision": settings.llm_vision_timeout_s}}


def latest_eval(settings: Settings) -> dict | None:
    """The newest EVAL_OUT_DIR/<run>/summary.json, whitelisted to aggregate keys."""
    root = Path(settings.eval_out_dir)
    try:
        runs = sorted((p for p in root.glob("*/summary.json") if p.is_file()), key=lambda p: p.stat().st_mtime)
    except OSError:
        return None
    if not runs:
        return None
    path = runs[-1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {"run": path.parent.name, "at": path.stat().st_mtime, **{k: raw.get(k) for k in EVAL_KEYS if k in raw}}


def ai_overview(settings: Settings, services, store, since: float) -> dict:
    stats = store.stats(since)
    hours = max(1 / 60, (time.time() - since) / 3600)
    finished = stats["events"].get("job.done", 0) + stats["events"].get("extract.done", 0)
    detections = sum(stats["dist"]["decidedBy"].values())
    return {
        "ocr": ocr_status(settings, services), "llm": llm_status(settings, services),
        "performance": {"stages": stats["stages"], "durations": stats["durations"],
                        "throughput": {"documentsPerHour": round(finished / hours, 2),
                                       "pagesPerHour": round(stats["pages"] / hours, 2)}},
        "decisions": {"total": detections, "ids": stats["ids"], "docType": stats["dist"]["docType"],
                      "decidedBy": stats["dist"]["decidedBy"], "confidence": stats["dist"]["confidence"],
                      "fallbacks": stats["dist"]["fallback"], "pageMethods": stats["dist"]["method"],
                      "languages": stats["dist"]["language"]},
        "errors": {"jobFailures": stats["dist"]["error"]},
        "accuracy": latest_eval(settings),
        "settings": {"visionThreshold": settings.vision_threshold, "rulesWeight": settings.rules_weight,
                     "visionWeight": settings.vision_weight, "llmEnabled": settings.llm_enabled,
                     "visionEnabled": settings.vision_enabled},
    }
