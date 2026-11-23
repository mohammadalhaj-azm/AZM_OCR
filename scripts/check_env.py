"""Phase 0 smoke test: checks Ollama (text + vision) and PaddleOCR, and measures speed.

Uses only a synthetic, generated image with fake data. Run:
    uv run python scripts/check_env.py [--skip-llm] [--skip-vision] [--skip-ocr]
"""

import argparse
import base64
import io
import json
import sys
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, ".")
from app.config import get_settings  # noqa: E402

SCHEMA = {
    "type": "object",
    "properties": {
        "isId": {"type": "boolean"},
        "docType": {
            "type": "string",
            "enum": ["national_id", "passport", "driver_license", "residence_permit", "other_id", "not_id"],
        },
        "confidence": {"type": "number"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["isId", "docType", "confidence", "reasons"],
}

SAMPLE_TEXT = (
    "جواز سفر PASSPORT\n"
    "الاسم: سامي تجريبي  Name: SAMPLE TESTER\n"
    "تاريخ الميلاد Date of Birth: 01/02/1990\n"
    "P<XXXTESTER<<SAMPLE<<<<<<<<<<<<<<<<<<<<<<<<<"
)


def synthetic_image() -> Image.Image:
    """A fake, ID-like card with mixed Arabic/English text."""
    import arabic_reshaper
    from bidi.algorithm import get_display

    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
    img = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(SAMPLE_TEXT.splitlines()):
        shaped = get_display(arabic_reshaper.reshape(line))
        draw.text((40, 30 + i * 95), shaped, fill="black", font=font)
    return img


def chat(client: httpx.Client, model: str, content, schema: bool = True) -> tuple[dict, float]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Classify the document. Reply only with JSON matching the schema."},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "reasoning_effort": "none",
    }
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "id_detection", "schema": SCHEMA, "strict": True},
        }
    t0 = time.perf_counter()
    r = client.post("/chat/completions", json=body)
    r.raise_for_status()
    return r.json(), time.perf_counter() - t0


def report_chat(label: str, data: dict, secs: float) -> None:
    msg = data["choices"][0]["message"]
    usage = data.get("usage", {})
    out_toks = usage.get("completion_tokens", 0)
    content = msg.get("content", "")
    try:
        parsed = json.loads(content)
        valid = all(k in parsed for k in SCHEMA["required"])
    except json.JSONDecodeError:
        parsed, valid = None, False
    print(f"  [{label}] {secs:.1f}s total | prompt_tokens={usage.get('prompt_tokens')} "
          f"completion_tokens={out_toks} | ~{out_toks / secs:.1f} out-tok/s (incl. prompt processing)")
    print(f"  [{label}] thinking present: {bool(msg.get('reasoning') or msg.get('reasoning_content'))} "
          f"| schema-valid JSON: {valid}")
    print(f"  [{label}] result: {parsed if parsed is not None else content[:300]}")


def check_llm(settings, img: Image.Image, do_text: bool, do_vision: bool) -> None:
    with httpx.Client(base_url=settings.llm_base_url, timeout=settings.llm_timeout_s,
                      headers={"Authorization": f"Bearer {settings.llm_api_key}"}) as client:
        models = {m["id"] for m in client.get("/models").json()["data"]}
        print(f"Ollama reachable at {settings.llm_base_url}; models: {sorted(models)}")

        if do_text:
            print(f"\nText LLM: {settings.llm_text_model}")
            data, secs = chat(client, settings.llm_text_model, "hi", schema=False)
            print(f"  cold load + tiny reply: {secs:.1f}s")
            data, secs = chat(client, settings.llm_text_model, SAMPLE_TEXT)
            report_chat("warm", data, secs)

        if do_vision:
            if settings.llm_vision_model not in models:
                print(f"\nVision model {settings.llm_vision_model} not installed; skipping.")
                return
            print(f"\nVision LLM: {settings.llm_vision_model}")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content = [
                {"type": "text", "text": "Is this an identity document? Answer with the JSON schema."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
            data, secs = chat(client, settings.llm_vision_model, content)
            report_chat("vision (incl. load)", data, secs)


def check_ocr(img: Image.Image) -> None:
    from paddleocr import PaddleOCR

    print("\nPaddleOCR (lang=ar)")
    t0 = time.perf_counter()
    ocr = PaddleOCR(lang="ar", use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False)
    print(f"  model init: {time.perf_counter() - t0:.1f}s")
    arr = np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR
    ocr.predict(arr)  # warm-up
    t0 = time.perf_counter()
    results = ocr.predict(arr)
    secs = time.perf_counter() - t0
    for res in results:
        texts, scores = res["rec_texts"], res["rec_scores"]
        print(f"  {len(texts)} text boxes in {secs:.2f}s (warm, 1200x420 image)")
        for t, s in zip(texts, scores):
            print(f"    {s:.3f}  {t}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-ocr", action="store_true")
    args = p.parse_args()

    settings = get_settings()
    img = synthetic_image()
    if not args.skip_llm or not args.skip_vision:
        check_llm(settings, img, not args.skip_llm, not args.skip_vision)
    if not args.skip_ocr:
        check_ocr(img)
    return 0


if __name__ == "__main__":
    sys.exit(main())
