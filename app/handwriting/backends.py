"""Handwriting model backends: the model in this process (transformers), or an OpenAI-compatible server (vLLM).

Model: sherif1313/Arabic-English-handwritten-OCR-v3, a full fine-tune of Qwen2.5-VL-3B-Instruct (Apache-2.0).
Two things in its repo must never apply silently, so both backends set them on every call:
- generation_config.json turns sampling on (do_sample=true, temperature=1e-6): we always decode greedily;
- the processor allows 12.8 MP images (~16k visual tokens): we cap it with HANDWRITING_MAX_PIXELS.
The model card's `min_new_tokens=50` is deliberately not used: it forces text out of empty pages.

The model gives no calibrated confidence, so none is reported (the API returns `confidence: null`).
Never logs prompts, images or model output: only timings and status.
"""

import importlib.util
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)

# The prompt the model was fine-tuned with, verbatim from its model card (including its spelling): "Please extract
# the complete Arabic text from this image from beginning to end, without abbreviation, addition or omission. Read
# all the text content in the image:"
DEFAULT_PROMPT = ("ارجو استخراج النص العربي كاملاً من هذه الصورة من البداية الى النهاية بدون اي اختصار ودون ذيادة "
                  "او حذف. اقرأ كل المحتوى النصي الموجود في الصورة:")


class HandwritingUnavailable(RuntimeError):
    """The handwriting model can't be used right now (disabled, not installed, not downloaded, unreachable).
    The message is safe to return to API clients: it never contains document content."""


@dataclass(frozen=True)
class HwOutput:
    text: str  # as the model produced it (special tokens removed)
    ms: int
    tokens: int | None = None  # generated tokens, when known
    finish_reason: str | None = None  # "stop", or "length" when max_new_tokens cut the output off


class HandwritingBackend(Protocol):
    name: str

    def transcribe(self, img_bgr: np.ndarray) -> HwOutput: ...

    def info(self) -> dict: ...

    def status(self) -> dict: ...

    def load(self) -> Any: ...


def prompt_of(s: Settings) -> str:
    return s.handwriting_prompt or DEFAULT_PROMPT


# --- transformers (in process) ---------------------------------------------------------------------------------------

@dataclass
class Loaded:
    """What `TransformersBackend` needs from torch/transformers; tests pass fakes."""
    model: Any
    processor: Any
    device: str
    dtype: str
    generation_config: Any  # callable(**kwargs) -> a generation config
    inference_mode: Any  # callable() -> context manager


def processor_kwargs(s: Settings) -> dict:
    # Checked with transformers 5.24: max_pixels becomes the image processor's size["longest_edge"] (an A4 page at
    # 200 DPI: 4978 -> 1282 input tokens with the default cap).
    return {"min_pixels": s.handwriting_min_pixels, "max_pixels": s.handwriting_max_pixels}


def generation_kwargs(s: Settings) -> dict:
    """Greedy decoding, set explicitly so the repo's generation_config.json (sampling) never applies. generate()
    fills only the fields left None from that file (checked with transformers 5.24: the mode stays greedy)."""
    return {"do_sample": False, "num_beams": 1, "max_new_tokens": s.handwriting_max_new_tokens,
            "repetition_penalty": s.handwriting_repetition_penalty}


def deps_installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


WEIGHTS_INDEX = "model.safetensors.index.json"
_REQUIRED = ("config.json", "preprocessor_config.json", "tokenizer_config.json")


def model_cached(model: str) -> bool:
    """The whole model is available offline (a local folder, or the Hugging Face cache): config, processor and every
    weight shard listed in the index (a download of only the small files doesn't count)."""
    path = Path(model)
    if path.is_dir():
        def find(name: str) -> str | None:
            return str(path / name) if (path / name).is_file() else None
    else:
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False

        def find(name: str) -> str | None:
            try:
                found = try_to_load_from_cache(model, name)
            except ValueError:  # not a valid repo id (e.g. a folder that doesn't exist)
                return None
            return found if isinstance(found, str) else None
    if not all(find(f) for f in _REQUIRED):
        return False
    if index := find(WEIGHTS_INDEX):
        try:
            shards = set(json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"].values())
        except (OSError, ValueError, KeyError, AttributeError):
            return False
        return bool(shards) and all(find(s) for s in shards)
    return find("model.safetensors") is not None


def load_transformers(s: Settings) -> Loaded:
    if not deps_installed():
        raise HandwritingUnavailable("the handwriting model needs torch and transformers: "
                                     "uv sync --extra handwriting")
    if not model_cached(s.handwriting_model):
        raise HandwritingUnavailable("the handwriting model is not downloaded: "
                                     "uv run python -m app.cli download-handwriting-model")
    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

    device = s.handwriting_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise HandwritingUnavailable("HANDWRITING_DEVICE=cuda but no CUDA device is available")
    dtype = s.handwriting_dtype
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"  # most CPUs have no fast bf16 matmul
    t0 = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        s.handwriting_model, dtype=getattr(torch, dtype), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(s.handwriting_model, local_files_only=True, **processor_kwargs(s))
    base = model.generation_config  # only the token ids are kept from it
    log.info("handwriting model loaded device=%s dtype=%s ms=%d", device, dtype, (time.perf_counter() - t0) * 1000)

    def generation_config(**kw):
        return GenerationConfig(eos_token_id=base.eos_token_id, pad_token_id=base.pad_token_id,
                                bos_token_id=base.bos_token_id, **kw)

    return Loaded(model, processor, device, dtype, generation_config, torch.inference_mode)


class TransformersBackend:
    name = "transformers"

    def __init__(self, settings: Settings, loader=load_transformers):
        self.s = settings
        self._loader = loader
        self._lock = threading.Lock()
        self._loaded: Loaded | None = None
        self._error: str | None = None

    def load(self) -> Loaded:
        with self._lock:
            if self._loaded is None:
                try:
                    self._loaded = self._loader(self.s)
                    self._error = None
                except HandwritingUnavailable as e:
                    self._error = str(e)
                    raise
                except Exception as e:  # noqa: BLE001 - e.g. out of memory, corrupt weights; never the document
                    self._error = f"the handwriting model could not be loaded ({type(e).__name__})"
                    log.error("handwriting model load failed error=%s", type(e).__name__)
                    raise HandwritingUnavailable(self._error) from e
            return self._loaded

    def transcribe(self, img_bgr: np.ndarray) -> HwOutput:
        from PIL import Image

        m = self.load()
        t0 = time.perf_counter()
        image = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))  # BGR -> RGB
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_of(self.s)}]}]
        chat = m.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = m.processor(text=[chat], images=[image], padding=True, return_tensors="pt").to(m.device)
        with m.inference_mode():
            out = m.model.generate(**inputs, generation_config=m.generation_config(**generation_kwargs(self.s)))
        new = out[:, inputs["input_ids"].shape[1]:]
        text = m.processor.batch_decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        n = int(new.shape[1])
        eos = m.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        cut = n >= self.s.handwriting_max_new_tokens and (n == 0 or int(new[0, -1]) not in eos)
        ms = round((time.perf_counter() - t0) * 1000)
        log.info("handwriting transcribe backend=transformers ms=%d tokens=%d", ms, n)
        return HwOutput(text, ms, n, "length" if cut else "stop")

    def info(self) -> dict:
        m = self._loaded
        return {"backend": self.name, "name": self.s.handwriting_model, "device": m.device if m else None}

    def status(self) -> dict:
        return {"backend": self.name, "depsInstalled": deps_installed(),
                "modelDownloaded": model_cached(self.s.handwriting_model), "loaded": self._loaded is not None,
                "error": self._error}


# --- OpenAI-compatible server (vLLM) ---------------------------------------------------------------------------------

class OpenAIBackend:
    name = "openai"

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        headers = {"Authorization": f"Bearer {settings.handwriting_api_key}"} if settings.handwriting_api_key else {}
        self.http = httpx.Client(base_url=settings.handwriting_base_url, timeout=settings.handwriting_timeout_s,
                                 transport=transport, headers=headers)

    def close(self) -> None:
        self.http.close()

    def load(self) -> None:  # nothing to load here: the server holds the model
        return None

    def request_body(self, img_bgr: np.ndarray) -> dict:
        from app.detection.llm import image_data_url

        image = {"type": "image_url", "image_url": {"url": image_data_url(img_bgr, self.s.handwriting_max_side)}}
        return {
            "model": self.s.handwriting_model,
            "messages": [{"role": "user", "content": [image, {"type": "text", "text": prompt_of(self.s)}]}],
            # explicit on every call: never the server's or the model repo's defaults (which sample)
            "temperature": 0,
            "top_p": 1,
            "max_tokens": self.s.handwriting_max_new_tokens,
            "repetition_penalty": self.s.handwriting_repetition_penalty,  # vLLM extension of the OpenAI API
        }

    def transcribe(self, img_bgr: np.ndarray) -> HwOutput:
        body = self.request_body(img_bgr)
        t0 = time.perf_counter()
        try:
            r = self.http.post("/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            log.error("handwriting transcribe backend=openai failed error=%s", type(e).__name__)
            raise HandwritingUnavailable(f"the handwriting model server failed ({type(e).__name__})") from e
        ms = round((time.perf_counter() - t0) * 1000)
        tokens = (data.get("usage") or {}).get("completion_tokens")
        log.info("handwriting transcribe backend=openai ms=%d tokens=%s", ms, tokens)
        return HwOutput(text, ms, tokens, choice.get("finish_reason"))

    def info(self) -> dict:
        return {"backend": self.name, "name": self.s.handwriting_model, "device": None}

    def status(self) -> dict:
        try:
            r = self.http.get("/models", timeout=3)
            r.raise_for_status()
            served = {m.get("id") for m in r.json().get("data", [])}
            return {"backend": self.name, "reachable": True, "modelServed": self.s.handwriting_model in served}
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return {"backend": self.name, "reachable": False, "modelServed": False}


def make_backend(s: Settings) -> HandwritingBackend:
    return OpenAIBackend(s) if s.handwriting_backend == "openai" else TransformersBackend(s)
