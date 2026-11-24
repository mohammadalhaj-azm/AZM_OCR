"""Handwriting model backends: the model in this process (transformers), or an OpenAI-compatible server (vLLM).

Default model: sherif1313/Arabic-English-handwritten-OCR-v3, a full fine-tune of Qwen2.5-VL-3B-Instruct (Apache-2.0).
Any Qwen2.5-VL checkpoint works, optionally with a PEFT LoRA adapter (HANDWRITING_ADAPTER) and in 4-bit
(HANDWRITING_QUANTIZATION=4bit: bitsandbytes NF4, CUDA only), e.g. QCRI/Fanar-2-Oryx-IVU with the
mabdulaziz499/arabic-htr-fanar-7b-lora adapter (config/handwriting_models.yaml).
Two things in such repos must never apply silently, so both backends set them on every call:
- generation_config.json turns sampling on (do_sample=true, temperature=1e-6): we always decode greedily;
- the processor allows 12.8 MP images (~16k visual tokens): we cap it with HANDWRITING_MAX_PIXELS.
The sherif model card's `min_new_tokens=50` is deliberately not used: it forces text out of empty pages.
An adapter whose weights don't match the model's layer names would load as a silent no-op (LoRA B matrices start
at zero), so loading checks that its weights really arrived (`adapter_applied`).

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


def required_packages(s: Settings | None = None) -> list[str]:
    pkgs = ["torch", "transformers"]
    if s is not None and s.handwriting_adapter:
        pkgs.append("peft")
    if s is not None and s.handwriting_quantization == "4bit":
        pkgs.append("bitsandbytes")
    return pkgs


def deps_installed(s: Settings | None = None) -> bool:
    return all(importlib.util.find_spec(m) is not None for m in required_packages(s))


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


ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def adapter_path(adapter: str) -> str | None:
    """The local folder of a PEFT adapter (a folder, or its Hugging Face cache snapshot); None if not downloaded."""
    if Path(adapter).is_dir():
        return adapter if all((Path(adapter) / f).is_file() for f in ADAPTER_FILES) else None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    folders = set()
    for f in ADAPTER_FILES:
        try:
            hit = try_to_load_from_cache(adapter, f)
        except ValueError:  # not a valid repo id
            return None
        if not isinstance(hit, str):
            return None
        folders.add(str(Path(hit).parent))
    return folders.pop() if len(folders) == 1 else None


def adapter_applied(named_params) -> tuple[int, int]:
    """(LoRA B matrices, how many are non-zero). PEFT creates every B as zeros and fills it from the adapter file, so
    a B that is still all zeros means its weights didn't load (e.g. layer names that don't match the model)."""
    total = nonzero = 0
    for name, p in named_params:
        if "lora_B" in name:
            total += 1
            t = p.detach() if hasattr(p, "detach") else p
            nonzero += bool(float(abs(t).sum()) > 0)
    return total, nonzero


# transformers >= 4.52 moved Qwen2-VL / Qwen2.5-VL's text layers from `model.*` to `model.language_model.*` and the
# vision tower from `visual.*` to `model.visual.*`. Base checkpoints are renamed on load by transformers, adapters are
# not: an adapter trained on the old layout (e.g. the Fanar HTR LoRA: `base_model.model.model.layers.N...`) matches no
# layer. This is the mapping those transformers versions used (`_checkpoint_conversion_mapping`), which PEFT applies
# to the adapter's keys after stripping its `base_model.model.` prefix. Keys already in the new layout don't match it.
QWEN_VL_ADAPTER_KEY_MAPPING = {
    r"^visual": "model.visual",
    r"^model(?!\.(language_model|visual))": "model.language_model",
}


def adapter_key_mapping(param_names) -> dict[str, str] | None:
    """The key mapping for an old-layout adapter, when the loaded model has the new layout; None otherwise."""
    return QWEN_VL_ADAPTER_KEY_MAPPING if any(n.startswith("model.language_model.") for n in param_names) else None


def load_transformers(s: Settings) -> Loaded:
    if not deps_installed(s):
        raise HandwritingUnavailable(f"the handwriting model needs {', '.join(required_packages(s))}: "
                                     "uv sync --extra handwriting (NVIDIA GPU: --extra handwriting-cuda)")
    if not model_cached(s.handwriting_model):
        raise HandwritingUnavailable("the handwriting model is not downloaded: "
                                     "uv run python -m app.cli download-handwriting-model")
    adapter = None
    if s.handwriting_adapter and (adapter := adapter_path(s.handwriting_adapter)) is None:
        raise HandwritingUnavailable("the handwriting adapter is not downloaded: "
                                     "uv run python -m app.cli download-handwriting-model")
    import torch
    from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

    device = s.handwriting_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise HandwritingUnavailable("HANDWRITING_DEVICE=cuda but no CUDA device is available")
    if s.handwriting_quantization == "4bit" and device != "cuda":
        raise HandwritingUnavailable("HANDWRITING_QUANTIZATION=4bit needs a CUDA GPU")
    dtype = s.handwriting_dtype
    if dtype == "auto":
        dtype = "bfloat16" if device == "cuda" else "float32"  # most CPUs have no fast bf16 matmul
    t0 = time.perf_counter()
    kwargs: dict[str, Any] = {"dtype": getattr(torch, dtype), "local_files_only": True}
    if s.handwriting_quantization == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=getattr(torch, dtype))
        kwargs["device_map"] = {"": 0}  # quantized weights are placed while loading (no .to() afterwards)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(s.handwriting_model, **kwargs)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(s.handwriting_model, **kwargs).to(device)
    base = model.generation_config  # only the token ids are kept from it
    if adapter:
        from peft import PeftModel

        mapping = adapter_key_mapping(n for n, _ in model.named_parameters())
        model = PeftModel.from_pretrained(model, adapter, key_mapping=mapping)
        total, nonzero = adapter_applied(model.named_parameters())
        if total == 0 or nonzero < total:
            log.error("handwriting adapter not applied lora_b=%d nonzero=%d", total, nonzero)
            raise HandwritingUnavailable(f"the handwriting adapter's weights were not applied ({nonzero} of {total} "
                                         "LoRA layers loaded): the adapter doesn't match the model")
    model.eval()
    processor = AutoProcessor.from_pretrained(s.handwriting_model, local_files_only=True, **processor_kwargs(s))
    log.info("handwriting model loaded device=%s dtype=%s quantization=%s adapter=%s ms=%d", device, dtype,
             s.handwriting_quantization, bool(adapter), (time.perf_counter() - t0) * 1000)

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
        out = {"backend": self.name, "depsInstalled": deps_installed(self.s),
               "modelDownloaded": model_cached(self.s.handwriting_model), "loaded": self._loaded is not None,
               "error": self._error}
        if self.s.handwriting_adapter:
            out["adapterDownloaded"] = adapter_path(self.s.handwriting_adapter) is not None
        return out


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
