"""Several handwriting models side by side (HANDWRITING_MODELS_FILE), chosen per request.

The file (see config/handwriting_models.yaml) holds no secrets:

    default: sherif-v3
    models:
      sherif-v3: {model: sherif1313/Arabic-English-handwritten-OCR-v3}
      fanar: {model: QCRI/Fanar-2-Oryx-IVU, adapter: mabdulaziz499/arabic-htr-fanar-7b-lora, quantization: 4bit, ...}

Each entry becomes a copy of the settings with its HANDWRITING_* values replaced; a key that isn't given keeps the
HANDWRITING_* value from the environment. Unknown keys are errors (a typo never drops a setting silently).
Without the file there is one model, `default`, configured by the HANDWRITING_* settings alone.
"""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import ConfigError, Settings

DEFAULT_ID = "default"
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


class ModelEntry(BaseModel):
    """One model. Field names are the HANDWRITING_* settings without the prefix (secrets, like an API key for the
    openai backend, stay in the environment)."""
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    backend: Literal["transformers", "openai"] | None = None
    base_url: str | None = None
    adapter: str | None = None
    quantization: Literal["none", "4bit"] | None = None
    device: Literal["auto", "cpu", "cuda"] | None = None
    dtype: Literal["auto", "float32", "bfloat16", "float16"] | None = None
    segmenter: Literal["whole_page", "lines"] | None = None
    prompt: str | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    max_side: int | None = None
    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    timeout_s: float | None = None
    max_lines: int | None = None


class ModelsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str
    models: dict[str, ModelEntry] = Field(min_length=1)


def model_settings(base: Settings, entry: ModelEntry) -> Settings:
    update = {f"handwriting_{k}": v for k, v in entry.model_dump(exclude_none=True).items()}
    return base.model_copy(update=update)


def load_models(s: Settings) -> tuple[str, dict[str, Settings]]:
    """(default id, {id: settings}) of the configured handwriting models. ConfigError when the file is invalid."""
    if not s.handwriting_models_file:
        return DEFAULT_ID, {DEFAULT_ID: s}
    path = Path(s.handwriting_models_file)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"HANDWRITING_MODELS_FILE {path} can't be read ({type(e).__name__})") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"HANDWRITING_MODELS_FILE {path} is not valid YAML: {e}") from None
    try:
        cfg = ModelsFile.model_validate(raw or {})
    except ValidationError as e:  # the file holds no secrets, so the details can be shown
        raise ConfigError(f"HANDWRITING_MODELS_FILE {path} is invalid: {e}") from None
    for model_id in cfg.models:
        if not MODEL_ID.fullmatch(model_id):
            raise ConfigError(f"handwriting model id {model_id!r}: use 1-64 of a-z 0-9 . _ - "
                              "(starting with a letter or digit)")
    if cfg.default not in cfg.models:
        raise ConfigError(f"HANDWRITING_MODELS_FILE {path}: default {cfg.default!r} is not one of the models "
                          f"({', '.join(cfg.models)})")
    return cfg.default, {model_id: model_settings(s, entry) for model_id, entry in cfg.models.items()}
