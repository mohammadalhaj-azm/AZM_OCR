"""Keep the runtime offline. Call before Paddle / Hugging Face code is imported."""

import os

OFFLINE_ENV = {
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",  # no connectivity check to model hosters
    "HF_HUB_OFFLINE": "1",  # huggingface_hub never goes online
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def apply_offline_env() -> None:
    for k, v in OFFLINE_ENV.items():
        os.environ.setdefault(k, v)
