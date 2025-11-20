"""Configuration helpers for Epstein Bench utilities.

Uses python-dotenv to load environment variables from a .env file and exposes
strongly-typed access via the ``AppConfig`` dataclass. The module is designed
to be imported from notebooks as well as production Python code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

DEFAULT_ENV_FILENAME = ".env"


def _optional_import_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except ModuleNotFoundError:
        return None


@dataclass(frozen=True)
class AppConfig:
    """Represents the runtime configuration for embedding and tagging modules."""

    huggingface_token: Optional[str]
    embeddings_model_name: str
    embeddings_device: str
    tagging_model_name: str
    tagging_max_output_tokens: int
    tagging_temperature: float
    tagging_backend: str
    tagging_device: str
    tagging_max_concurrent_requests: int
    default_device: str

    def resolve_device(self, requested: Optional[str] = None) -> str:
        """Return the device string to use for heavy models."""

        if requested:
            return requested
        if self.default_device:
            return self.default_device

        torch = _optional_import_torch()
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        if torch is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"

    def as_dict(self) -> Dict[str, Any]:
        """Return a serializable dict representation."""

        return asdict(self)



def load_config(env_file: str | Path | None = None, *, override: bool = False) -> AppConfig:
    """Load settings from the given .env file (default: project root)."""

    env_path = Path(env_file) if env_file else Path(DEFAULT_ENV_FILENAME)
    return _load_config_cached(str(env_path.resolve()), override)


def reload_config(env_file: str | Path | None = None, *, override: bool = False) -> AppConfig:
    """Force a reload of environment variables (useful in notebooks)."""

    _load_config_cached.cache_clear()
    return load_config(env_file=env_file, override=override)


@lru_cache(maxsize=4)
def _load_config_cached(env_path: str, override: bool) -> AppConfig:
    load_dotenv(env_path, override=override)

    huggingface_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_API_TOKEN")
    embeddings_model_name = os.getenv(
        "EMBEDDINGS_MODEL_NAME",
        "sentence-transformers/all-mpnet-base-v2",
    )
    embeddings_device = os.getenv("EMBEDDINGS_DEVICE", "")
    tagging_model_name = (
        os.getenv("TAGGING_MODEL_NAME")
        or os.getenv("OPENAI_TAGGING_MODEL")
        or "gpt-4o-mini"
    )
    tagging_max_output_tokens = int(os.getenv("TAGGING_MAX_OUTPUT_TOKENS", "512"))
    tagging_temperature = float(os.getenv("TAGGING_TEMPERATURE", "0.2"))
    tagging_backend = os.getenv("TAGGING_BACKEND", "openai").lower()
    tagging_device = os.getenv("TAGGING_DEVICE", "")
    tagging_max_concurrent_requests = int(os.getenv("TAGGING_MAX_CONCURRENT_REQUESTS", "5"))
    default_device = os.getenv("DEFAULT_DEVICE", "")

    return AppConfig(
        huggingface_token=huggingface_token,
        embeddings_model_name=embeddings_model_name,
        embeddings_device=embeddings_device,
        tagging_model_name=tagging_model_name,
        tagging_max_output_tokens=tagging_max_output_tokens,
        tagging_temperature=tagging_temperature,
        tagging_backend=tagging_backend,
        tagging_device=tagging_device,
        tagging_max_concurrent_requests=tagging_max_concurrent_requests,
        default_device=default_device,
    )

