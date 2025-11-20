"""Utility modules for working with the Epstein Bench project.

This package exposes helpers for configuration management, text embeddings,
and switchable (OpenAI or local GPT-OSS) document tagging so they can be reused
across notebooks, CLIs, and batch jobs.
"""

from .config import AppConfig, load_config, reload_config

try:  # Optional, some environments don't have sentence_transformers installed.
    from .embeddings import EmbeddingClient
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    EmbeddingClient = None  # type: ignore[assignment]

from .tagging import GPTOSSNERTagger, TaggingResult

__all__ = [
    "AppConfig",
    "EmbeddingClient",
    "GPTOSSNERTagger",
    "TaggingResult",
    "load_config",
    "reload_config",
]

