"""Epstein Bench: a RAG benchmark over the public Epstein Files corpus.

The package is a pipeline (generate -> verify -> pool -> finalize -> score),
not a framework. Systems under test never import this code; they consume
``questions.jsonl`` and emit ``predictions.jsonl`` (see docs/methodology.md).
"""

__version__ = "1.1.0"

# Dataset release version. Bump on any change to shipped task files.
DATASET_VERSION = "v1.1"
