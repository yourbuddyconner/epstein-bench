from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.types import Document


class CorpusRetriever:
    """Simple cosine-similarity retriever over chunk embeddings."""

    def __init__(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        embed_fn=None,
    ):
        if not embeddings:
            raise ValueError("Embeddings are required for CorpusRetriever.")

        self.documents = documents
        self.embed_fn = embed_fn

        matrix = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        self.matrix = matrix / norms

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self.embed_fn is None:
            raise ValueError("embed_fn is required for query embeddings.")
        vectors = self.embed_fn(texts)
        arr = np.array(vectors, dtype=float)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / norms

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Return top-k most similar documents for the query text."""
        query_vec = self._embed([query])[0]
        sims = self.matrix @ query_vec
        top_idx = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_idx]

