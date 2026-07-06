"""Reference retrievers: BM25, dense (OpenAI embeddings), and RRF hybrid.

Used both for relevance-pool construction and by the reference baselines.
All retrievers index chunks but return *document*-level rankings (max chunk
score per doc), because documents are the citation unit of the benchmark.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .llm import LLM

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over chunks, aggregated to documents."""

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks = chunks
        self.doc_ids = [c["doc_id"] for c in chunks]
        self.token_counts: list[Counter[str]] = []
        self.lengths: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            toks = tokenize(c["text"])
            counts = Counter(toks)
            self.token_counts.append(counts)
            self.lengths.append(len(toks))
            for term in counts:
                self.postings[term].append(i)
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.n = len(chunks)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i in posting:
                tf = self.token_counts[i][term]
                denom = tf + self.k1 * (
                    1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1)
                )
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return _to_doc_ranking(scores, self.doc_ids, k)


class DenseRetriever:
    """Cosine similarity over cached chunk embeddings."""

    def __init__(self, chunks: list[dict], llm: LLM, cache_path: str | Path):
        self.chunks = chunks
        self.doc_ids = [c["doc_id"] for c in chunks]
        self.llm = llm
        cache_path = Path(cache_path)
        if cache_path.exists():
            self.matrix = np.load(cache_path)
            if self.matrix.shape[0] != len(chunks):
                raise ValueError(
                    f"embedding cache {cache_path} has {self.matrix.shape[0]} rows "
                    f"but corpus has {len(chunks)} chunks; delete the cache to rebuild"
                )
        else:
            vecs = llm.embed([c["text"] for c in chunks])
            self.matrix = np.asarray(vecs, dtype=np.float32)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, self.matrix)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.clip(norms, 1e-9, None)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = np.asarray(self.llm.embed([query])[0], dtype=np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-9)
        sims = self.matrix @ q
        scores = {i: float(sims[i]) for i in np.argsort(-sims)[: max(k * 8, 64)]}
        return _to_doc_ranking(scores, self.doc_ids, k)


class HybridRetriever:
    """Reciprocal-rank fusion of any set of retrievers."""

    def __init__(self, retrievers: list, rrf_k: int = 60):
        self.retrievers = retrievers
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        fused: dict[str, float] = defaultdict(float)
        for r in self.retrievers:
            for rank, (doc_id, _score) in enumerate(r.search(query, k * 2)):
                fused[doc_id] += 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return ranked[:k]


def _to_doc_ranking(
    chunk_scores: dict[int, float], doc_ids: list[str], k: int
) -> list[tuple[str, float]]:
    doc_scores: dict[str, float] = defaultdict(float)
    for i, score in chunk_scores.items():
        doc_scores[doc_ids[i]] = max(doc_scores[doc_ids[i]], score)
    ranked = sorted(doc_scores.items(), key=lambda kv: -kv[1])
    return ranked[:k]


def build_retrievers(config: Config, chunks: list[dict], llm: LLM) -> dict[str, object]:
    bm25 = BM25Retriever(chunks)
    # cache keyed by chunk count so a corpus rebuild invalidates it
    dense = DenseRetriever(
        chunks, llm, config.build_dir / f"chunk_embeddings_{len(chunks)}.npy"
    )
    hybrid = HybridRetriever([bm25, dense])
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}
