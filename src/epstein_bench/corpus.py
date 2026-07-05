"""Corpus preparation: stream, quality-screen, chunk, and index entities.

Source: ``aurora2424/Epstein-Files`` — the full public Epstein Files release
(~4.1M rows, 340GB including raw media bytes). We stream with parquet column
projection so only the text columns are ever downloaded; rows without
``text_content`` (images, audio, video) are skipped. The text-bearing rows
form the retrieval corpus.

Outputs (under ``build/``), written incrementally (the corpus does not fit
comfortably in memory at full scale):
- ``docs.jsonl``      — {doc_id, text, quality, meta}
- ``chunks.jsonl``    — {chunk_id, doc_id, text}
- ``entities.json``   — {normalized_name: {"aliases": [...], "doc_ids": [...]}}

Quality verdicts: tasks are generated only from ``clean`` docs; ``degraded``
docs stay in the retrieval corpus as natural distractors; ``garbage`` is
excluded from everything.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from .config import Config
from .io_utils import read_jsonl
from .llm import LLM

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_PRINTABLE_RE = re.compile(r"[\x20-\x7e\n\r\t]")

# Small built-in wordlist fallback keeps the screen dependency-free; NLTK's
# `words` corpus is used when available (much better dictionary ratio signal).
_FALLBACK_WORDS = frozenset(
    "the of and to in a is that for it as was with be by on not he this are or "
    "his from at which but have an had they you were her all she there would "
    "their we him been has when who will no more if out so said what up its "
    "about into than them can only other time new some could these two may "
    "then do first any my now such like our over man me even most made after "
    "also did many before must through years where much your way well down "
    "should because each just those people mr how too little state good very "
    "make world still own see men work long get here between both life being "
    "under never day same another know while last might us great old year off "
    "come since against go came right used take three email sent subject "
    "please thank thanks best regards dear call meeting house committee".split()
)


def _load_wordlist() -> frozenset[str]:
    try:
        from nltk.corpus import words

        return frozenset(w.lower() for w in words.words())
    except Exception:  # noqa: BLE001 - corpus not downloaded; fallback is fine
        return _FALLBACK_WORDS


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    printable = len(_PRINTABLE_RE.findall(text))
    return 1.0 - printable / len(text)


def dictionary_ratio(text: str, wordlist: frozenset[str]) -> float:
    tokens = _WORD_RE.findall(text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in wordlist)
    return hits / len(tokens)


def screen_heuristic(text: str, config: Config, wordlist: frozenset[str]) -> str:
    """Return 'clean' | 'degraded' | 'garbage' | 'borderline' (needs LLM check)."""
    if len(text) < config.screen_min_chars:
        return "garbage"
    g = garbage_ratio(text)
    d = dictionary_ratio(text, wordlist)
    if g > config.screen_max_garbage_ratio + config.screen_borderline_band:
        return "garbage"
    if d < config.screen_min_dictionary_ratio - config.screen_borderline_band:
        return "degraded"
    borderline = (
        abs(g - config.screen_max_garbage_ratio) <= config.screen_borderline_band
        or abs(d - config.screen_min_dictionary_ratio) <= config.screen_borderline_band
    )
    if borderline:
        return "borderline"
    if g <= config.screen_max_garbage_ratio and d >= config.screen_min_dictionary_ratio:
        return "clean"
    return "degraded"


def resolve_borderline(text: str, llm: LLM) -> str:
    try:
        verdict = llm.chat_json(
            "[READABILITY] Is the following OCR'd document text readable enough that "
            "a careful human could reliably extract facts from it? Respond with JSON "
            '{"readable": true|false}.\n\n---\n' + text[:4000]
        )
        return "clean" if verdict.get("readable") else "degraded"
    except Exception:  # noqa: BLE001 - fail closed: unresolved -> degraded
        return "degraded"


def screen_document(
    text: str, config: Config, wordlist: frozenset[str], llm: LLM | None = None
) -> str:
    """Return 'clean' | 'degraded' | 'garbage' (borderline resolved via LLM)."""
    verdict = screen_heuristic(text, config, wordlist)
    if verdict == "borderline":
        return resolve_borderline(text, llm) if llm is not None else "degraded"
    return verdict


def chunk_text(text: str, chunk_tokens: int, overlap: int) -> list[str]:
    """Whitespace-token windows; a doc under one window stays whole."""
    tokens = text.split()
    if len(tokens) <= chunk_tokens:
        return [text] if text.strip() else []
    step = max(1, chunk_tokens - overlap)
    return [
        " ".join(tokens[i : i + chunk_tokens]) for i in range(0, len(tokens), step)
    ]


# -- entity alias index -----------------------------------------------------------

_NAME_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")
_NAME_STOPWORDS = frozenset(
    "The This That From Sent To Subject Dear Best Thank Thanks Please New York "
    "United States House Oversight Palm Beach Monday Tuesday Wednesday Thursday "
    "Friday Saturday Sunday January February March April May June July August "
    "September October November December".split()
)


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().split())


def extract_names(text: str) -> list[str]:
    """Capitalized multi-word sequences that don't start/end with stopwords."""
    out = []
    for m in _NAME_RE.finditer(text):
        parts = m.group(1).split()
        if parts[0] in _NAME_STOPWORDS or parts[-1] in _NAME_STOPWORDS:
            continue
        out.append(m.group(1))
    return out


class EntityAccumulator:
    """Streaming builder for the entity alias index."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.aliases: dict[str, set[str]] = defaultdict(set)
        self.doc_ids: dict[str, set[str]] = defaultdict(set)

    def add_doc(self, doc_id: str, text: str) -> None:
        seen: set[str] = set()
        for name in extract_names(text):
            norm = _normalize_name(name)
            self.aliases[norm].add(name)
            self.doc_ids[norm].add(doc_id)
            if norm not in seen:
                self.counts[norm] += 1
                seen.add(norm)

    def build(self, min_count: int) -> dict[str, dict[str, list[str]]]:
        return {
            norm: {
                "aliases": sorted(self.aliases[norm]),
                "doc_ids": sorted(self.doc_ids[norm]),
            }
            for norm, c in self.counts.items()
            if c >= min_count
        }


def build_entity_index(
    docs: list[dict], min_count: int
) -> dict[str, dict[str, list[str]]]:
    acc = EntityAccumulator()
    for doc in docs:
        if doc["quality"] != "garbage":
            acc.add_doc(doc["doc_id"], doc["text"])
    return acc.build(min_count)


# -- corpus streaming ---------------------------------------------------------------


def _stream_hf_rows(config: Config) -> Iterator[dict]:
    """Stream text-bearing rows with true parquet column projection.

    We read the dataset's parquet shards directly with pyarrow over
    HfFileSystem: `iter_batches(columns=...)` issues HTTP range reads for the
    projected column chunks only. (`datasets` streaming pulls whole row
    groups — media bytes included, ~83KB/row on this 340GB dataset — which
    measured ~30 docs/min; direct projection skips the media entirely.)
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{config.hf_dataset}/**/*.parquet"))
    if not paths:
        raise RuntimeError(f"no parquet shards found for {config.hf_dataset}")
    cols = list(config.hf_columns)
    for path in paths:
        with fs.open(path, "rb") as f:
            pf = pq.ParquetFile(f)
            available = [c for c in cols if c in pf.schema_arrow.names]
            for batch in pf.iter_batches(batch_size=512, columns=available):
                yield from batch.to_pylist()


def build_corpus(
    config: Config, llm: LLM, rows: list[dict] | Iterator[dict] | None = None
) -> dict[str, int]:
    """Stream the corpus, screen, chunk, and index entities.

    Three passes so LLM readability checks parallelize instead of blocking the
    stream: (1) stream rows -> docs_raw.jsonl with heuristic verdicts;
    (2) resolve 'borderline' docs with parallel LLM calls; (3) write final
    docs/chunks/entities from disk. ``rows`` bypasses the HuggingFace download
    for tests/fixtures.
    """
    from .io_utils import parallel_map, read_jsonl

    config.ensure_dirs()
    row_iter: Iterator[dict] = iter(rows) if rows is not None else _stream_hf_rows(config)
    wordlist = _load_wordlist()

    # pass 1: stream + heuristic screen
    raw_path = Path(config.build_dir) / "docs_raw.jsonl"
    n_docs = 0
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for idx, row in enumerate(row_iter):
            text = (row.get(config.hf_text_column) or "").strip()
            if not text:
                continue  # image/audio/video rows carry no text
            if config.doc_limit and n_docs >= config.doc_limit:
                break
            doc_id = str(row.get(config.hf_id_column) or f"doc_{idx}")
            meta = {
                k: row[k]
                for k in ("file_name", "file_type", "online_url")
                if row.get(k)
            }
            quality = screen_heuristic(text, config, wordlist)
            raw_f.write(
                json.dumps(
                    {"doc_id": doc_id, "text": text, "quality": quality, "meta": meta},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_docs += 1

    # pass 2: resolve borderline docs with parallel LLM readability checks
    borderline = [
        (d["doc_id"], d["text"]) for d in read_jsonl(raw_path) if d["quality"] == "borderline"
    ]
    resolved = dict(
        zip(
            (doc_id for doc_id, _ in borderline),
            parallel_map(
                lambda p: resolve_borderline(p[1], llm), borderline, config.max_workers
            ),
        )
    )

    # pass 3: final artifacts
    acc = EntityAccumulator()
    quality_counts: Counter[str] = Counter()
    n_chunks = 0
    docs_path = Path(config.build_dir) / "docs.jsonl"
    chunks_path = Path(config.build_dir) / "chunks.jsonl"
    with docs_path.open("w", encoding="utf-8") as docs_f, chunks_path.open(
        "w", encoding="utf-8"
    ) as chunks_f:
        for doc in read_jsonl(raw_path):
            if doc["quality"] == "borderline":
                doc["quality"] = resolved.get(doc["doc_id"], "degraded")
            docs_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            quality_counts[doc["quality"]] += 1
            if doc["quality"] == "garbage":
                continue
            acc.add_doc(doc["doc_id"], doc["text"])
            for i, chunk in enumerate(
                chunk_text(doc["text"], config.chunk_tokens, config.chunk_overlap)
            ):
                chunks_f.write(
                    json.dumps(
                        {
                            "chunk_id": f"{doc['doc_id']}#{i}",
                            "doc_id": doc["doc_id"],
                            "text": chunk,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_chunks += 1
    raw_path.unlink()

    entities = acc.build(config.entity_min_count)
    (Path(config.build_dir) / "entities.json").write_text(
        json.dumps(entities, ensure_ascii=False)
    )

    return {
        "docs": n_docs,
        "clean": quality_counts.get("clean", 0),
        "degraded": quality_counts.get("degraded", 0),
        "garbage": quality_counts.get("garbage", 0),
        "chunks": n_chunks,
        "entities": len(entities),
        "borderline_resolved": len(borderline),
    }


def load_docs(config: Config) -> list[dict]:
    return list(read_jsonl(config.build_dir / "docs.jsonl"))


def load_chunks(config: Config) -> list[dict]:
    return list(read_jsonl(config.build_dir / "chunks.jsonl"))


def load_entities(config: Config) -> dict:
    return json.loads((Path(config.build_dir) / "entities.json").read_text())
