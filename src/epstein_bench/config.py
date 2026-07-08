"""Pinned configuration for the benchmark pipeline.

Everything that affects dataset content or scores lives here so a release is
reproducible from config + seed alone. Judge model and prompt versions are
part of the dataset contract: changing them is a new benchmark version.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# override=True so .env is authoritative over a stale exported OPENAI_API_KEY
# (a rotated key left exported in the shell must not shadow the file).
load_dotenv(override=True)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # --- corpus ---
    # Full public Epstein Files release (~4.1M rows incl. media; only the
    # projected text columns below are ever downloaded).
    hf_dataset: str = "aurora2424/Epstein-Files"
    hf_text_column: str = "text_content"
    hf_id_column: str = "doc_id"
    hf_columns: tuple[str, ...] = (
        "doc_id",
        "file_name",
        "file_type",
        "online_url",
        "text_content",
    )
    doc_limit: int | None = None  # cap on text-bearing docs consumed (dev runs)

    chunk_tokens: int = 512
    chunk_overlap: int = 50

    # quality screen thresholds (fractions of characters/words)
    screen_min_chars: int = 200
    screen_max_garbage_ratio: float = 0.15  # non-ascii-printable chars
    screen_min_dictionary_ratio: float = 0.55  # words found in wordlist
    screen_borderline_band: float = 0.10  # +/- band around thresholds -> LLM check

    # entity alias index
    entity_min_count: int = 3  # min corpus frequency to index a name

    # --- wide scan + entity-complete selection (v1.1) ---
    scan_shards: int | None = None  # parquet shards to scan (None = all)
    scan_workers: int = 16  # parallel shard readers (network-bound)
    mention_min_count: int = 5  # min docs for a name to enter the mention index
    notability_candidates: int = 200  # top names (by doc count) given the LLM check
    n_target_entities: int = 40  # notable entities whose docs are ALL included
    max_entity_docs: int = 2000  # entities above this are too pervasive to be
    # entity-complete targets (e.g. Epstein himself); they stay in the corpus
    # via other targets' docs + backbone but get no dossier
    # hub entities excluded as dossier targets under ANY spelling (OCR variants
    # of Epstein fragment his mentions and slip the doc cap otherwise)
    exclude_entity_substrings: tuple[str, ...] = (
        "epstein",
        "maxwell",
        "morgan chase",  # bank fragment that passed the person-notability check
    )
    backbone_docs: int = 30000  # random non-target docs kept as haystack

    # --- models (pinned) ---
    cheap_model: str = "gpt-4o-mini-2024-07-18"
    strong_model: str = "gpt-4o-2024-08-06"
    # scoring judge; part of the release. A strong model here because correctness
    # judging approaches human agreement at this tier, whereas generation/gauntlet
    # filtering (cheap_model) tolerates a weaker model.
    judge_model: str = "gpt-5.5-2026-04-23"
    embed_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    seed: int = 20260705

    # --- generation targets ---
    target_tasks: int = 1000
    # end-to-end yield measured ~8-20% in shakedown runs; 5x keeps headroom
    oversample_factor: float = 5.0
    type_mix: dict[str, float] = field(
        default_factory=lambda: {
            "single_hop": 0.37,
            "dossier": 0.18,
            "aggregation": 0.15,
            "timeline": 0.10,
            "unanswerable": 0.15,
            "false_premise": 0.05,
        }
    )
    # facts below this newsworthiness rating (1-5) are not made into tasks
    min_salience: int = 3

    # --- verification ---
    answerability_f1_floor: float = 0.10
    aggregation_recovery_floor: float = 0.80  # fraction of items stage 2 must recover

    # --- pooling ---
    pool_top_k: int = 20  # per retriever, doc-level
    pool_judge_batch: int = 8  # docs judged per LLM call
    stability_sample_rate: float = 0.10  # fraction re-judged by strong model

    # --- scoring ---
    recall_ks: tuple[int, ...] = (5, 20)
    ndcg_k: int = 10
    max_retrieved: int = 20
    # only the first N citations count toward the correctness gate, so dumping
    # the whole retrieval list into `citations` cannot game a chance gold hit
    gate_max_citations: int = 3
    bootstrap_iterations: int = 1000  # resamples for score confidence intervals

    # --- paths ---
    build_dir: Path = REPO_ROOT / "build"
    dataset_dir: Path = REPO_ROOT / "dataset"
    cache_dir: Path = REPO_ROOT / "build" / "llm_cache"

    # --- runtime ---
    max_workers: int = 8  # parallel LLM calls in generate/verify/pool stages
    stub_llm: bool = field(
        default_factory=lambda: os.environ.get("EPSTEIN_BENCH_STUB_LLM") == "1"
    )
    openai_api_key: str | None = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    max_llm_retries: int = 5

    def ensure_dirs(self) -> None:
        for d in (self.build_dir, self.dataset_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
