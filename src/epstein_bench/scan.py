"""Wide corpus scan + entity-complete selection (v1.1).

Instead of taking the first N documents, we:

1. ``scan``   — read every (or ``scan_shards``) parquet shard in parallel,
   caching each shard's text rows locally (``build/scan/part_*.jsonl``) with
   the capitalized names they mention. Resumable per shard.
2. ``select`` — aggregate mentions, take the most-mentioned names, ask the
   LLM which are publicly notable people, and build the retrieval corpus as
   ALL documents mentioning the chosen target entities plus a seeded random
   backbone of other documents (the haystack). Targets are persisted to
   ``build/targets.json`` for the dossier generator.

Entity-complete selection is what makes dossier/timeline gold sets honest:
"the documented timeline of X" is only verifiable if every doc mentioning X
is in the corpus.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

from .config import Config
from .corpus import build_corpus, extract_names, _normalize_name
from .io_utils import parallel_map, read_jsonl
from .llm import LLM, LLMError

SCAN_DIR = "scan"


def _scan_shard(config: Config, index: int, path: str) -> dict:
    """Cache one shard's text rows locally. Skips work if the part exists."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out = Path(config.build_dir) / SCAN_DIR / f"part_{index:05d}.jsonl"
    if out.exists():
        return {"part": index, "cached": True}
    tmp = out.with_suffix(".tmp")
    n = 0
    cols = list(config.hf_columns)
    fs = HfFileSystem()
    with fs.open(path, "rb") as f, tmp.open("w", encoding="utf-8") as w:
        pf = pq.ParquetFile(f)
        available = [c for c in cols if c in pf.schema_arrow.names]
        for batch in pf.iter_batches(batch_size=512, columns=available):
            for row in batch.to_pylist():
                text = (row.get(config.hf_text_column) or "").strip()
                if not text:
                    continue
                names = sorted(set(extract_names(text)))  # cased; normalized at select
                w.write(
                    json.dumps(
                        {
                            "doc_id": str(row.get(config.hf_id_column) or f"s{index}r{n}"),
                            "text": text,
                            "names": names,
                            "meta": {
                                k: row[k]
                                for k in ("file_name", "file_type", "online_url")
                                if row.get(k)
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    tmp.rename(out)
    return {"part": index, "docs": n}


def scan_corpus(config: Config) -> dict:
    from huggingface_hub import HfFileSystem

    (Path(config.build_dir) / SCAN_DIR).mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    paths = sorted(fs.glob(f"datasets/{config.hf_dataset}/**/*.parquet"))
    if config.scan_shards:
        paths = paths[: config.scan_shards]
    results = parallel_map(
        lambda ip: _scan_shard(config, ip[0], ip[1]),
        list(enumerate(paths)),
        config.scan_workers,
    )
    total = sum(r.get("docs", 0) for r in results)
    return {"shards": len(paths), "new_docs_cached": total}


def _iter_scan_docs(config: Config) -> Iterator[dict]:
    for part in sorted((Path(config.build_dir) / SCAN_DIR).glob("part_*.jsonl")):
        yield from read_jsonl(part)


def _check_notability(llm: LLM, name: str, aliases: list[str], n_docs: int) -> bool:
    try:
        resp = llm.chat_json(
            "[NOTABLE] Is this a specific, publicly notable PERSON — someone "
            "named in public reporting, public office, business leadership, or "
            "court records (not a company, place, boilerplate phrase, or a "
            "private individual with no public profile)? Respond with JSON "
            '{"notable": true|false}.'
            f"\n\nName: {aliases[0] if aliases else name} (appears in {n_docs} documents)"
        )
        return bool(resp.get("notable"))
    except LLMError:
        return False


def select_corpus(config: Config, llm: LLM) -> dict:
    """Build the retrieval corpus from the scan cache, entity-complete."""
    # aggregate mentions
    doc_count: Counter[str] = Counter()
    aliases: dict[str, set[str]] = defaultdict(set)
    doc_ids: dict[str, list[str]] = defaultdict(list)
    total_docs = 0
    for doc in _iter_scan_docs(config):
        total_docs += 1
        for cased in doc["names"]:
            norm = _normalize_name(cased)
            doc_count[norm] += 1
            aliases[norm].add(cased)
            doc_ids[norm].append(doc["doc_id"])
    if not total_docs:
        raise RuntimeError("scan cache is empty — run `scan` first")

    # notability check on the most-mentioned names
    candidates = [
        (name, n)
        for name, n in doc_count.most_common(config.notability_candidates * 3)
        if config.mention_min_count <= n <= config.max_entity_docs
    ][: config.notability_candidates]
    verdicts = parallel_map(
        lambda c: _check_notability(llm, c[0], sorted(aliases[c[0]]), c[1]),
        candidates,
        config.max_workers,
    )
    targets = [name for (name, _n), ok in zip(candidates, verdicts) if ok][
        : config.n_target_entities
    ]
    if not targets:
        raise RuntimeError("no notable target entities found — corpus too small?")

    target_doc_ids = {d for name in targets for d in doc_ids[name]}

    # backbone: seeded sample of everything else
    rng = random.Random(config.seed + 8)
    other_ids = []
    for doc in _iter_scan_docs(config):
        if doc["doc_id"] not in target_doc_ids:
            other_ids.append(doc["doc_id"])
    rng.shuffle(other_ids)
    backbone = set(other_ids[: config.backbone_docs])
    selected = target_doc_ids | backbone

    (Path(config.build_dir) / "targets.json").write_text(
        json.dumps(
            {
                name: {"aliases": sorted(aliases[name]) or [name], "doc_ids": sorted(set(doc_ids[name]))}
                for name in targets
            },
            ensure_ascii=False,
        )
    )

    def rows() -> Iterator[dict]:
        for doc in _iter_scan_docs(config):
            if doc["doc_id"] in selected:
                yield {
                    config.hf_id_column: doc["doc_id"],
                    config.hf_text_column: doc["text"],
                    **doc.get("meta", {}),
                }

    config.doc_limit = None  # selection defines the corpus; no cap
    stats = build_corpus(config, llm, rows=rows())
    stats.update(
        {
            "scanned_docs": total_docs,
            "target_entities": len(targets),
            "target_docs": len(target_doc_ids),
            "backbone_docs": len(backbone),
        }
    )
    # alias info flows to the entity index too; also fix target aliases from scan
    return stats


def load_targets(config: Config) -> dict:
    return json.loads((Path(config.build_dir) / "targets.json").read_text())
