"""Cut pooled tasks into versioned dataset splits.

Outputs, per split, under ``dataset/<version>/<split>/``:
- ``tasks.jsonl``      — full records including gold (public; self-run benchmark)
- ``questions.jsonl``  — the exact file systems consume: {task_id, type, question}
plus a top-level ``manifest.json`` with counts and questions-file hashes.

``dev`` is a small, fixed subset of ``full`` (tasks whose source documents fall
inside a seeded ~1K-doc subset) for cheap iteration; it is not
leaderboard-eligible.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from . import DATASET_VERSION
from .config import Config
from .corpus import load_docs
from .io_utils import read_jsonl, write_jsonl

DEV_DOC_SUBSET = 1000
DEV_MAX_TASKS = 150


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_dataset(config: Config) -> dict:
    tasks = list(read_jsonl(config.build_dir / "pooled.jsonl"))
    rng = random.Random(config.seed + 6)
    rng.shuffle(tasks)
    tasks = tasks[: config.target_tasks]

    clean_ids = [d["doc_id"] for d in load_docs(config) if d["quality"] == "clean"]
    rng2 = random.Random(config.seed + 7)
    rng2.shuffle(clean_ids)
    dev_docs = set(clean_ids[:DEV_DOC_SUBSET])

    dev = [
        t
        for t in tasks
        if t["source_doc_ids"] and all(d in dev_docs for d in t["source_doc_ids"])
    ][:DEV_MAX_TASKS]
    dev_ids = {t["task_id"] for t in dev}
    # top up dev with unanswerable tasks (they have no source docs)
    for t in tasks:
        if len(dev) >= DEV_MAX_TASKS:
            break
        if t["type"] == "unanswerable" and t["task_id"] not in dev_ids:
            dev.append(t)
            dev_ids.add(t["task_id"])

    out_root = config.dataset_dir / DATASET_VERSION
    manifest: dict = {"version": DATASET_VERSION, "splits": {}}
    for split, split_tasks in (("full", tasks), ("dev", dev)):
        split_dir = out_root / split
        write_jsonl(split_dir / "tasks.jsonl", split_tasks)
        write_jsonl(
            split_dir / "questions.jsonl",
            [
                {"task_id": t["task_id"], "type": t["type"], "question": t["question"]}
                for t in split_tasks
            ],
        )
        by_type: dict[str, int] = {}
        for t in split_tasks:
            by_type[t["type"]] = by_type.get(t["type"], 0) + 1
        manifest["splits"][split] = {
            "n_tasks": len(split_tasks),
            "by_type": by_type,
            "questions_sha256": _sha256(split_dir / "questions.jsonl"),
        }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
