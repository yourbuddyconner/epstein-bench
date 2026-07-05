"""Pooled retrieval ground truth (TREC-style).

For each verified task: run three diverse retrievers over the full corpus,
union their top-K documents (source docs force-included; aggregation tasks
also include all alias-index candidates), and judge each pooled document as
supports / partial / irrelevant. The gold set is every 'supports' document.

A sample of tasks is re-judged by the strong model; tasks whose sampled
verdicts flip between supports and irrelevant are dropped as unstable.

Relevance sets are pooled, not exhaustive — this is documented in the
dataset card and versioned with the release.
"""

from __future__ import annotations

import random

from .config import Config
from .corpus import load_chunks, load_docs
from .io_utils import read_jsonl, write_jsonl
from .llm import LLM, LLMError
from .retrievers import build_retrievers


def _reference(task: dict) -> str:
    if task["answer"]:
        return task["answer"]
    if task["items"]:
        return "; ".join(i["item"] for i in task["items"])
    return ""


def build_pool(task: dict, retrievers: dict, config: Config) -> list[str]:
    pooled: set[str] = set(task["source_doc_ids"])
    pooled.update(task["provenance"].get("candidate_doc_ids", []))
    for r in retrievers.values():
        pooled.update(
            doc_id for doc_id, _ in r.search(task["question"], config.pool_top_k)
        )
    return sorted(pooled)


def judge_pool(
    task: dict,
    pooled: list[str],
    docs_by_id: dict[str, dict],
    llm: LLM,
    config: Config,
    model: str | None = None,
) -> dict[str, str]:
    """Return {doc_id: 'supports'|'partial'|'irrelevant'}."""
    verdicts: dict[str, str] = {}
    reference = _reference(task)
    for start in range(0, len(pooled), config.pool_judge_batch):
        batch = [d for d in pooled[start : start + config.pool_judge_batch] if d in docs_by_id]
        if not batch:
            continue
        listing = "\n\n".join(
            f"[DOC {i}] id={d}\n{docs_by_id[d]['text'][:1500]}"
            for i, d in enumerate(batch)
        )
        resp = llm.chat_json(
            "[POOLJUDGE] For each document below, judge whether it supports the "
            "reference answer to the question: 'supports' (states the answer or an "
            "item of it), 'partial' (related but insufficient), or 'irrelevant'. "
            f'Respond with JSON {{"verdicts": [str x {len(batch)}]}} in document order.'
            f"\n\nQuestion: {task['question']}\nReference answer: {reference}\n\n{listing}",
            model=model,
        )
        got = resp.get("verdicts") or []
        for i, doc_id in enumerate(batch):
            v = str(got[i]).lower() if i < len(got) else "irrelevant"
            verdicts[doc_id] = v if v in ("supports", "partial", "irrelevant") else "irrelevant"
    return verdicts


def pool_tasks(config: Config, llm: LLM) -> dict[str, int]:
    docs = load_docs(config)
    docs_by_id = {d["doc_id"]: d for d in docs}
    chunks = load_chunks(config)
    retrievers = build_retrievers(config, chunks, llm)
    rng = random.Random(config.seed + 5)

    final: list[dict] = []
    dropped: list[dict] = []
    for task in read_jsonl(config.build_dir / "verified.jsonl"):
        if task["type"] == "unanswerable":
            task["gold_docs"] = []
            final.append(task)
            continue
        try:
            pooled = build_pool(task, retrievers, config)
            verdicts = judge_pool(task, pooled, docs_by_id, llm, config)
            gold = sorted(d for d, v in verdicts.items() if v == "supports")
            # a task whose own source docs aren't judged supportive is suspect
            if not any(d in gold for d in task["source_doc_ids"]):
                dropped.append({"task_id": task["task_id"], "reason": "source_not_supportive"})
                continue
            # stability re-check on a sample
            if rng.random() < config.stability_sample_rate:
                strong = judge_pool(
                    task, pooled, docs_by_id, llm, config, model=config.strong_model
                )
                flips = sum(
                    1
                    for d in pooled
                    if {verdicts.get(d), strong.get(d)} == {"supports", "irrelevant"}
                )
                if flips:
                    dropped.append({"task_id": task["task_id"], "reason": "unstable_pool"})
                    continue
            task["gold_docs"] = gold
            task["provenance"]["pool_size"] = len(pooled)
            task["provenance"]["pool_judge_model"] = config.cheap_model
            final.append(task)
        except LLMError as e:
            dropped.append({"task_id": task["task_id"], "reason": f"error:{e}"})

    write_jsonl(config.build_dir / "pooled.jsonl", final)
    write_jsonl(config.build_dir / "pool_dropped.jsonl", dropped)
    return {"pooled": len(final), "dropped": len(dropped)}
