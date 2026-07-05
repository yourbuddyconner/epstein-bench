"""Score a predictions file against a dataset split.

Headline metric: **cited answer correctness** — the answer matches the
reference (pinned LLM judge) AND at least one cited document is in the gold
supporting set. Per type:

- single_hop / timeline: binary cited correctness
- aggregation: item-level precision/recall/F1; an item counts only when it is
  matched AND a cited document supports that item
- unanswerable: refusal accuracy (a confident answer is a hallucination)

Retrieval diagnostics: recall@5, recall@20, nDCG@10 against pooled gold sets.
Overall = macro-average of the per-type headline numbers. The judge model and
prompt live here and are part of the release: changing them is a new version.
"""

from __future__ import annotations

import math

from .config import Config
from .io_utils import read_jsonl
from .llm import LLM, LLMError


def recall_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = len(set(retrieved[:k]) & set(gold))
    return hits / len(gold)


def ndcg_at_k(retrieved: list[str], gold: list[str], k: int) -> float:
    if not gold:
        return 0.0
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(retrieved[:k])
        if d in gold_set
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def _judge_answer(llm: LLM, config: Config, task: dict, prediction: str) -> dict:
    """Pinned scoring judge. Prompt version: v1 (part of the release)."""
    return llm.chat_json(
        "[SCOREJUDGE] Judge a system's answer for a QA benchmark (judge prompt v1). "
        "Decide: (a) is the answer a refusal/abstention ('cannot be determined', "
        "'not in the corpus', etc.)? (b) if not a refusal, does it state the same "
        "fact as the reference answer (wording may differ)? Respond with JSON "
        '{"correct": true|false, "is_refusal": true|false}.'
        f"\n\nQuestion: {task['question']}\nReference answer: {task.get('answer') or 'N/A'}"
        f"\nSystem answer: {prediction}",
        model=config.judge_model,
    )


def _judge_aggregation(llm: LLM, config: Config, task: dict, prediction: str) -> dict:
    items = [i["item"] for i in task["items"]]
    return llm.chat_json(
        "[AGGJUDGE] A system answered a list-style question (judge prompt v1). "
        "For each gold item, say whether the system's answer includes it (wording "
        "may differ). Also count how many distinct items the system listed that "
        "are NOT among the gold items. Respond with JSON "
        f'{{"matched_items": [true|false x {len(items)}], "extra_items": int}}.'
        f"\n\nQuestion: {task['question']}\nGold items: {items}\nSystem answer: {prediction}",
        model=config.judge_model,
    )


def score_predictions(
    config: Config, llm: LLM, tasks: list[dict], predictions: list[dict]
) -> dict:
    preds_by_id = {p["task_id"]: p for p in predictions}
    missing = [t["task_id"] for t in tasks if t["task_id"] not in preds_by_id]
    if missing:
        raise ValueError(
            f"predictions missing {len(missing)} task(s), e.g. {missing[:3]}"
        )

    per_type_scores: dict[str, list[float]] = {}
    recalls: dict[int, list[float]] = {k: [] for k in config.recall_ks}
    ndcgs: list[float] = []
    errors = 0

    for task in tasks:
        pred = preds_by_id[task["task_id"]]
        answer = str(pred.get("answer") or "")
        citations = [str(c) for c in (pred.get("citations") or [])][: config.max_retrieved]
        retrieved = [str(r) for r in (pred.get("retrieved") or [])][: config.max_retrieved]
        gold = task.get("gold_docs") or []

        if task["type"] != "unanswerable":
            for k in config.recall_ks:
                recalls[k].append(recall_at_k(retrieved, gold, k))
            ndcgs.append(ndcg_at_k(retrieved, gold, config.ndcg_k))

        try:
            score = _score_one(llm, config, task, answer, citations, gold)
        except LLMError:
            errors += 1
            score = 0.0  # fail closed: unscorable predictions earn nothing
        per_type_scores.setdefault(task["type"], []).append(score)

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    per_type = {t: avg(v) for t, v in per_type_scores.items()}
    report = {
        "overall_cited_correctness": avg(list(per_type.values())),
        "per_type": per_type,
        "retrieval": {
            **{f"recall@{k}": avg(v) for k, v in recalls.items()},
            f"ndcg@{config.ndcg_k}": avg(ndcgs),
        },
        "n_tasks": len(tasks),
        "judge_model": config.judge_model,
        "judge_prompt_version": "v1",
        "judge_errors": errors,
    }
    return report


def _score_one(
    llm: LLM,
    config: Config,
    task: dict,
    answer: str,
    citations: list[str],
    gold: list[str],
) -> float:
    if task["type"] == "unanswerable":
        verdict = _judge_answer(llm, config, task, answer or "(empty)")
        return 1.0 if verdict.get("is_refusal") else 0.0

    if not answer:
        return 0.0

    if task["type"] == "aggregation":
        verdict = _judge_aggregation(llm, config, task, answer)
        matched = list(verdict.get("matched_items") or [])
        cited_matched = 0
        for i, item in enumerate(task["items"]):
            if i < len(matched) and matched[i]:
                supported = set(item["doc_ids"]) | set(gold)
                if any(c in supported for c in citations):
                    cited_matched += 1
        extra = max(0, int(verdict.get("extra_items") or 0))
        n_gold = len(task["items"])
        recall = cited_matched / n_gold if n_gold else 0.0
        precision = (
            cited_matched / (cited_matched + extra) if (cited_matched + extra) else 0.0
        )
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # single_hop / timeline
    verdict = _judge_answer(llm, config, task, answer)
    if verdict.get("is_refusal") or not verdict.get("correct"):
        return 0.0
    return 1.0 if any(c in set(gold) for c in citations) else 0.0
