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

import numpy as np

from .config import Config
from .io_utils import parallel_map, read_jsonl
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


def _judge_premise_id(llm: LLM, config: Config, task: dict, prediction: str) -> bool:
    """Did the system flag the *specific* false presupposition (not just refuse)?

    Only meaningful for `false_premise` tasks that were refused; drives the
    `premise_id_rate` diagnostic, never the headline.
    """
    verdict = llm.chat_json(
        "[PREMISEID] A question presupposed a fact that is FALSE / unsupported by "
        "the source corpus. Did the system's response correctly identify or flag "
        "that specific false presupposition, rather than merely declining in "
        'general? Respond with JSON {"identified": true|false}.'
        f"\n\nQuestion: {task['question']}"
        f"\nFalse presupposition: {task.get('false_element') or 'N/A'}"
        f"\nSystem answer: {prediction}",
        model=config.judge_model,
    )
    return bool(verdict.get("identified"))


def _citation_gold(task: dict, gold: list[str]) -> set[str]:
    """Documents that would make a citation 'correct' for citation P/R.

    Pooled gold, plus per-item supporting docs for list types. Empty for the
    no-gold types (unanswerable / false_premise), where citation P/R is undefined.
    """
    gold_set = set(gold)
    for item in task.get("items") or []:
        gold_set |= set(item.get("doc_ids") or [])
    return gold_set


def _citation_pr(citations: list[str], gold_set: set[str]) -> tuple[float, float]:
    """ALCE-style citation precision and recall for one prediction."""
    hits = len(set(citations) & gold_set)
    precision = hits / len(citations) if citations else 0.0
    recall = hits / len(gold_set) if gold_set else 0.0
    return precision, recall


def _stratified_bootstrap_ci(
    values_by_type: dict[str, list[float]], rng: np.random.Generator, iters: int
) -> list[float]:
    """95% CI on the macro-average, resampling within each type.

    Stratified (per-type) resampling keeps every type present in every draw, so
    the macro structure is stable and the interval reflects within-type variance
    — which is dominated by the tiny-n types (dossier n=7, timeline n=27).
    """
    types = [t for t, v in values_by_type.items() if v]
    if not types:
        return [0.0, 0.0]
    arrays = {t: np.asarray(values_by_type[t], dtype=float) for t in types}
    macros = np.empty(iters, dtype=float)
    for b in range(iters):
        per_type_means = [
            arrays[t][rng.integers(0, len(arrays[t]), len(arrays[t]))].mean()
            for t in types
        ]
        macros[b] = float(np.mean(per_type_means))
    lo, hi = np.percentile(macros, [2.5, 97.5])
    return [round(float(lo), 6), round(float(hi), 6)]


def score_predictions(
    config: Config, llm: LLM, tasks: list[dict], predictions: list[dict]
) -> dict:
    preds_by_id = {p["task_id"]: p for p in predictions}
    missing = [t["task_id"] for t in tasks if t["task_id"] not in preds_by_id]
    if missing:
        raise ValueError(
            f"predictions missing {len(missing)} task(s), e.g. {missing[:3]}"
        )

    # types that carry no gold documents (abstention/rejection tasks): excluded
    # from retrieval and citation diagnostics
    NO_GOLD = ("unanswerable", "false_premise")

    def judge_task(task: dict) -> dict:
        pred = preds_by_id[task["task_id"]]
        answer = str(pred.get("answer") or "")
        citations = [str(c) for c in (pred.get("citations") or [])][: config.max_retrieved]
        gold = task.get("gold_docs") or []
        out = {
            "type": task["type"],
            "cited": 0.0,
            "uncited": 0.0,
            "error": False,
            "cit_prec": None,
            "cit_rec": None,
            "premise_refused": None,
            "premise_id": None,
        }
        try:
            cited, uncited = _score_one(llm, config, task, answer, citations, gold)
            out["cited"], out["uncited"] = cited, uncited
            if task["type"] not in NO_GOLD:
                prec, rec = _citation_pr(citations, _citation_gold(task, gold))
                out["cit_prec"], out["cit_rec"] = prec, rec
            if task["type"] == "false_premise":
                refused = uncited >= 1.0  # score is 1.0 iff the system refused
                out["premise_refused"] = refused
                if refused:
                    out["premise_id"] = _judge_premise_id(llm, config, task, answer)
        except LLMError:
            out["error"] = True  # fail closed: unscorable earns nothing
        return out

    recalls: dict[int, list[float]] = {k: [] for k in config.recall_ks}
    ndcgs: list[float] = []

    for task in tasks:
        if task["type"] not in NO_GOLD:
            pred = preds_by_id[task["task_id"]]
            retrieved = [str(r) for r in (pred.get("retrieved") or [])][: config.max_retrieved]
            gold = task.get("gold_docs") or []
            for k in config.recall_ks:
                recalls[k].append(recall_at_k(retrieved, gold, k))
            ndcgs.append(ndcg_at_k(retrieved, gold, config.ndcg_k))

    judged = parallel_map(judge_task, tasks, config.max_workers)
    errors = sum(1 for r in judged if r["error"])
    per_type_scores: dict[str, list[float]] = {}
    uncited_scores: dict[str, list[float]] = {}
    cit_precs: list[float] = []
    cit_recs: list[float] = []
    premise_refused = 0
    premise_identified = 0
    for r in judged:
        per_type_scores.setdefault(r["type"], []).append(r["cited"])
        uncited_scores.setdefault(r["type"], []).append(r["uncited"])
        if r["cit_prec"] is not None:
            cit_precs.append(r["cit_prec"])
            cit_recs.append(r["cit_rec"])
        if r["premise_refused"]:
            premise_refused += 1
            if r["premise_id"]:
                premise_identified += 1

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    per_type = {t: avg(v) for t, v in per_type_scores.items()}
    per_type_uncited = {t: avg(v) for t, v in uncited_scores.items()}
    all_cited = [r["cited"] for r in judged]
    all_uncited = [r["uncited"] for r in judged]
    rng = np.random.default_rng(config.seed)
    report = {
        "overall_cited_correctness": avg(list(per_type.values())),
        "overall_cited_correctness_ci95": _stratified_bootstrap_ci(
            per_type_scores, rng, config.bootstrap_iterations
        ),
        # task-weighted (micro) counterpart to the macro headline, so a tiny-n
        # type (dossier n=7) does not silently dominate the reported number
        "overall_cited_correctness_micro": avg(all_cited),
        "per_type": per_type,
        # correctness ignoring the citation gate — for closed-book systems this
        # measures parametric knowledge of the corpus (training contamination)
        "overall_uncited_correctness": avg(list(per_type_uncited.values())),
        "overall_uncited_correctness_ci95": _stratified_bootstrap_ci(
            uncited_scores, rng, config.bootstrap_iterations
        ),
        "overall_uncited_correctness_micro": avg(all_uncited),
        "per_type_uncited": per_type_uncited,
        # ALCE-style attribution quality, over answerable tasks
        "citation_precision": avg(cit_precs),
        "citation_recall": avg(cit_recs),
        "retrieval": {
            **{f"recall@{k}": avg(v) for k, v in recalls.items()},
            f"ndcg@{config.ndcg_k}": avg(ndcgs),
        },
        "n_tasks": len(tasks),
        "judge_model": config.judge_model,
        "judge_prompt_version": "v1",
        "judge_errors": errors,
    }
    # false_premise: of the tasks a system refused, how often did it name the
    # specific false presupposition (diagnostic, not part of the headline)
    if premise_refused:
        report["premise_id_rate"] = premise_identified / premise_refused
        report["premise_refused_n"] = premise_refused
    return report


def _item_f1(matched_count: int, extra: int, n_gold: int) -> float:
    recall = matched_count / n_gold if n_gold else 0.0
    precision = (
        matched_count / (matched_count + extra) if (matched_count + extra) else 0.0
    )
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _score_one(
    llm: LLM,
    config: Config,
    task: dict,
    answer: str,
    citations: list[str],
    gold: list[str],
) -> tuple[float, float]:
    """Return (cited, uncited) scores.

    Cited is the headline (grounding required). Uncited ignores the citation
    gate — for closed-book/parametric systems it measures how much of the
    corpus the model already knows from training.
    """
    # abstention/rejection types: the target behavior is to refuse
    if task["type"] in ("unanswerable", "false_premise"):
        verdict = _judge_answer(llm, config, task, answer or "(empty)")
        score = 1.0 if verdict.get("is_refusal") else 0.0
        return score, score

    if not answer:
        return 0.0, 0.0

    # only the first N citations count toward the gate, so a system cannot dump
    # its whole retrieval list to fish for a chance gold hit
    gate = citations[: config.gate_max_citations]

    if task["type"] in ("aggregation", "dossier"):
        verdict = _judge_aggregation(llm, config, task, answer)
        matched = list(verdict.get("matched_items") or [])
        cited_matched = 0
        uncited_matched = 0
        for i, item in enumerate(task["items"]):
            if i < len(matched) and matched[i]:
                uncited_matched += 1
                supported = set(item["doc_ids"]) | set(gold)
                if any(c in supported for c in gate):
                    cited_matched += 1
        extra = max(0, int(verdict.get("extra_items") or 0))
        n_gold = len(task["items"])
        return (
            _item_f1(cited_matched, extra, n_gold),
            _item_f1(uncited_matched, extra, n_gold),
        )

    # single_hop / timeline
    verdict = _judge_answer(llm, config, task, answer)
    if verdict.get("is_refusal") or not verdict.get("correct"):
        return 0.0, 0.0
    cited = 1.0 if any(c in set(gold) for c in gate) else 0.0
    return cited, 1.0
