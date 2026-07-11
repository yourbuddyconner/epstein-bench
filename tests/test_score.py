import json

import pytest

from epstein_bench import llm as llm_mod
from epstein_bench.score import ndcg_at_k, recall_at_k, score_predictions


def test_recall_at_k_hand_computed():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 2) == 0.5
    assert recall_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert recall_at_k([], ["a"], 5) == 0.0


def test_ndcg_at_k_hand_computed():
    # single gold doc at rank 1 -> perfect
    assert ndcg_at_k(["g"], ["g"], 10) == 1.0
    # gold at rank 2 of 2 with one gold: dcg = 1/log2(3), idcg = 1
    import math

    assert ndcg_at_k(["x", "g"], ["g"], 10) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["x", "y"], ["g"], 10) == 0.0


def _single_hop_task(task_id="t1"):
    return {
        "task_id": task_id,
        "type": "single_hop",
        "question": "On what date did Alice Example email Bob Sample?",
        "answer": "January 10, 2015",
        "items": None,
        "gold_docs": ["d1", "d5"],
        "source_doc_ids": ["d1"],
        "provenance": {},
    }


def _pred(task_id="t1", **over):
    base = {
        "task_id": task_id,
        "answer": "January 10, 2015",
        "citations": ["d1"],
        "retrieved": ["d1", "d5", "d9"],
    }
    base.update(over)
    return base


def test_cited_correctness_requires_supporting_citation(config, llm):
    tasks = [_single_hop_task()]
    # correct answer, citation NOT in gold set -> cited 0, uncited 1
    report = score_predictions(config, llm, tasks, [_pred(citations=["d9"])])
    assert report["per_type"]["single_hop"] == 0.0
    assert report["per_type_uncited"]["single_hop"] == 1.0
    assert report["overall_uncited_correctness"] == 1.0
    # correct answer with supporting citation -> 1
    report = score_predictions(config, llm, tasks, [_pred()])
    assert report["per_type"]["single_hop"] == 1.0
    assert report["overall_cited_correctness"] == 1.0


def test_uncited_measures_parametric_knowledge(config, llm):
    """A citation-free correct answer (closed-book/parametric system) scores
    0 on the headline but full credit on uncited correctness."""
    report = score_predictions(
        config, llm, [_single_hop_task()], [_pred(citations=[], retrieved=[])]
    )
    assert report["per_type"]["single_hop"] == 0.0
    assert report["per_type_uncited"]["single_hop"] == 1.0


def test_wrong_answer_scores_zero(config, llm):
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = json.dumps(
        {"correct": False, "is_refusal": False}
    )
    report = score_predictions(config, llm, [_single_hop_task()], [_pred()])
    assert report["per_type"]["single_hop"] == 0.0


def test_unanswerable_scores_refusal(config, llm):
    task = {
        "task_id": "u1",
        "type": "unanswerable",
        "question": "What was Alice Example's role at Acme Corporation in 1999?",
        "answer": None,
        "items": None,
        "gold_docs": [],
        "source_doc_ids": [],
        "provenance": {},
    }
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = json.dumps(
        {"correct": False, "is_refusal": True}
    )
    report = score_predictions(
        config, llm, [task], [_pred("u1", answer="Not determinable from the corpus.")]
    )
    assert report["per_type"]["unanswerable"] == 1.0
    # confident hallucination -> 0
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = json.dumps(
        {"correct": False, "is_refusal": False}
    )
    report = score_predictions(
        config, llm, [task], [_pred("u1", answer="She was the CEO.")]
    )
    assert report["per_type"]["unanswerable"] == 0.0


def test_aggregation_item_level_f1(config, llm):
    task = {
        "task_id": "a1",
        "type": "aggregation",
        "question": "Which people are named in correspondence with Alice Example?",
        "answer": None,
        "items": [
            {"item": "Bob Sample", "doc_ids": ["d1"]},
            {"item": "Carol Case", "doc_ids": ["d2"]},
        ],
        "gold_docs": ["d1", "d2"],
        "source_doc_ids": ["d1", "d2"],
        "provenance": {},
    }
    # matched both, one extra item, citations cover both -> P=2/3, R=1, F1=0.8
    llm_mod.STUB_OVERRIDES["AGGJUDGE"] = json.dumps(
        {"matched_items": [True, True], "extra_items": 1}
    )
    report = score_predictions(
        config,
        llm,
        [task],
        [_pred("a1", answer="Bob Sample, Carol Case, Dan Extra", citations=["d1", "d2"])],
    )
    assert report["per_type"]["aggregation"] == pytest.approx(0.8)
    # matched items without supporting citations don't count
    report = score_predictions(
        config,
        llm,
        [task],
        [_pred("a1", answer="Bob Sample, Carol Case, Dan Extra", citations=["d9"])],
    )
    assert report["per_type"]["aggregation"] == 0.0


def test_gate_ignores_citations_past_the_cap(config, llm):
    """A supporting citation beyond gate_max_citations must not earn credit;
    dumping the whole retrieval list into `citations` cannot fish for a gold hit."""
    config.gate_max_citations = 3
    tasks = [_single_hop_task()]  # gold_docs = ["d1", "d5"]
    # gold citation sits at index 3 -> outside the gate -> cited 0, uncited 1
    report = score_predictions(
        config, llm, tasks, [_pred(citations=["x1", "x2", "x3", "d1"])]
    )
    assert report["per_type"]["single_hop"] == 0.0
    assert report["per_type_uncited"]["single_hop"] == 1.0
    # gold citation inside the gate -> full credit
    report = score_predictions(
        config, llm, tasks, [_pred(citations=["d1", "x2", "x3", "x4"])]
    )
    assert report["per_type"]["single_hop"] == 1.0


def test_citation_precision_recall_reported(config, llm):
    tasks = [_single_hop_task()]  # gold_docs = ["d1", "d5"]
    report = score_predictions(config, llm, tasks, [_pred(citations=["d1", "d9"])])
    # one of two citations is gold -> precision 0.5; one of two gold cited -> recall 0.5
    assert report["citation_precision"] == pytest.approx(0.5)
    assert report["citation_recall"] == pytest.approx(0.5)


def test_confidence_intervals_and_micro_present(config, llm):
    tasks = [_single_hop_task("t1"), _single_hop_task("t2")]
    report = score_predictions(
        config, llm, tasks, [_pred("t1"), _pred("t2", citations=["d9"])]
    )
    lo, hi = report["overall_cited_correctness_ci95"]
    assert lo <= report["overall_cited_correctness"] <= hi
    assert 0.0 <= report["overall_cited_correctness_micro"] <= 1.0
    # deterministic across runs (seeded RNG)
    again = score_predictions(
        config, llm, tasks, [_pred("t1"), _pred("t2", citations=["d9"])]
    )
    assert again["overall_cited_correctness_ci95"] == [lo, hi]


def _false_premise_task(task_id="fp1"):
    return {
        "task_id": task_id,
        "type": "false_premise",
        "question": "When Alice Example met Bob Sample in Geneva in 2015, who introduced them?",
        "answer": None,
        "items": None,
        "false_element": "a meeting between Alice Example and Bob Sample in Geneva",
        "gold_docs": [],
        "source_doc_ids": [],
        "provenance": {},
    }


def test_false_premise_rewards_rejection_and_reports_id_rate(config, llm):
    task = _false_premise_task()
    # system rejects the premise -> full credit; stub PREMISEID says it named it
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = json.dumps(
        {"correct": False, "is_refusal": True}
    )
    report = score_predictions(
        config, llm, [task], [_pred("fp1", answer="No such meeting is documented.")]
    )
    assert report["per_type"]["false_premise"] == 1.0
    assert report["premise_refused_n"] == 1
    assert report["premise_id_rate"] == 1.0
    # false_premise carries no gold docs -> excluded from citation diagnostics
    assert report["citation_precision"] == 0.0


def test_false_premise_accepting_the_premise_scores_zero(config, llm):
    task = _false_premise_task()
    # system answers the follow-up (accepts the false premise) -> 0, and there
    # is no refusal so premise_id_rate is not reported
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = json.dumps(
        {"correct": False, "is_refusal": False}
    )
    report = score_predictions(
        config, llm, [task], [_pred("fp1", answer="They were introduced by Carol Case.")]
    )
    assert report["per_type"]["false_premise"] == 0.0
    assert "premise_id_rate" not in report


def test_usage_and_cost_aggregated_into_report(config, llm):
    tasks = [_single_hop_task("t1"), _single_hop_task("t2")]
    preds = [
        _pred("t1", usage={"input_tokens": 1000, "output_tokens": 200}, cost_usd=0.012),
        _pred("t2", usage={"input_tokens": 3000, "output_tokens": 400}, cost_usd=0.015),
    ]
    report = score_predictions(config, llm, tasks, preds)
    assert report["tokens_total"] == 4600
    assert report["tokens_per_task"] == pytest.approx(2300.0)
    assert report["cost_usd_total"] == pytest.approx(0.027)
    assert report["cost_usd_per_task"] == pytest.approx(0.0135)


def test_usage_absent_for_cheap_baselines(config, llm):
    # predictions without usage/cost -> no telemetry keys in the report
    report = score_predictions(config, llm, [_single_hop_task()], [_pred()])
    assert "tokens_total" not in report and "cost_usd_total" not in report


def test_missing_predictions_rejected(config, llm):
    with pytest.raises(ValueError, match="missing"):
        score_predictions(config, llm, [_single_hop_task()], [])


def test_retrieval_metrics_in_report(config, llm):
    report = score_predictions(config, llm, [_single_hop_task()], [_pred()])
    assert report["retrieval"]["recall@5"] == 1.0
    assert report["retrieval"]["recall@20"] == 1.0
    assert 0 < report["retrieval"]["ndcg@10"] <= 1.0
    assert report["judge_model"] == config.judge_model


def test_ndcg_duplicates_cannot_inflate_gain():
    """Repeating a gold doc at every rank must not stack gain (or nDCG > 1)."""
    assert ndcg_at_k(["g"] * 10, ["g"], 10) == 1.0
    assert ndcg_at_k(["g", "g", "x"], ["g", "x"], 10) == ndcg_at_k(
        ["g", "x"], ["g", "x"], 10
    )


def test_duplicate_task_ids_rejected(config, llm):
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions(
            config, llm, [_single_hop_task()], [_pred(), _pred(answer="other")]
        )


def test_unknown_task_ids_rejected(config, llm):
    """Padding predictions with extra task_ids (telemetry dilution) is an error."""
    with pytest.raises(ValueError, match="unknown"):
        score_predictions(
            config, llm, [_single_hop_task()], [_pred(), _pred("bogus-id")]
        )


def test_aggregation_gate_is_per_item(config, llm):
    """Citing one item's doc must not earn cited credit for OTHER items."""
    task = {
        "task_id": "a2",
        "type": "aggregation",
        "question": "Which people are named in correspondence with Alice Example?",
        "answer": None,
        "items": [
            {"item": "Bob Sample", "doc_ids": ["d1"]},
            {"item": "Carol Case", "doc_ids": ["d2"]},
        ],
        "gold_docs": ["d1", "d2"],
        "source_doc_ids": ["d1", "d2"],
        "provenance": {},
    }
    llm_mod.STUB_OVERRIDES["AGGJUDGE"] = json.dumps(
        {"matched_items": [True, True], "extra_items": 0}
    )
    report = score_predictions(
        config, llm, [task], [_pred("a2", answer="Bob Sample, Carol Case", citations=["d1"])]
    )
    # only Bob's item is cited-supported; Carol's is an unsupported claim that
    # costs precision: P = 1/2, R = 1/2 -> F1 = 1/2
    assert report["per_type"]["aggregation"] == pytest.approx(0.5)
    assert report["per_type_uncited"]["aggregation"] == 1.0


def test_judge_failure_aborts_instead_of_scoring_zero(config, llm):
    """A systemically broken judge must raise, not score every task as 0."""
    from epstein_bench.llm import LLMError

    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = "this is not json"
    tasks = [_single_hop_task("t1"), _single_hop_task("t2")]
    with pytest.raises(LLMError, match="judge failed"):
        score_predictions(config, llm, tasks, [_pred("t1"), _pred("t2")])
