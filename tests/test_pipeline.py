"""End-to-end smoke test: the full pipeline on fixture docs with the stub LLM,
then a submission bundle round-trip. Runs in CI with no API key.
"""

import json

from conftest import fixture_rows

from epstein_bench import DATASET_VERSION
from epstein_bench import llm as llm_mod
from epstein_bench.corpus import build_corpus
from epstein_bench.finalize import finalize_dataset
from epstein_bench.generate import generate_candidates
from epstein_bench.io_utils import read_jsonl, write_jsonl
from epstein_bench.pool import pool_tasks
from epstein_bench.score import score_predictions
from epstein_bench.submit import build_bundle, validate_bundle
from epstein_bench.verify import verify_candidates

REFUSAL = "The corpus does not contain enough information to answer this question."


def _perfect_predictions(tasks):
    preds = []
    for t in tasks:
        if t["type"] == "unanswerable":
            answer = REFUSAL
        elif t["type"] == "aggregation":
            answer = ", ".join(i["item"] for i in t["items"])
        else:
            answer = t["answer"]
        preds.append(
            {
                "task_id": t["task_id"],
                "answer": answer,
                "citations": t["gold_docs"][:3],
                "retrieved": t["gold_docs"],
            }
        )
    return preds


def test_full_pipeline_smoke(config, llm):
    # refusal-aware scoring judge for the synthetic predictions
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = lambda prompt: json.dumps(
        {"correct": True, "is_refusal": REFUSAL in prompt}
    )

    corpus_stats = build_corpus(config, llm, rows=fixture_rows())
    assert corpus_stats["clean"] == 8

    gen_stats = generate_candidates(config, llm)
    assert gen_stats["total"] > 0
    assert gen_stats.get("single_hop", 0) > 0
    assert gen_stats.get("unanswerable", 0) > 0

    verify_stats = verify_candidates(config, llm)
    assert verify_stats["verified"] > 0

    pool_stats = pool_tasks(config, llm)
    assert pool_stats["pooled"] > 0

    manifest = finalize_dataset(config)
    assert manifest["version"] == DATASET_VERSION
    assert manifest["splits"]["full"]["n_tasks"] > 0
    assert manifest["splits"]["dev"]["n_tasks"] > 0

    # every finalized answerable task has pooled gold docs and provenance
    tasks = list(
        read_jsonl(config.dataset_dir / DATASET_VERSION / "full" / "tasks.jsonl")
    )
    for t in tasks:
        assert t["provenance"]["verified"] is True
        if t["type"] != "unanswerable":
            assert t["gold_docs"], f"task {t['task_id']} has no gold docs"

    # questions.jsonl never leaks answers or gold docs
    questions = list(
        read_jsonl(config.dataset_dir / DATASET_VERSION / "full" / "questions.jsonl")
    )
    assert set(questions[0].keys()) == {"task_id", "type", "question"}

    # a system that answers perfectly with supported citations scores well
    report = score_predictions(config, llm, tasks, _perfect_predictions(tasks))
    assert report["overall_cited_correctness"] > 0.5
    assert report["retrieval"]["recall@20"] == 1.0

    # submission round trip: build a bundle, validate + rescore it in "CI"
    preds_path = config.build_dir / "preds.jsonl"
    write_jsonl(preds_path, _perfect_predictions(tasks))
    bundle = build_bundle(
        config, preds_path, "Smoke System", config.build_dir / "submissions", "full"
    )
    ci_report = validate_bundle(config, llm, bundle)
    assert ci_report["system_name"] == "Smoke System"
    assert ci_report["overall_cited_correctness"] > 0.5
    assert (bundle / "scores.json").exists()


def test_validate_rejects_tampered_questions_hash(config, llm):
    llm_mod.STUB_OVERRIDES["SCOREJUDGE"] = json.dumps(
        {"correct": True, "is_refusal": False}
    )
    build_corpus(config, llm, rows=fixture_rows())
    generate_candidates(config, llm)
    verify_candidates(config, llm)
    pool_tasks(config, llm)
    finalize_dataset(config)
    tasks = list(
        read_jsonl(config.dataset_dir / DATASET_VERSION / "full" / "tasks.jsonl")
    )
    preds_path = config.build_dir / "preds.jsonl"
    write_jsonl(preds_path, _perfect_predictions(tasks))
    bundle = build_bundle(
        config, preds_path, "Tamper", config.build_dir / "submissions", "full"
    )
    meta = json.loads((bundle / "metadata.json").read_text())
    meta["questions_sha256"] = "0" * 64
    (bundle / "metadata.json").write_text(json.dumps(meta))
    import pytest

    with pytest.raises(ValueError, match="sha256"):
        validate_bundle(config, llm, bundle)
