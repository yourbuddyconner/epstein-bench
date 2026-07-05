import json

from conftest import fixture_rows

from epstein_bench import llm as llm_mod
from epstein_bench.corpus import build_corpus
from epstein_bench.generate import generate_candidates
from epstein_bench.io_utils import read_jsonl
from epstein_bench.pool import pool_tasks
from epstein_bench.verify import verify_candidates

IRRELEVANT = json.dumps({"verdicts": ["irrelevant"] * 16})
SUPPORTS = json.dumps({"verdicts": ["supports"] * 16})


def _run_through_verify(config, llm):
    build_corpus(config, llm, rows=fixture_rows())
    generate_candidates(config, llm)
    verify_candidates(config, llm)


def test_pool_rescues_source_docs_via_strong_model(config, llm):
    """Cheap judge misses the source doc -> strong-model rescue keeps the task."""
    _run_through_verify(config, llm)
    llm_mod.STUB_OVERRIDES["POOLJUDGE"] = IRRELEVANT
    llm_mod.STUB_OVERRIDES["POOLRESCUE"] = SUPPORTS
    stats = pool_tasks(config, llm)
    assert stats["pooled"] > 0
    # rescued tasks keep their source docs as gold
    for task in read_jsonl(config.build_dir / "pooled.jsonl"):
        if task["type"] != "unanswerable":
            assert set(task["source_doc_ids"]) <= set(task["gold_docs"])


def test_pool_drops_when_rescue_also_fails(config, llm):
    _run_through_verify(config, llm)
    llm_mod.STUB_OVERRIDES["POOLJUDGE"] = IRRELEVANT
    llm_mod.STUB_OVERRIDES["POOLRESCUE"] = IRRELEVANT
    stats = pool_tasks(config, llm)
    dropped = list(read_jsonl(config.build_dir / "pool_dropped.jsonl"))
    assert dropped and all(d["reason"] == "source_not_supportive" for d in dropped)
    # only unanswerable tasks (no source docs to certify) survive
    kept = list(read_jsonl(config.build_dir / "pooled.jsonl"))
    assert all(t["type"] == "unanswerable" for t in kept)
    assert stats["pooled"] == len(kept)


def test_pool_happy_path_needs_no_rescue(config, llm):
    _run_through_verify(config, llm)
    calls = {"rescue": 0}
    llm_mod.STUB_OVERRIDES["POOLRESCUE"] = lambda prompt: (
        calls.__setitem__("rescue", calls["rescue"] + 1) or SUPPORTS
    )
    stats = pool_tasks(config, llm)
    assert stats["pooled"] > 0
    assert calls["rescue"] == 0  # supports-by-default cheap judge suffices
