"""Scan/select + dossier task family (v1.1)."""

import json
import re

from conftest import fixture_rows

from epstein_bench import llm as llm_mod
from epstein_bench.corpus import load_docs
from epstein_bench.generate import gen_dossier, gen_single_hop, generate_candidates
from epstein_bench.io_utils import write_jsonl
from epstein_bench.scan import SCAN_DIR, load_targets, select_corpus
from epstein_bench.score import score_predictions
from epstein_bench.verify import Gauntlet


def _write_scan_parts(config):
    """Simulate a completed scan from the fixture rows."""
    from epstein_bench.corpus import extract_names

    part = config.build_dir / SCAN_DIR / "part_00000.jsonl"
    rows = []
    for r in fixture_rows():
        text = (r.get("text_content") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "doc_id": r["doc_id"],
                "text": text,
                "names": sorted(set(extract_names(text))),
                "meta": {k: r[k] for k in ("file_name", "file_type") if r.get(k)},
            }
        )
    write_jsonl(part, rows)


def _dossier_stub_with_real_ids(prompt: str) -> str:
    doc_ids = re.findall(r"\[DOC \d+\] id=(\S+)", prompt)
    items = [
        {"item": f"2015-01-1{i} — emailed Bob Sample (doc {d})", "doc_ids": [d]}
        for i, d in enumerate(doc_ids[:3])
    ]
    return json.dumps(
        {"question": "What is the documented timeline of Alice Example's contact with Bob Sample?", "items": items}
    )


def test_select_builds_entity_complete_corpus(config, llm):
    _write_scan_parts(config)
    stats = select_corpus(config, llm)
    assert stats["target_entities"] >= 1
    assert stats["clean"] == 8
    targets = load_targets(config)
    assert "alice example" in targets
    # every doc mentioning the target is in the corpus (entity-complete)
    corpus_ids = {d["doc_id"] for d in load_docs(config)}
    assert set(targets["alice example"]["doc_ids"]) <= corpus_ids


def test_gen_dossier_produces_multi_doc_item_tasks(config, llm):
    _write_scan_parts(config)
    select_corpus(config, llm)
    llm_mod.STUB_OVERRIDES["DOSSIER"] = _dossier_stub_with_real_ids
    tasks = gen_dossier(config, llm, load_docs(config), load_targets(config), n=3)
    assert tasks
    t = tasks[0]
    assert t["type"] == "dossier"
    assert len(t["items"]) >= 3
    assert len(t["source_doc_ids"]) >= 2
    assert t["provenance"]["target_entity"] in ("alice example", "bob sample")


def test_dossier_passes_gauntlet_and_scores_like_aggregation(config, llm):
    _write_scan_parts(config)
    select_corpus(config, llm)
    llm_mod.STUB_OVERRIDES["DOSSIER"] = _dossier_stub_with_real_ids
    docs = load_docs(config)
    tasks = gen_dossier(config, llm, docs, load_targets(config), n=1)
    task = tasks[0]
    # answerability must recover the items from the gold docs
    llm_mod.STUB_OVERRIDES["ANSWER"] = json.dumps(
        {"answer": "; ".join(i["item"] for i in task["items"]), "found": True}
    )
    passed, reason = Gauntlet(config, llm, docs).run(task)
    assert passed, reason
    # scoring: perfect answer with per-item citations -> F1 1.0
    task["gold_docs"] = task["source_doc_ids"]
    llm_mod.STUB_OVERRIDES["AGGJUDGE"] = json.dumps(
        {"matched_items": [True] * len(task["items"]), "extra_items": 0}
    )
    report = score_predictions(
        config,
        llm,
        [task],
        [
            {
                "task_id": task["task_id"],
                "answer": "; ".join(i["item"] for i in task["items"]),
                "citations": task["gold_docs"],
                "retrieved": task["gold_docs"],
            }
        ],
    )
    assert report["per_type"]["dossier"] == 1.0


def test_low_salience_facts_are_dropped(config, llm):
    _write_scan_parts(config)
    select_corpus(config, llm)
    llm_mod.STUB_OVERRIDES["FACTS"] = json.dumps(
        {
            "facts": [
                {
                    "fact": "boring",
                    "question": "What is the Amex ticket number for Alice Example?",
                    "answer": "12345",
                    "salience": 1,
                }
            ]
        }
    )
    assert gen_single_hop(config, llm, load_docs(config), n=5) == []


def test_generate_skips_dossier_without_targets(config, llm):
    """Direct-corpus mode (no scan/select) must still work end to end."""
    from epstein_bench.corpus import build_corpus

    build_corpus(config, llm, rows=fixture_rows())
    stats = generate_candidates(config, llm)
    assert stats.get("dossier", 0) == 0
    assert stats["total"] > 0
