import json

from epstein_bench import llm as llm_mod
from epstein_bench.verify import Gauntlet, token_f1

DOCS = [
    {"doc_id": "d1", "quality": "clean", "text": "Alice emailed Bob on January 10, 2015."},
    {"doc_id": "d2", "quality": "clean", "text": "Bob replied in March 2015."},
    {"doc_id": "d3", "quality": "clean", "text": "Unrelated committee schedule."},
    {"doc_id": "d4", "quality": "clean", "text": "More unrelated material."},
]


def _task(**over):
    base = {
        "task_id": "t1",
        "type": "single_hop",
        "question": "On what date did Alice Example email Bob Sample?",
        "answer": "January 10, 2015",
        "items": None,
        "source_doc_ids": ["d1"],
        "provenance": {},
    }
    base.update(over)
    return base


def test_token_f1():
    assert token_f1("January 10, 2015", "january 10 2015") == 1.0
    assert token_f1("totally different", "january 10 2015") == 0.0
    assert 0 < token_f1("January 2015", "January 10, 2015") < 1


def test_gauntlet_passes_good_task(config, llm):
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert passed, reason


def test_rejects_non_standalone(config, llm):
    llm_mod.STUB_OVERRIDES["STANDALONE"] = json.dumps(
        {"standalone": False, "reason": "bare initials"}
    )
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert not passed and reason == "standalone"


def test_rejects_unanswerable_from_gold_docs(config, llm):
    llm_mod.STUB_OVERRIDES["ANSWER"] = json.dumps({"answer": None, "found": False})
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert not passed and reason == "answerability"


def test_rejects_wrong_reference_answer(config, llm):
    # fresh attempt disagrees with the reference -> answerability fails
    llm_mod.STUB_OVERRIDES["MATCH"] = json.dumps({"match": False})
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert not passed and reason == "answerability"


def test_rejects_answerable_closed_book(config, llm):
    llm_mod.STUB_OVERRIDES["CLOSEDBOOK"] = json.dumps(
        {"answer": "January 10, 2015", "found": True}
    )
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert not passed and reason == "necessity"


def test_rejects_single_doc_sufficient_timeline(config, llm):
    llm_mod.STUB_OVERRIDES["SINGLEDOC"] = json.dumps(
        {"answer": "January to March 2015", "found": True}
    )
    task = _task(
        type="timeline",
        question="Over what period did Alice Example correspond with Bob Sample?",
        answer="January 2015 to March 2015",
        source_doc_ids=["d1", "d2"],
    )
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(task)
    assert not passed and reason == "necessity"


def test_rejects_on_adjudication(config, llm):
    llm_mod.STUB_OVERRIDES["ADJUDICATE"] = json.dumps(
        {"pass": False, "category": "ambiguous"}
    )
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert not passed and reason == "adjudication:ambiguous"


def test_llm_error_fails_closed(config, llm):
    llm_mod.STUB_OVERRIDES["STANDALONE"] = "not json at all"
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(_task())
    assert not passed and reason.startswith("error:")


def test_unanswerable_skips_answerability(config, llm):
    task = _task(
        type="unanswerable",
        question="What was Alice Example's role at Acme Corporation in 1999?",
        answer=None,
        source_doc_ids=[],
    )
    # would fail answerability if it ran; unanswerable tasks must skip it
    llm_mod.STUB_OVERRIDES["ANSWER"] = json.dumps({"answer": None, "found": False})
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(task)
    assert passed, reason


def test_false_premise_skips_answerability_and_necessity(config, llm):
    task = _task(
        type="false_premise",
        question="When Alice Example met Bob Sample in Geneva in 2015, who introduced them?",
        answer=None,
        source_doc_ids=[],
        false_element="a meeting between Alice Example and Bob Sample in Geneva",
    )
    # stages 2-3 would fail with no gold docs; false_premise must skip them and
    # be gated only by standalone + adjudication
    llm_mod.STUB_OVERRIDES["ANSWER"] = json.dumps({"answer": None, "found": False})
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(task)
    assert passed, reason


def test_false_premise_dropped_when_corpus_supports_premise(config, llm):
    """A premise the documents actually support (only a detail perturbed) must
    be dropped, else scoring would punish a correct system for not refusing."""
    task = _task(
        type="false_premise",
        question="When Alice Example met Bob Sample in Geneva in 2015, who introduced them?",
        answer=None,
        source_doc_ids=[],
        false_element="a meeting between Alice Example and Bob Sample in Geneva",
    )
    llm_mod.STUB_OVERRIDES["FPSUPPORT"] = json.dumps({"verdict": "supports"})
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(task)
    assert not passed and reason == "adjudication:supported"


def test_false_premise_dropped_when_wording_tips_off(config, llm):
    task = _task(
        type="false_premise",
        question="Who supposedly introduced Alice Example to Bob Sample at the fictional Geneva meeting?",
        answer=None,
        source_doc_ids=[],
        false_element="a meeting between Alice Example and Bob Sample in Geneva",
    )
    llm_mod.STUB_OVERRIDES["FPQUALITY"] = json.dumps(
        {"plausible": True, "tips_off": True}
    )
    g = Gauntlet(config, llm, DOCS)
    passed, reason = g.run(task)
    assert not passed and reason == "adjudication:tips_off"
