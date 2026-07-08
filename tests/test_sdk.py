import json

import pytest

from epstein_bench.sdk import REFUSAL, Prediction, System, Task, run
from epstein_bench.sdk.agentic import AgenticRAG


# -- run() harness --------------------------------------------------------------


class _EchoSystem(System):
    def predict(self, task: Task) -> Prediction:
        return Prediction(
            answer=f"answer to {task.task_id}",
            citations=[f"c{i}" for i in range(30)],  # over the cap on purpose
            retrieved=[f"r{i}" for i in range(30)],
        )


class _BoomSystem(System):
    def predict(self, task: Task) -> Prediction:
        raise RuntimeError("kaboom")


class _FlakySystem(System):
    """Fails on exactly one task_id, succeeds on the rest."""

    def __init__(self, fail_id):
        self.fail_id = fail_id

    def predict(self, task: Task) -> Prediction:
        if task.task_id == self.fail_id:
            raise RuntimeError("blip")
        return Prediction(answer=f"ok {task.task_id}", citations=["d1"], retrieved=["d1"])


def _write_questions(tmp_path, rows):
    p = tmp_path / "questions.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_run_writes_conformant_capped_predictions(tmp_path):
    q = _write_questions(
        tmp_path, [{"task_id": "t1", "type": "single_hop", "question": "Q1?"}]
    )
    out = tmp_path / "preds.jsonl"
    n = run(_EchoSystem(), q, out, max_retrieved=20, workers=1)
    assert n == 1
    pred = json.loads(out.read_text().splitlines()[0])
    assert pred["task_id"] == "t1"
    assert pred["answer"] == "answer to t1"
    # citations and retrieved capped to max_retrieved
    assert len(pred["citations"]) == 20
    assert len(pred["retrieved"]) == 20


def test_run_aborts_on_systemic_failure(tmp_path):
    # every task errors (e.g. exhausted credits) -> abort loudly, write nothing
    q = _write_questions(
        tmp_path, [{"task_id": f"t{i}", "question": "Q?"} for i in range(6)]
    )
    out = tmp_path / "preds.jsonl"
    with pytest.raises(RuntimeError, match="aborting run"):
        run(_BoomSystem(), q, out, workers=1)
    assert not out.exists()  # degraded file must not be written


def test_run_tolerates_occasional_failure(tmp_path):
    # one flaky task among ten -> recorded as a refusal, the run still completes
    rows = [{"task_id": f"t{i}", "question": "Q?"} for i in range(10)]
    q = _write_questions(tmp_path, rows)
    out = tmp_path / "preds.jsonl"
    n = run(_FlakySystem(fail_id="t7"), q, out, workers=1)
    assert n == 10
    preds = {json.loads(l)["task_id"]: json.loads(l) for l in out.read_text().splitlines()}
    assert preds["t7"]["answer"] == REFUSAL and preds["t7"]["citations"] == []
    assert preds["t3"]["answer"] == "ok t3"


# -- AgenticRAG loop (fake Anthropic client) ------------------------------------


class _Block:
    def __init__(self, type_, name=None, input=None, id=None):
        self.type = type_
        self.name = name
        self.input = input
        self.id = id


class _Usage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_read_input_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _Resp:
    def __init__(self, content, stop_reason="tool_use", usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or _Usage(input_tokens=1000, output_tokens=200)


class _FakeClient:
    """Replays a scripted list of responses; records nothing about messages."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

        class _Messages:
            def create(inner, **kwargs):  # noqa: N805
                self.calls += 1
                return self._responses.pop(0)

        self.messages = _Messages()


class _StubRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k):
        self.queries.append(query)
        return [("d1", 1.0), ("d2", 0.5)]


DOCS = {"d1": {"doc_id": "d1", "text": "Alice emailed Bob."}, "d2": {"doc_id": "d2", "text": "More."}}


def test_agentic_searches_then_submits():
    responses = [
        _Resp([_Block("tool_use", name="search", input={"query": "alice bob"}, id="s1")]),
        _Resp([_Block("tool_use", name="submit_answer",
                      input={"answer": "They emailed.", "citations": ["d1"]}, id="s2")]),
    ]
    retr = _StubRetriever()
    agent = AgenticRAG(_FakeClient(responses), "claude-sonnet-5", retr, DOCS)
    pred = agent.predict(Task(task_id="t1", question="Did Alice email Bob?"))
    assert pred.answer == "They emailed."
    assert pred.citations == ["d1"]
    assert pred.retrieved == ["d1", "d2"]  # accumulated from the search
    assert retr.queries == ["alice bob"]
    # usage summed across both API calls; cost priced from the model
    assert pred.usage["input_tokens"] == 2000 and pred.usage["output_tokens"] == 400
    assert pred.usage["requests"] == 2
    # 2000/1e6*3 + 400/1e6*15 = 0.006 + 0.006 = 0.012
    assert pred.cost_usd == pytest.approx(0.012)


def test_agentic_refuses_when_never_submitting():
    # model keeps emitting plain text, never calls submit_answer
    responses = [_Resp([_Block("text")], stop_reason="end_turn") for _ in range(20)]
    agent = AgenticRAG(_FakeClient(responses), "test-model", _StubRetriever(), DOCS)
    pred = agent.predict(Task(task_id="t1", question="Q?"))
    assert pred.answer == REFUSAL
    assert pred.citations == []


def test_agentic_enforces_search_budget():
    # three search requests but max_searches=1 -> retriever hit at most once,
    # then the model is told to submit and does
    responses = [
        _Resp([_Block("tool_use", name="search", input={"query": "a"}, id="s1")]),
        _Resp([_Block("tool_use", name="search", input={"query": "b"}, id="s2")]),
        _Resp([_Block("tool_use", name="submit_answer",
                      input={"answer": "x", "citations": []}, id="s3")]),
    ]
    retr = _StubRetriever()
    agent = AgenticRAG(_FakeClient(responses), "test-model", retr, DOCS, max_searches=1)
    pred = agent.predict(Task(task_id="t1", question="Q?"))
    assert pred.answer == "x"
    assert len(retr.queries) == 1  # second search was refused by the budget
