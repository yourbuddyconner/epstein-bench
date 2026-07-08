import json

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


def test_run_fails_closed_to_refusal(tmp_path):
    q = _write_questions(tmp_path, [{"task_id": "t1", "question": "Q?"}])
    out = tmp_path / "preds.jsonl"
    run(_BoomSystem(), q, out, workers=1)
    pred = json.loads(out.read_text().splitlines()[0])
    assert pred["answer"] == REFUSAL
    assert pred["citations"] == [] and pred["retrieved"] == []


# -- AgenticRAG loop (fake Anthropic client) ------------------------------------


class _Block:
    def __init__(self, type_, name=None, input=None, id=None):
        self.type = type_
        self.name = name
        self.input = input
        self.id = id


class _Resp:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason


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
    agent = AgenticRAG(_FakeClient(responses), "test-model", retr, DOCS)
    pred = agent.predict(Task(task_id="t1", question="Did Alice email Bob?"))
    assert pred.answer == "They emailed."
    assert pred.citations == ["d1"]
    assert pred.retrieved == ["d1", "d2"]  # accumulated from the search
    assert retr.queries == ["alice bob"]


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
