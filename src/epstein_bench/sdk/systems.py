"""Reference `System` implementations: retrieval and no-retrieval baselines.

These are the bm25 / dense / hybrid / closed_book / parametric baselines,
expressed on the SDK's `System` interface so they dogfood the same contract an
external system uses. The retrieval prompt and the no-context prompts are pinned
here (they use the same ``[BASELINE]`` / ``[PARAMETRIC]`` stub tags as before).
"""

from __future__ import annotations

from ..llm import LLM
from . import REFUSAL, Prediction, Retriever, System, Task

CONTEXT_DOCS = 5  # docs shown to the generator
TOP_K = 20  # retrieved list length reported (contract max)


def _answer_with_context(
    llm: LLM, question: str, context_docs: list[dict]
) -> tuple[str, list[str]]:
    listing = "\n\n".join(
        f"[DOC id={d['doc_id']}]\n{d['text'][:2000]}" for d in context_docs
    )
    resp = llm.chat_json(
        "[BASELINE] Answer the question using ONLY the documents provided, and "
        "cite the ids of the documents that support your answer. If the documents "
        'do not contain the answer, respond {"answer": null, "citations": []}. '
        'Respond with JSON {"answer": str|null, "citations": [str]}.'
        f"\n\nQuestion: {question}\n\n{listing}"
    )
    answer = resp.get("answer")
    citations = [str(c) for c in (resp.get("citations") or [])]
    if not answer:
        return REFUSAL, []
    return str(answer), citations


def _answer_closed_book(llm: LLM, question: str) -> str:
    resp = llm.chat_json(
        "[BASELINE] Answer this question from your own knowledge. If you do not "
        'know, respond {"answer": null, "citations": []}. Respond with JSON '
        '{"answer": str|null, "citations": []}.'
        f"\n\nQuestion: {question}"
    )
    return str(resp.get("answer")) if resp.get("answer") else REFUSAL


def _answer_parametric(llm: LLM, question: str) -> str:
    """Best-effort recall from training data (parametric-knowledge probe)."""
    resp = llm.chat_json(
        "[PARAMETRIC] This question is about the publicly released Epstein "
        "files, which you may have seen during training. Answer from your own "
        "stored knowledge — recall as specifically as you can, and give your "
        "best answer even if you are not fully certain. Only respond "
        '{"answer": null} if you have no relevant knowledge at all. '
        'Respond with JSON {"answer": str|null, "citations": []}.'
        f"\n\nQuestion: {question}"
    )
    return str(resp.get("answer")) if resp.get("answer") else REFUSAL


class RetrievalSystem(System):
    """Retrieve top-k, show the top few docs to the generator, answer + cite."""

    def __init__(
        self,
        llm: LLM,
        retriever: Retriever,
        docs_by_id: dict[str, dict],
        *,
        context_docs: int = CONTEXT_DOCS,
        top_k: int = TOP_K,
    ):
        self.llm = llm
        self.retriever = retriever
        self.docs_by_id = docs_by_id
        self.context_docs = context_docs
        self.top_k = top_k

    def predict(self, task: Task) -> Prediction:
        ranked = self.retriever.search(task.question, self.top_k)
        retrieved = [doc_id for doc_id, _ in ranked]
        context = [
            self.docs_by_id[d]
            for d in retrieved[: self.context_docs]
            if d in self.docs_by_id
        ]
        answer, citations = _answer_with_context(self.llm, task.question, context)
        return Prediction(answer=answer, citations=citations, retrieved=retrieved)


class NoContextSystem(System):
    """Closed-book (retrieval-necessity control) or parametric (contamination
    probe) — no documents, no citations, no retrieval list."""

    def __init__(self, llm: LLM, mode: str):
        assert mode in ("closed_book", "parametric")
        self.llm = llm
        self.mode = mode

    def predict(self, task: Task) -> Prediction:
        if self.mode == "parametric":
            answer = _answer_parametric(self.llm, task.question)
        else:
            answer = _answer_closed_book(self.llm, task.question)
        return Prediction(answer=answer, citations=[], retrieved=[])
