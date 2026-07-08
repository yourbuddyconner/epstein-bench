"""`AgenticRAG`: an LLM tool-use agent as a reference `System`.

A bounded ReAct-style loop on the Anthropic Messages API. The model is given two
tools — ``search`` (backed by any SDK ``Retriever``) and ``submit_answer`` — and
must gather evidence, then commit. It is NOT told the task type, and is
instructed to reject a question whose premise the documents do not establish
(the behavior the ``false_premise`` task family probes).

The Anthropic client is injected, so the loop is exercised in tests with a fake
client and no API key. Construct the real one with ``anthropic.Anthropic()`` (it
reads ``ANTHROPIC_API_KEY``); pin ``model`` to e.g. ``claude-sonnet-5`` or
``claude-opus-4-8``.
"""

from __future__ import annotations

from ..llm import LLMError
from . import REFUSAL, Prediction, Retriever, System, Task

_SEARCH_TOOL = {
    "name": "search",
    "description": (
        "Search the document corpus. Returns the most relevant documents "
        "(id + text excerpt). Call it several times with refined queries to "
        "gather evidence before answering."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "k": {"type": "integer", "description": "how many documents (default 5)"},
        },
        "required": ["query"],
    },
}

_SUBMIT_TOOL = {
    "name": "submit_answer",
    "description": (
        "Submit your final answer once you have gathered enough evidence. Cite "
        "the doc ids that support it. If the documents do not support the "
        "question — including when it presupposes a meeting, relationship, or "
        "fact the documents do not establish — reject it: give an answer that "
        "explicitly declines, with empty citations."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "doc ids that support the answer (empty if declining)",
            },
        },
        "required": ["answer", "citations"],
    },
}

_PREAMBLE = (
    "You are answering a question about a large corpus of noisy, OCR'd documents "
    "(emails, depositions, calendars, financial records). Use the `search` tool "
    "to find supporting documents, then call `submit_answer`. Answer ONLY what "
    "the documents actually state and cite the doc ids that support each claim. "
    "If you cannot find support — including when the question assumes a meeting, "
    "relationship, or fact the documents do not establish — do not answer it: "
    "submit an answer that explicitly says the corpus does not support it, with "
    "empty citations. Do not use outside knowledge.\n\nQuestion: "
)


class AgenticRAG(System):
    def __init__(
        self,
        client,
        model: str,
        retriever: Retriever,
        docs_by_id: dict[str, dict],
        *,
        max_searches: int = 4,
        context_k: int = 5,
        max_tokens: int = 4096,
    ):
        self.client = client
        self.model = model
        self.retriever = retriever
        self.docs_by_id = docs_by_id
        self.max_searches = max_searches
        self.context_k = context_k
        self.max_tokens = max_tokens

    def _run_search(self, query: str, k: int, retrieved: list[str]) -> str:
        ranked = self.retriever.search(query, k)
        for doc_id, _ in ranked:
            if doc_id not in retrieved:
                retrieved.append(doc_id)
        docs = []
        for doc_id, _ in ranked:
            doc = self.docs_by_id.get(doc_id)
            if doc:
                docs.append(f"[DOC id={doc_id}]\n{doc['text'][:1500]}")
        return "\n\n".join(docs) if docs else "No documents matched."

    def predict(self, task: Task) -> Prediction:
        messages: list[dict] = [
            {"role": "user", "content": _PREAMBLE + task.question}
        ]
        retrieved: list[str] = []
        searches = 0
        # a few turns beyond the search budget for reasoning + the final submit
        for _ in range(self.max_searches + 3):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    tools=[_SEARCH_TOOL, _SUBMIT_TOOL],
                    messages=messages,
                )
            except LLMError:
                break
            except Exception:  # noqa: BLE001 — treat any client error as unscorable
                break
            messages.append({"role": "assistant", "content": resp.content})
            tool_uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not tool_uses:
                if getattr(resp, "stop_reason", None) == "end_turn":
                    # answered in prose without submitting — nudge once, else stop
                    messages.append(
                        {
                            "role": "user",
                            "content": "Call submit_answer with your final answer and citations.",
                        }
                    )
                continue
            results: list[dict] = []
            for tu in tool_uses:
                if tu.name == "submit_answer":
                    answer = str((tu.input or {}).get("answer") or "").strip()
                    citations = [str(c) for c in ((tu.input or {}).get("citations") or [])]
                    return Prediction(
                        answer=answer or REFUSAL,
                        citations=citations,
                        retrieved=retrieved[:20],
                    )
                if tu.name == "search":
                    if searches >= self.max_searches:
                        results.append(_tool_result(tu.id, "Search limit reached. Call submit_answer now."))
                        continue
                    searches += 1
                    query = str((tu.input or {}).get("query") or "")
                    k = int((tu.input or {}).get("k") or self.context_k)
                    results.append(_tool_result(tu.id, self._run_search(query, k, retrieved)))
                else:
                    results.append(_tool_result(tu.id, f"Unknown tool: {tu.name}", is_error=True))
            messages.append({"role": "user", "content": results})
        # never submitted within the budget
        return Prediction(answer=REFUSAL, citations=[], retrieved=retrieved[:20])


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block
