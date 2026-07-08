"""Optional, producer-side SDK for building Epstein Bench systems.

This package helps you *produce* a conformant ``predictions.jsonl`` from a
``questions.jsonl``. It is entirely optional: the file contract is the ground
truth, and scoring recomputes everything from raw predictions. This module
imports nothing from ``score``/``pool`` and never loads gold data — the "systems
never import benchmark code" rule is about the *scoring* boundary, which a
producer-side helper does not cross.

Implement a ``Retriever`` and/or a ``System``, then hand it to ``run`` to get a
conformant predictions file. Reference implementations live in
``epstein_bench.sdk.systems`` (bm25/dense/hybrid/closed-book) and
``epstein_bench.sdk.agentic`` (an LLM tool-use agent).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..io_utils import parallel_map, read_jsonl, write_jsonl

# Correct behavior for tasks a system cannot ground (unanswerable / false_premise)
# is an explicit refusal; this is the canonical phrasing systems may emit.
REFUSAL = "The corpus does not contain enough information to answer this question."


@dataclass
class Task:
    """One row of ``questions.jsonl``. ``type`` may be ``None`` — a system is
    not required to be told the task type, which is closer to production."""

    task_id: str
    question: str
    type: str | None = None


@dataclass
class Prediction:
    """What a ``System`` returns for a task. ``run`` serializes this into the
    contract's ``predictions.jsonl`` shape and enforces the length caps."""

    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)


@runtime_checkable
class Retriever(Protocol):
    """Document-level ranking over the corpus. The reference retrievers already
    satisfy this; implement it to drop your own retriever into the agent."""

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs, best first."""
        ...


@runtime_checkable
class System(Protocol):
    """A system under test: turns a question into a grounded, cited answer."""

    def predict(self, task: Task) -> Prediction:
        ...


def run(
    system: System,
    questions_path: str | Path,
    out_path: str | Path,
    *,
    max_retrieved: int = 20,
    workers: int = 8,
) -> int:
    """Run ``system`` over a questions file and write a conformant predictions
    file. Fails closed: any per-task exception is recorded as a refusal rather
    than crashing the run. ``citations`` and ``retrieved`` are capped to the
    contract limit. Returns the number of predictions written."""
    questions = list(read_jsonl(questions_path))

    def predict_one(q: dict) -> dict:
        task = Task(task_id=q["task_id"], question=q["question"], type=q.get("type"))
        try:
            pred = system.predict(task)
        except Exception:  # noqa: BLE001 — a system error must not lose the run
            pred = Prediction(answer=REFUSAL)
        return {
            "task_id": task.task_id,
            "answer": str(pred.answer or ""),
            "citations": [str(c) for c in (pred.citations or [])][:max_retrieved],
            "retrieved": [str(r) for r in (pred.retrieved or [])][:max_retrieved],
        }

    rows = parallel_map(predict_one, questions, workers)
    return write_jsonl(out_path, rows)


__all__ = ["Task", "Prediction", "Retriever", "System", "run", "REFUSAL"]
