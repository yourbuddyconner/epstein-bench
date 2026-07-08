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

import sys
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
    contract's ``predictions.jsonl`` shape and enforces the length caps.

    ``usage``/``cost_usd`` are optional operational telemetry (token counts and
    dollar cost of producing this answer). They are ignored by scoring but
    carried through so a leaderboard can report the accuracy/cost tradeoff —
    especially relevant for agentic systems that spend many tokens per task."""

    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    usage: dict | None = None
    cost_usd: float | None = None


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
    abort_ratio: float = 0.5,
) -> int:
    """Run ``system`` over a questions file and write a conformant predictions
    file. Returns the number of predictions written.

    Error policy — occasional per-task blips are tolerated (recorded as a
    refusal, the run continues), but a *systemic* failure (bad key, exhausted
    credits, wrong model) is surfaced, not swallowed: a canary probe of the
    first few tasks aborts immediately if they all error, and if more than
    ``abort_ratio`` of tasks error the run raises instead of writing a degraded
    file that would score as a pile of refusals. ``citations``/``retrieved``
    are capped to the contract limit."""
    questions = list(read_jsonl(questions_path))
    errors: dict[str, Exception] = {}

    def predict_one(q: dict) -> dict:
        task = Task(task_id=q["task_id"], question=q["question"], type=q.get("type"))
        try:
            pred = system.predict(task)
        except Exception as e:  # noqa: BLE001 — captured, judged in aggregate below
            errors[task.task_id] = e
            pred = Prediction(answer=REFUSAL)
        row = {
            "task_id": task.task_id,
            "answer": str(pred.answer or ""),
            "citations": [str(c) for c in (pred.citations or [])][:max_retrieved],
            "retrieved": [str(r) for r in (pred.retrieved or [])][:max_retrieved],
        }
        # optional operational telemetry — ignored by scoring, carried for the
        # accuracy/cost tradeoff on the leaderboard
        if pred.usage is not None:
            row["usage"] = pred.usage
        if pred.cost_usd is not None:
            row["cost_usd"] = pred.cost_usd
        return row

    # fail-fast canary: if the first few tasks ALL error, the system is broken
    # (not flaky) — abort before spending on the rest and surface the real error.
    probe_n = min(3, len(questions))
    probe_rows = [predict_one(q) for q in questions[:probe_n]]
    if probe_n and len(errors) == probe_n:
        raise RuntimeError(
            f"system failed on the first {probe_n} tasks — aborting run. "
            f"Last error: {errors[questions[probe_n - 1]['task_id']]!r}"
        )

    rows = probe_rows + parallel_map(predict_one, questions[probe_n:], workers)

    if errors and len(errors) >= max(1, int(len(questions) * abort_ratio)):
        example = next(iter(errors.values()))
        raise RuntimeError(
            f"{len(errors)}/{len(questions)} tasks errored — refusing to write a "
            f"degraded predictions file (it would score as spurious refusals). "
            f"Fix the underlying error and re-run. Example: {example!r}"
        )
    if errors:
        print(
            f"[sdk.run] warning: {len(errors)}/{len(questions)} tasks errored, "
            f"recorded as refusals; e.g. {next(iter(errors.values()))!r}",
            file=sys.stderr,
        )
    return write_jsonl(out_path, rows)


__all__ = ["Task", "Prediction", "Retriever", "System", "run", "REFUSAL"]
