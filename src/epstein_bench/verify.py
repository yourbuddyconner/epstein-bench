"""The verification gauntlet. Every candidate passes all stages or is discarded.

Stages (cheap model for 1-3, strong model for 4):
  1. standalone     — question interpretable with no source document in view
  2. answerability  — a fresh prompt, given the gold docs, recovers the answer
  3. necessity      — closed-book and distractor-context attempts must FAIL;
                      multi-doc types must not be answerable from any single doc
  4. adjudication   — strong-model pass/fail with a failure category

Unanswerable tasks run stages 1 and 4 only (their absence check happened at
generation time). All failures are logged with the failing stage so the
generator can be tuned; failures never pass silently (LLM errors reject).
"""

from __future__ import annotations

import random
import re
from collections import Counter

from .config import Config
from .corpus import load_docs
from .io_utils import parallel_map, read_jsonl, write_jsonl
from .llm import LLM, LLMError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def token_f1(pred: str, gold: str) -> float:
    p = _TOKEN_RE.findall(pred.lower())
    g = _TOKEN_RE.findall(gold.lower())
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def _doc_context(doc_ids: list[str], docs_by_id: dict[str, dict], limit: int = 2500) -> str:
    return "\n\n".join(
        f"[DOC id={d}]\n{docs_by_id[d]['text'][:limit]}"
        for d in doc_ids
        if d in docs_by_id
    )


class Gauntlet:
    def __init__(self, config: Config, llm: LLM, docs: list[dict]):
        self.config = config
        self.llm = llm
        self.docs_by_id = {d["doc_id"]: d for d in docs}
        self.clean_doc_ids = [d["doc_id"] for d in docs if d["quality"] == "clean"]

    def _task_rng(self, task: dict) -> random.Random:
        # per-task seeding keeps distractor sampling deterministic under
        # parallel verification (a shared RNG would depend on thread order)
        return random.Random(f"{self.config.seed}:{task['task_id']}")

    # -- stage primitives ----------------------------------------------------

    def _attempt_answer(
        self, question: str, context: str | None, tag: str = "ANSWER"
    ) -> str | None:
        """Ask the cheap model to answer; None means it (correctly) couldn't.

        The tag names the gauntlet stage (ANSWER / CLOSEDBOOK / DISTRACTOR /
        SINGLEDOC) so stages are distinguishable in caches, logs, and stubs.
        """
        if context:
            prompt = (
                f"[{tag}] Answer the question using ONLY the documents provided. "
                'If they do not contain the answer, respond {"answer": null, '
                '"found": false}. Respond with JSON {"answer": str|null, '
                '"found": true|false}.'
                f"\n\nQuestion: {question}\n\n{context}"
            )
        else:
            prompt = (
                f"[{tag}] Answer this question from your own knowledge if you can. "
                'Respond with JSON {"answer": str|null, "found": true|false}.'
                f"\n\nQuestion: {question}"
            )
        resp = self.llm.chat_json(prompt)
        if resp.get("found") and resp.get("answer"):
            return str(resp["answer"])
        return None

    def _matches(self, prediction: str, reference: str) -> bool:
        resp = self.llm.chat_json(
            "[MATCH] Do these two answers state the same fact? Minor wording or "
            'formatting differences are fine. Respond with JSON {"match": true|false}.'
            f"\n\nReference: {reference}\nPrediction: {prediction}"
        )
        return bool(resp.get("match"))

    # -- stages ---------------------------------------------------------------

    def stage_standalone(self, task: dict) -> bool:
        resp = self.llm.chat_json(
            "[STANDALONE] A question for a document-corpus benchmark must be "
            "interpretable on its own: it names concrete people/organizations "
            "(no bare initials, no 'the document/this email'), and is not about "
            "generic boilerplate (disclaimers, signatures). Also FAIL questions "
            "that presuppose a unique unnamed artifact — 'the email from X', "
            "'the letter', 'the meeting' — unless the question itself pins it "
            "down with distinguishing details (subject, date range, recipient); "
            "a corpus can contain many emails from the same person. Does this "
            'question qualify? Respond with JSON {"standalone": true|false, '
            '"reason": str}.'
            f"\n\nQuestion: {task['question']}"
        )
        return bool(resp.get("standalone"))

    def stage_answerability(self, task: dict) -> bool:
        context = _doc_context(task["source_doc_ids"], self.docs_by_id)
        if task["type"] in ("aggregation", "dossier"):
            pred = self._attempt_answer(
                task["question"] + " (List every item you can find.)", context
            )
            if pred is None:
                return False
            recovered = sum(
                1
                for item in task["items"]
                if token_f1(pred, item["item"]) > 0 and item["item"].lower() in pred.lower()
            )
            return recovered / len(task["items"]) >= self.config.aggregation_recovery_floor
        pred = self._attempt_answer(task["question"], context)
        if pred is None:
            return False
        if not self._matches(pred, task["answer"]):
            return False
        return token_f1(pred, task["answer"]) >= self.config.answerability_f1_floor

    def stage_necessity(self, task: dict) -> bool:
        reference = (
            task["answer"]
            if task["answer"]
            else "; ".join(i["item"] for i in task["items"] or [])
        )
        # closed book must fail
        pred = self._attempt_answer(task["question"], None, tag="CLOSEDBOOK")
        if pred is not None and self._matches(pred, reference):
            return False
        # random distractor context must fail
        distractor_ids = [
            d for d in self._task_rng(task).sample(
                self.clean_doc_ids, min(3, len(self.clean_doc_ids))
            )
            if d not in task["source_doc_ids"]
        ]
        if distractor_ids:
            pred = self._attempt_answer(
                task["question"],
                _doc_context(distractor_ids, self.docs_by_id),
                tag="DISTRACTOR",
            )
            if pred is not None and self._matches(pred, reference):
                return False
        # multi-doc types: no single gold doc may suffice
        if task["type"] in ("aggregation", "timeline", "dossier") and len(task["source_doc_ids"]) > 1:
            for doc_id in task["source_doc_ids"]:
                pred = self._attempt_answer(
                    task["question"],
                    _doc_context([doc_id], self.docs_by_id),
                    tag="SINGLEDOC",
                )
                if pred is not None and self._matches(pred, reference):
                    return False
        return True

    def stage_adjudicate(self, task: dict) -> tuple[bool, str]:
        if task["type"] == "unanswerable":
            body = (
                f"Question: {task['question']}\n"
                "This is an intended-UNANSWERABLE task: the correct system behavior "
                "is to refuse. Judge whether the question is plausible, concrete, "
                "and clearly scoped (a system could not accidentally be right)."
            )
        else:
            reference = (
                task["answer"]
                if task["answer"]
                else "; ".join(i["item"] for i in task["items"] or [])
            )
            body = (
                f"Question: {task['question']}\nReference answer: {reference}\n\n"
                "Evidence:\n"
                + _doc_context(task["source_doc_ids"], self.docs_by_id, limit=2500)
            )
        resp = self.llm.chat_json(
            "[ADJUDICATE] You are the final quality gate for a public RAG "
            "benchmark. Fail anything ambiguous, wrongly answered, not grounded "
            "in the evidence, or trivially guessable. Respond with JSON "
            '{"pass": true|false, "category": "ok|ambiguous|wrong|not_grounded|trivial"}.'
            "\n\n" + body,
            model=self.config.strong_model,
        )
        return bool(resp.get("pass")), str(resp.get("category", "unspecified"))

    # -- driver ----------------------------------------------------------------

    def run(self, task: dict) -> tuple[bool, str]:
        """Return (passed, failure_reason). LLM errors fail closed."""
        try:
            if not self.stage_standalone(task):
                return False, "standalone"
            if task["type"] != "unanswerable":
                if not self.stage_answerability(task):
                    return False, "answerability"
                if not self.stage_necessity(task):
                    return False, "necessity"
            ok, category = self.stage_adjudicate(task)
            if not ok:
                return False, f"adjudication:{category}"
            return True, ""
        except LLMError as e:
            return False, f"error:{e}"


def verify_candidates(config: Config, llm: LLM) -> dict[str, int]:
    docs = load_docs(config)
    gauntlet = Gauntlet(config, llm, docs)
    candidates = list(read_jsonl(config.build_dir / "candidates.jsonl"))
    outcomes = parallel_map(
        lambda t: gauntlet.run(t), candidates, config.max_workers
    )
    verified: list[dict] = []
    rejected: list[dict] = []
    for task, (passed, reason) in zip(candidates, outcomes):
        if passed:
            task["provenance"]["verified"] = True
            task["provenance"]["adjudicator_model"] = config.strong_model
            verified.append(task)
        else:
            rejected.append({"task_id": task["task_id"], "type": task["type"], "reason": reason})
    write_jsonl(config.build_dir / "verified.jsonl", verified)
    write_jsonl(config.build_dir / "rejected.jsonl", rejected)
    reasons = Counter(r["reason"].split(":")[0] for r in rejected)
    return {
        "verified": len(verified),
        "rejected": len(rejected),
        **{f"rejected_{k}": v for k, v in reasons.items()},
    }
