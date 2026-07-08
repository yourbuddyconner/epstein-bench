"""The verification gauntlet. Every candidate passes all stages or is discarded.

Stages (cheap model for 1-3, strong model for 4):
  1. standalone     — question interpretable with no source document in view
  2. answerability  — a fresh prompt, given the gold docs, recovers the answer
  3. necessity      — closed-book and distractor-context attempts must FAIL;
                      multi-doc types must not be answerable from any single doc
  4. adjudication   — strong-model pass/fail with a failure category

Abstention/rejection tasks (unanswerable, false_premise) run stages 1 and 4
only (their absence check happened at generation time). All failures are logged
with the failing stage so the generator can be tuned; failures never pass
silently (LLM errors reject).
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

    def _count_recovered_items(self, items: list[str], prediction: str) -> int:
        """LLM-judged item recovery. Substring matching fails for list/timeline
        items (e.g. dated events) that are never reproduced verbatim."""
        resp = self.llm.chat_json(
            "[RECOVER] A reviewer answered a list/timeline question using the "
            "source documents. For each expected item, is it present in the "
            "reviewer's answer — same fact or dated event, even if the wording "
            "or date format differs? Respond with JSON "
            f'{{"present": [true|false x {len(items)}]}} in the given order.'
            f"\n\nExpected items: {items}\n\nReviewer answer: {prediction}"
        )
        present = list(resp.get("present") or [])
        return sum(1 for i in range(len(items)) if i < len(present) and present[i])

    def stage_answerability(self, task: dict) -> bool:
        context = _doc_context(task["source_doc_ids"], self.docs_by_id)
        if task["type"] in ("aggregation", "dossier"):
            pred = self._attempt_answer(
                task["question"] + " (List every item or dated event you can find.)",
                context,
            )
            if pred is None:
                return False
            recovered = self._count_recovered_items(
                [i["item"] for i in task["items"]], pred
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
        # each type gets an instruction matched to its success condition; the
        # generic "fail anything not grounded in the evidence" rule is right for
        # answerable tasks but wrong for the rejection types, whose whole point
        # is that no evidence grounds the question.
        header = "[ADJUDICATE] You are the final quality gate for a public RAG benchmark. "
        if task["type"] == "unanswerable":
            instruction = (
                header + "This is an intended-UNANSWERABLE task: the correct system "
                "behavior is to refuse. Pass it only if the question is plausible, "
                "concrete, and clearly scoped so a system could not accidentally be "
                'right. Respond with JSON {"pass": true|false, '
                '"category": "ok|ambiguous|trivial|answerable"}.'
            )
            body = f"Question: {task['question']}"
        elif task["type"] == "false_premise":
            # two focused judgments: a neutral support check, then a quality
            # check. A single omnibus prompt proved unstable exactly on the
            # cases that matter (premises that perturb only the date/place of a
            # *real* meeting, which must be dropped).
            return self._adjudicate_false_premise(task)
        else:
            instruction = (
                header + "Fail anything ambiguous, wrongly answered, not grounded in "
                "the evidence, or trivially guessable. Respond with JSON "
                '{"pass": true|false, "category": "ok|ambiguous|wrong|not_grounded|trivial"}.'
            )
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
        resp = self.llm.chat_json(instruction + "\n\n" + body, model=self.config.strong_model)
        return bool(resp.get("pass")), str(resp.get("category", "unspecified"))

    def _adjudicate_false_premise(self, task: dict) -> tuple[bool, str]:
        """Two-stage adjudication for false_premise (both on the strong model).

        1. Support: judged neutrally from the most on-topic documents, does the
           corpus support the presupposed fact? 'supports' -> drop (the premise
           is really true, only a detail perturbed; scoring would wrongly punish
           a correct system). 'absent' (unsupported) or 'contradicts' (corpus
           refutes it) both make a legitimate rejection task.
        2. Quality: the premise must sound plausible/concrete and must not tip
           off, in its wording, that it is false.
        """
        docs = _doc_context(
            task["provenance"].get("absence_doc_ids", []), self.docs_by_id, limit=1500
        )
        support = self.llm.chat_json(
            "[FPSUPPORT] Below is a factual CLAIM and some documents. Judging ONLY "
            "from the documents, does the evidence support that the claim is true? "
            "Answer 'supports' if any document states or clearly implies it (even "
            "with a differing detail such as date or place); 'contradicts' if a "
            "document shows it did not happen; 'absent' if the documents neither "
            'support nor contradict it. Respond JSON {"verdict": '
            '"supports|contradicts|absent"}.'
            f"\n\nCLAIM: {task.get('false_element') or task['question']}\n\nDocuments:\n{docs}",
            model=self.config.strong_model,
        )
        if str(support.get("verdict")) == "supports":
            return False, "supported"
        quality = self.llm.chat_json(
            "[FPQUALITY] A benchmark question deliberately presupposes a FABRICATED "
            "fact; a good system must reject it. Judge only the question's wording, "
            "not whether the fact is true. Is the presupposed fact concrete and "
            "believable to a reader who has not seen the source documents (real-"
            "seeming named entities, a coherent specific event), rather than absurd "
            "or generic? And does the wording avoid hinting that the premise is "
            'false? Respond JSON {"plausible": true|false, "tips_off": true|false}.'
            f"\n\nQuestion: {task['question']}"
            f"\nPresupposed (fabricated) fact: {task.get('false_element') or 'N/A'}",
            model=self.config.strong_model,
        )
        if not quality.get("plausible"):
            return False, "implausible"
        if quality.get("tips_off"):
            return False, "tips_off"
        return True, "ok"

    # -- driver ----------------------------------------------------------------

    def run(self, task: dict) -> tuple[bool, str]:
        """Return (passed, failure_reason). LLM errors fail closed."""
        try:
            if not self.stage_standalone(task):
                return False, "standalone"
            # abstention/rejection types carry no gold answer to recover; their
            # absence check happened at generation time, so they skip 2-3
            if task["type"] not in ("unanswerable", "false_premise"):
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
