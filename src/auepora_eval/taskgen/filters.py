from __future__ import annotations

from typing import Protocol, List, Optional, Set

from ..core.types import EvaluationSample
from .config import LLMClient


class SampleFilter(Protocol):
    def keep(self, sample: EvaluationSample) -> bool:
        ...


class AnswerPresenceFilter:
    """
    Ensures that the answer string is present as a substring in at least one of the relevant docs.
    Useful for extractive single-hop tasks. Allows excluding task types (e.g., multi-hop).
    """

    def __init__(self, *, exclude_types: Optional[Set[str]] = None):
        self.exclude_types = exclude_types or set()

    def keep(self, sample: EvaluationSample) -> bool:
        sample_type = sample.labels.get("type")
        if sample_type in self.exclude_types:
            return True

        if not sample.reference_answer or not sample.reference_answer.text:
            return False

        if not sample.relevant_docs:
            return False

        answer_text = sample.reference_answer.text.lower()
        for doc in sample.relevant_docs:
            if answer_text in doc.text.lower():
                return True

        return False


class QuestionLengthFilter:
    """
    Filters out questions that are too short or too long.
    """

    def __init__(self, min_len: int = 5, max_len: int = 50):
        self.min_len = min_len
        self.max_len = max_len

    def keep(self, sample: EvaluationSample) -> bool:
        q_len = len(sample.query.split())
        return self.min_len <= q_len <= self.max_len


LLM_JUDGE_PROMPT = """You are verifying if an answer is supported by the provided passages.

Passages:
{context}

Question: {question}
Proposed Answer: {answer}

Respond with "YES" if the answer is fully supported by the passages, otherwise respond with "NO".
"""


class LLMConsistencyFilter:
    """
    Uses an LLM to verify that the answer is supported by the provided context.
    Helpful for multi-hop or abstractive answers where strict string matching fails.
    """

    def __init__(
        self,
        llm: LLMClient,
        include_types: Optional[Set[str]] = None,
        max_context_chars: int = 4000,
    ):
        self.llm = llm
        self.include_types = include_types or {"multi_hop"}
        self.max_context_chars = max_context_chars

    def keep(self, sample: EvaluationSample) -> bool:
        sample_type = sample.labels.get("type")
        if self.include_types and sample_type not in self.include_types:
            return True

        if not sample.reference_answer or not sample.reference_answer.text:
            return False

        if not sample.relevant_docs:
            return False

        context = "\n\n".join(doc.text for doc in sample.relevant_docs[:3])
        prompt = LLM_JUDGE_PROMPT.format(
            context=context[: self.max_context_chars],
            question=sample.query,
            answer=sample.reference_answer.text,
        )
        judgment = self.llm.generate(prompt).strip().lower()
        return judgment.startswith("yes")


from concurrent.futures import ThreadPoolExecutor

def apply_filters(samples: List[EvaluationSample], filters: List[SampleFilter], max_workers: int = 3) -> List[EvaluationSample]:
    if not samples:
        return []

    def _should_keep(s: EvaluationSample) -> bool:
        return all(f.keep(s) for f in filters)

    filtered = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map ensures order is preserved
        results = executor.map(_should_keep, samples)
        
        for s, keep in zip(samples, results):
            if keep:
                filtered.append(s)
                
    return filtered
