from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory, Document


class LLMCritic(ABC):
    """Abstract interface for LLM-based judges."""

    @abstractmethod
    def score(self, *, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Return a numeric score parsed from an LLM completion."""
        raise NotImplementedError


def _format_docs(docs: List[Document]) -> str:
    chunks = []
    for idx, doc in enumerate(docs, start=1):
        snippet = doc.text.strip()
        chunks.append(f"[Doc {idx}] {snippet}")
    return "\n\n".join(chunks)


class LLMFaithfulnessScore(Metric):
    def __init__(
        self,
        critic: LLMCritic,
        *,
        evidence_source: str = "retrieved",
        top_k: Optional[int] = 3,
    ):
        assert evidence_source in {"retrieved", "relevant"}
        self.critic = critic
        self.evidence_source = evidence_source
        self.top_k = top_k
        self.name = "llm_faithfulness"
        self.target = TargetCategory.GENERATION_FAITHFULNESS

    def required_fields(self) -> List[str]:
        if self.evidence_source == "relevant":
            return ["relevant_docs"]
        return []

    def _evidence(self, sample: EvaluationSample, output: SystemOutputs) -> List[Document]:
        if self.evidence_source == "relevant":
            return sample.relevant_docs or []
        docs = [r.doc for r in output.retrieved]
        if self.top_k is not None:
            docs = docs[: self.top_k]
        return docs

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []
        
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        def calc_score(item):
            sample, out = item
            evidence_docs = self._evidence(sample, out)
            if not evidence_docs:
                return None
            prompt = (
                "Determine how well the answer is supported by the provided evidence. "
                "Return a numeric score between 0 and 1 where 1 means fully supported.\n\n"
                f"Question:\n{sample.query}\n\n"
                f"Answer:\n{out.response.text}\n\n"
                f"Evidence:\n{_format_docs(evidence_docs)}\n\n"
                "Score:"
            )
            score = self.critic.score(prompt=prompt, metadata={"sample_id": sample.sample_id})
            score = max(0.0, min(1.0, float(score)))
            return score

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(calc_score, zip(samples, outputs))
            results = tqdm(results, total=len(samples), desc=f"Computing {self.name}")
            
            for res in results:
                if res is not None:
                    scores.append(res)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class LLMAnswerQuality(Metric):
    def __init__(
        self,
        critic: LLMCritic,
        *,
        question_weight: float = 1.0,
    ):
        self.critic = critic
        self.question_weight = question_weight
        self.name = "llm_answer_quality"
        self.target = TargetCategory.GENERATION_RELEVANCE

    def required_fields(self) -> List[str]:
        return []

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []
        
        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        
        def calc_score(item):
            sample, out = item
            prompt = (
                "Evaluate how well the answer addresses the question. "
                "Respond with a numeric score between 0 and 1 where 1 means perfectly relevant, "
                "helpful, and clear.\n\n"
                f"Question:\n{sample.query}\n\n"
                f"Answer:\n{out.response.text}\n\n"
                "Score:"
            )
            score = self.critic.score(prompt=prompt, metadata={"sample_id": sample.sample_id})
            score = max(0.0, min(1.0, float(score)))
            return score

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(calc_score, zip(samples, outputs))
            scores = list(tqdm(results, total=len(samples), desc=f"Computing {self.name}"))

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )

