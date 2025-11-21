from __future__ import annotations

from typing import List, Set

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


def _gold_ids(sample: EvaluationSample) -> Set[str]:
    if not sample.relevant_docs:
        return set()
    return {d.doc_id for d in sample.relevant_docs if d.doc_id}


class RecallAtK(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"recall@{k}"
        self.target = TargetCategory.RETRIEVAL_RELEVANCE

    def required_fields(self) -> List[str]:
        return ["relevant_docs"]

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []

        for sample, out in zip(samples, outputs):
            gold_ids = _gold_ids(sample)
            if not gold_ids:
                continue

            retrieved_ids = {r.doc.doc_id for r in out.retrieved[: self.k]}
            hit = len(gold_ids & retrieved_ids) / len(gold_ids)
            scores.append(hit)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class PrecisionAtK(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"precision@{k}"
        self.target = TargetCategory.RETRIEVAL_RELEVANCE

    def required_fields(self) -> List[str]:
        return ["relevant_docs"]

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []
        skipped = 0

        for sample, out in zip(samples, outputs):
            gold_ids = _gold_ids(sample)
            if not gold_ids:
                skipped += 1
                continue

            top = out.retrieved[: self.k]
            if not top:
                scores.append(0.0)
                continue

            retrieved_ids = [r.doc.doc_id for r in top]
            hits = sum(1 for doc_id in retrieved_ids if doc_id in gold_ids)
            denom = min(self.k, len(retrieved_ids))
            scores.append(hits / denom if denom > 0 else 0.0)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores), "num_skipped": skipped},
        )


class HitRateAtK(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"hit_rate@{k}"
        self.target = TargetCategory.RETRIEVAL_RELEVANCE

    def required_fields(self) -> List[str]:
        return ["relevant_docs"]

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        hits: List[float] = []

        for sample, out in zip(samples, outputs):
            gold_ids = _gold_ids(sample)
            if not gold_ids:
                continue
            top_ids = {r.doc.doc_id for r in out.retrieved[: self.k]}
            hits.append(1.0 if gold_ids & top_ids else 0.0)

        value = sum(hits) / len(hits) if hits else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(hits)},
        )

