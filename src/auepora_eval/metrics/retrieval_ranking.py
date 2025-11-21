from __future__ import annotations

from math import log2
from typing import List, Optional, Set

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


def _gold_ids(sample: EvaluationSample) -> Set[str]:
    if not sample.relevant_docs:
        return set()
    return {d.doc_id for d in sample.relevant_docs if d.doc_id}


class MRRAtK(Metric):
    def __init__(self, k: Optional[int] = None):
        self.k = k
        suffix = f"@{k}" if k is not None else ""
        self.name = f"mrr{suffix}"
        self.target = TargetCategory.RETRIEVAL_ACCURACY

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
            gold = _gold_ids(sample)
            if not gold:
                continue

            limit = self.k if self.k is not None else len(out.retrieved)
            rr = 0.0
            for idx, retrieved in enumerate(out.retrieved[:limit]):
                if retrieved.doc.doc_id in gold:
                    rr = 1.0 / (idx + 1)
                    break
            scores.append(rr)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class MeanAveragePrecision(Metric):
    def __init__(self, k: Optional[int] = None):
        self.k = k
        suffix = f"@{k}" if k is not None else ""
        self.name = f"map{suffix}"
        self.target = TargetCategory.RETRIEVAL_ACCURACY

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
            gold = _gold_ids(sample)
            if not gold:
                continue

            limit = self.k if self.k is not None else len(out.retrieved)
            hits = 0
            precisions: List[float] = []
            for idx, retrieved in enumerate(out.retrieved[:limit]):
                if retrieved.doc.doc_id in gold:
                    hits += 1
                    precisions.append(hits / (idx + 1))

            ap = sum(precisions) / len(gold) if gold else 0.0
            scores.append(ap)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )


class NDCGAtK(Metric):
    def __init__(self, k: int):
        self.k = k
        self.name = f"ndcg@{k}"
        self.target = TargetCategory.RETRIEVAL_ACCURACY

    def required_fields(self) -> List[str]:
        return ["relevant_docs"]

    @staticmethod
    def _dcg(rel: List[int]) -> float:
        score = 0.0
        for idx, rel_i in enumerate(rel, start=1):
            score += (2**rel_i - 1) / log2(idx + 1)
        return score

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        scores: List[float] = []

        for sample, out in zip(samples, outputs):
            gold = _gold_ids(sample)
            if not gold:
                continue

            rel = [
                1 if retrieved.doc.doc_id in gold else 0
                for retrieved in out.retrieved[: self.k]
            ]
            dcg = self._dcg(rel)

            ideal_rel = [1] * min(len(gold), self.k)
            idcg = self._dcg(ideal_rel)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            scores.append(ndcg)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )

