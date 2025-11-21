from __future__ import annotations

from itertools import combinations
from typing import List, Optional

import numpy as np

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


class MeanLatency(Metric):
    def __init__(self, timing_key: str = "end_to_end"):
        self.name = f"mean_latency[{timing_key}]"
        self.target = TargetCategory.LATENCY
        self._key = timing_key

    def required_fields(self) -> List[str]:
        return []  # only needs outputs

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        vals = [out.timings[self._key] for out in outputs if self._key in out.timings]
        value = sum(vals) / len(vals) if vals else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(vals)},
        )


class QuantileLatency(Metric):
    def __init__(self, timing_key: str = "end_to_end", quantile: float = 0.95):
        assert 0 < quantile <= 1
        self._key = timing_key
        self.quantile = quantile
        self.name = f"latency_q{quantile:.2f}[{timing_key}]"
        self.target = TargetCategory.LATENCY

    def required_fields(self) -> List[str]:
        return []

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        vals = sorted(out.timings[self._key] for out in outputs if self._key in out.timings)
        if not vals:
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": 0, "quantile": self.quantile},
            )
        idx = int(np.ceil(self.quantile * len(vals))) - 1
        idx = max(0, min(idx, len(vals) - 1))
        value = vals[idx]
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(vals), "quantile": self.quantile},
        )


class DistinctN(Metric):
    def __init__(self, n: int = 2):
        assert n >= 1
        self.n = n
        self.name = f"distinct_{n}"
        self.target = TargetCategory.DIVERSITY

    def required_fields(self) -> List[str]:
        return []

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        total = 0
        unique = set()

        for out in outputs:
            tokens = out.response.text.split()
            if len(tokens) < self.n:
                continue
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i : i + self.n])
                unique.add(ngram)
                total += 1

        value = len(unique) / total if total > 0 else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(outputs)},
        )


class IntraListDiversity(Metric):
    def __init__(self, *, top_k: int = 5):
        self.top_k = top_k
        self.name = f"intra_list_diversity@{top_k}"
        self.target = TargetCategory.DIVERSITY

    def required_fields(self) -> List[str]:
        return []

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        scores: List[float] = []
        for out in outputs:
            embeddings = []
            for retrieved in out.retrieved[: self.top_k]:
                emb = retrieved.doc.metadata.get("embedding") if retrieved.doc.metadata else None
                if emb is not None:
                    embeddings.append(np.array(emb, dtype=float))
            if len(embeddings) < 2:
                continue

            sims = []
            for a, b in combinations(embeddings, 2):
                denom = np.linalg.norm(a) * np.linalg.norm(b)
                sims.append(float(np.dot(a, b) / denom)) if denom else sims.append(0.0)
            avg_sim = sum(sims) / len(sims) if sims else 0.0
            scores.append(1.0 - avg_sim)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )

