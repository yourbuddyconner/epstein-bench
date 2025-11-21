from typing import List

from rouge_score import rouge_scorer

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


class RougeLAnswer(Metric):
    def __init__(self):
        self.name = "rougeL_answer"
        self.target = TargetCategory.GENERATION_CORRECTNESS
        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def required_fields(self) -> List[str]:
        return ["reference_answer"]

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        scores: List[float] = []

        for sample, out in zip(samples, outputs):
            if not sample.reference_answer:
                continue

            ref = sample.reference_answer.text
            pred = out.response.text
            score = self._scorer.score(ref, pred)["rougeL"].fmeasure
            scores.append(score)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )

