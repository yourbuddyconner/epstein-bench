from __future__ import annotations

from typing import Dict, List, Optional

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


class NoiseRobustness(Metric):
    def __init__(self, base_metric: Metric, noise_scenarios: Optional[List[str]] = None):
        self.base_metric = base_metric
        self.noise_scenarios = noise_scenarios or ["paraphrase", "typo"]
        self.name = f"noise_robustness[{base_metric.name}]"
        self.target = TargetCategory.NOISE_ROBUSTNESS

    def required_fields(self) -> List[str]:
        return self.base_metric.required_fields()

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)

        base_indices = [
            idx for idx, sample in enumerate(samples)
            if "variant_of" not in (sample.labels or {})
        ]
        noise_indices = [
            idx for idx, sample in enumerate(samples)
            if sample.labels and sample.labels.get("scenario") in self.noise_scenarios
        ]

        def _subset(indices: List[int]):
            return [samples[i] for i in indices], [outputs[i] for i in indices]

        base_samples, base_outputs = _subset(base_indices)
        noisy_samples, noisy_outputs = _subset(noise_indices)

        base_result = self.base_metric.compute(base_samples, base_outputs) if base_samples else MetricResult(self.base_metric.name, self.base_metric.target, 0.0, {"num_samples": 0})
        noise_result = self.base_metric.compute(noisy_samples, noisy_outputs) if noisy_samples else MetricResult(self.base_metric.name, self.base_metric.target, 0.0, {"num_samples": 0})

        base_score = base_result.value
        noisy_score = noise_result.value
        robustness = noisy_score / base_score if base_score > 1e-8 else 0.0

        return MetricResult(
            name=self.name,
            target=self.target,
            value=robustness,
            details={
                "base_score": base_score,
                "noise_score": noisy_score,
                "base_samples": base_result.details.get("num_samples", 0),
                "noise_samples": noise_result.details.get("num_samples", 0),
                "noise_scenarios": self.noise_scenarios,
            },
        )


class NegativeRejectionRate(Metric):
    def __init__(self, refusal_patterns: Optional[List[str]] = None):
        self.refusal_patterns = [p.lower() for p in (refusal_patterns or ["i don't know", "cannot answer", "not enough information"])]
        self.name = "negative_rejection_rate"
        self.target = TargetCategory.NEGATIVE_REJECTION

    def required_fields(self) -> List[str]:
        return []

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)
        negatives = [
            (sample, out)
            for sample, out in zip(samples, outputs)
            if sample.labels and sample.labels.get("scenario") == "unanswerable"
        ]

        if not negatives:
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": 0},
            )

        rejections = 0
        for sample, out in negatives:
            text = out.response.text.lower()
            if any(pattern in text for pattern in self.refusal_patterns):
                rejections += 1

        value = rejections / len(negatives)
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(negatives)},
        )


class CounterfactualConsistency(Metric):
    def __init__(self, similarity_metric: Metric, scenario_label: str = "counterfactual"):
        if not hasattr(similarity_metric, "pair_score"):
            raise ValueError("similarity_metric must expose a pair_score(text_a, text_b) method.")
        self.similarity_metric = similarity_metric
        self.scenario_label = scenario_label
        self.name = f"counterfactual_consistency[{similarity_metric.name}]"
        self.target = TargetCategory.COUNTERFACTUAL_ROBUSTNESS

    def required_fields(self) -> List[str]:
        return self.similarity_metric.required_fields()

    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        assert len(samples) == len(outputs)

        base_map: Dict[str, int] = {sample.sample_id: idx for idx, sample in enumerate(samples)}
        pairs: List[int] = []
        for idx, sample in enumerate(samples):
            labels = sample.labels or {}
            if labels.get("scenario") == self.scenario_label and "variant_of" in labels:
                base_idx = base_map.get(labels["variant_of"])
                if base_idx is not None:
                    pairs.append(idx)

        if not pairs:
            return MetricResult(
                name=self.name,
                target=self.target,
                value=0.0,
                details={"num_samples": 0},
            )

        base_samples = []
        base_outputs = []
        cf_samples = []
        cf_outputs = []

        for idx in pairs:
            variant_of = samples[idx].labels["variant_of"]
            base_idx = base_map[variant_of]
            base_samples.append(samples[base_idx])
            base_outputs.append(outputs[base_idx])
            cf_samples.append(samples[idx])
            cf_outputs.append(outputs[idx])

        similarities = []
        for base_output, cf_output in zip(base_outputs, cf_outputs):
            score = float(
                self.similarity_metric.pair_score(
                    base_output.response.text,
                    cf_output.response.text,
                )
            )
            similarities.append(score)

        value = sum(similarities) / len(similarities) if similarities else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(similarities)},
        )

