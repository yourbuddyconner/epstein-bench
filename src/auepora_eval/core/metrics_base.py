from abc import ABC, abstractmethod
from typing import List

from .types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


class Metric(ABC):
    name: str
    target: TargetCategory

    @abstractmethod
    def required_fields(self) -> List[str]:
        """Return names of EvaluationSample attributes that must be non-None.

        Example: ["relevant_docs"], ["reference_answer"].
        This is used by EvaluationPlan to validate datasets.
        """
        ...

    @abstractmethod
    def compute(
        self,
        samples: List[EvaluationSample],
        outputs: List[SystemOutputs],
    ) -> MetricResult:
        ...

