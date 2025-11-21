from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .types import TargetCategory, EvaluationSample
from .metrics_base import Metric


@dataclass
class EvaluationPlan:
    metrics: List[Metric] = field(default_factory=list)

    def grouped_by_target(self) -> Dict[TargetCategory, List[Metric]]:
        out: Dict[TargetCategory, List[Metric]] = {}
        for m in self.metrics:
            out.setdefault(m.target, []).append(m)
        return out

    def validate_dataset(self, samples: Sequence[EvaluationSample]) -> None:
        if not samples:
            return

        for metric in self.metrics:
            required = metric.required_fields()
            for field_name in required:
                # We only require that AT LEAST ONE sample has the field, or that ALL samples?
                # The spec says: "We don’t enforce that every sample has all fields; some metrics can tolerate partial coverage"
                # But the spec code says: "if not any(...) raise ValueError"
                # This implies at least ONE sample must have the data, otherwise the metric is useless.
                if not any(getattr(s, field_name) is not None for s in samples):
                    raise ValueError(
                        f"Dataset missing required field '{field_name}' "
                        f"for metric '{metric.name}'."
                    )

