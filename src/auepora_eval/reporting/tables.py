from typing import Dict, List

from ..core.types import MetricResult, TargetCategory


def results_to_dict(results: List[MetricResult]) -> Dict[str, Dict[str, float]]:
    """Group results by target category, then metric name.

    Output example:
    {
      "RETRIEVAL_RELEVANCE": {"recall@5": 0.83},
      "GENERATION_CORRECTNESS": {"rougeL_answer": 0.61}
    }
    """
    out: Dict[str, Dict[str, float]] = {}
    for r in results:
        target_name = r.target.name
        out.setdefault(target_name, {})[r.name] = r.value
    return out


def results_to_markdown(results: List[MetricResult]) -> str:
    lines = ["| Target | Metric | Value |", "|--------|--------|-------|"]
    for r in results:
        lines.append(f"| {r.target.name} | {r.name} | {r.value:.4f} |")
    return "\n".join(lines)

