from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class TargetCategory(Enum):
    RETRIEVAL_RELEVANCE = auto()
    RETRIEVAL_ACCURACY = auto()
    GENERATION_RELEVANCE = auto()
    GENERATION_FAITHFULNESS = auto()
    GENERATION_CORRECTNESS = auto()
    LATENCY = auto()
    DIVERSITY = auto()
    NOISE_ROBUSTNESS = auto()
    NEGATIVE_REJECTION = auto()
    COUNTERFACTUAL_ROBUSTNESS = auto()


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    doc: Document
    score: float
    rank: int


@dataclass
class Response:
    text: str
    structured: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSample:
    sample_id: str
    query: str

    # Optional ground truths / annotations
    relevant_docs: Optional[List[Document]] = None
    candidate_docs: Optional[List[Document]] = None
    reference_answer: Optional[Response] = None

    labels: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemOutputs:
    retrieved: List[RetrievedDocument]
    response: Response

    timings: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    target: TargetCategory
    value: float
    details: Dict[str, Any] = field(default_factory=dict)

