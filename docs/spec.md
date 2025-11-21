# Auepora-Based RAG Evaluation Library – Engineering Spec

## 1. Purpose and Scope

We want a Python library that operationalizes the **Auepora evaluation framework** for Retrieval-Augmented Generation (RAG) systems.

The library should:

1. Encode Auepora’s conceptual separation into:

   * **Targets** – what aspects of system behavior we care about.
   * **Datasets** – what annotated data is required to evaluate those targets.
   * **Metrics** – how we turn predictions + ground truth into scores.
2. Provide a **pluggable interface** for arbitrary RAG systems (retriever + generator or end-to-end pipeline).
3. Allow users to define evaluation plans that specify which targets and metrics to run, then execute these plans over a dataset.
4. Produce machine- and human-consumable outputs (Python objects, JSON, optional tabular/markdown summaries).

This document explains:

* How Auepora’s framework structures the evaluation space for RAG.
* How we map that framework to concrete Python types and modules.
* How to implement each module and how they interact.

## 2. Auepora Framework – Conceptual Overview

### 2.1 Entities in a RAG System

A typical RAG system has:

* **Query** (q): user’s question / task.
* **Corpus** (C): collection of documents (d) that the retriever searches over.
* **Retriever**: maps (q) → ranked list of documents (R(q) = [d_1, d_2, …]).
* **Generator**: consumes (q) + retrieved docs and outputs a **Response** (y).

Auepora’s evaluation framework formalizes **what we assess** at three levels:

1. **Retrieval**

   * How well does the retriever fetch relevant documents?
   * How accurate is the ranked list compared to ground-truth relevant docs or candidate sets?
2. **Generation**

   * How relevant is the generated answer to the query?
   * How faithful is it to the retrieved evidence?
   * How correct is it with respect to reference answers?
3. **Additional / System-Level Aspects**

   * Robustness to noisy queries / adversarial inputs.
   * Ability to reject negatives / counterfactuals.
   * Latency, diversity, etc.

In the paper, these dimensions are grouped into a **Target space** with multiple sub-targets, plus a **Dataset space** (what annotations we need) and a **Metric space** (how we measure each target).

### 2.2 Target–Dataset–Metric Decomposition

Auepora decomposes evaluation design into three coupled choices:

1. **Targets (T)** – what behavior we want to measure.

   * Examples:

     * Retrieval relevance: do we retrieve relevant docs for the query?
     * Generation correctness: is the answer semantically correct vs. a reference answer?
     * Faithfulness: does the answer stay grounded in retrieved docs?
     * Latency, diversity, robustness, etc.

2. **Dataset (D)** – what annotated data we have.

   * For each query we may have:

     * Ground-truth relevant docs.
     * Candidate doc sets.
     * Reference answers.
     * Labels for robustness scenarios (e.g. adversarial / counterfactual variants).

3. **Metrics (M)** – how we map system outputs + dataset to numbers.

   * E.g. Recall@k, MRR, MAP for retrieval.
   * ROUGE, BLEU, BERTScore for text answers.
   * LLM-as-a-judge scores (faithfulness / coherence) if plugged in.
   * Structured measures like accuracy for QA, etc.

The evaluation plan is essentially selecting **which Targets to evaluate**, ensuring the **Dataset provides required annotations**, and then configuring **Metrics** that operate on those pieces.

## 3. Top-Level Library Design

### 3.1 Package Layout

We’ll use the following top-level structure:

```text
auepora_eval/
  __init__.py
  core/
    __init__.py
    types.py          # Shared dataclasses and enums
    targets.py        # Target definitions / taxonomies (thin over enums in types.py)
    rag_interface.py  # Abstract interfaces for RAG systems
    dataset.py        # Dataset abstraction and in-memory implementation
    metrics_base.py   # Metric abstraction and helpers
    plan.py           # EvaluationPlan binding targets + metrics
    runner.py         # AueporaEvaluator orchestrator
  metrics/
    __init__.py
    retrieval_basic.py    # Recall@k, MRR, MAP, nDCG, etc.
    generation_text.py    # ROUGE, BLEU, BERTScore, etc.
    additional.py         # Latency, robustness, diversity, etc.
  io/
    __init__.py
    jsonl.py          # JSONL loader/saver for datasets
    parquet.py        # (optional) Parquet loader/saver
  reporting/
    __init__.py
    tables.py         # MetricResult → tables / dicts
    plots.py          # (optional) simple matplotlib helpers
  examples/
    __init__.py
    simple_rag.py     # Example of plugging in a toy RAG
    example_plan.py   # Example of building an EvaluationPlan
```

## 4. Core Data Model (`core/types.py`)

### 4.1 Targets

Implementation: an enum covering Auepora-style evaluation targets. We can extend this later, but v1 should support:

* Retrieval:

  * `RETRIEVAL_RELEVANCE` – relevance of retrieved docs to query.
  * `RETRIEVAL_ACCURACY` – ranking accuracy vs. ground truth candidates.
* Generation:

  * `GENERATION_RELEVANCE` – answer relevance to query.
  * `GENERATION_FAITHFULNESS` – answer groundedness in evidence docs.
  * `GENERATION_CORRECTNESS` – answer correctness vs. reference answer.
* Additional / system-level:

  * `LATENCY` – time-based performance.
  * `DIVERSITY` – variety of retrieved docs or responses.
  * `NOISE_ROBUSTNESS`, `NEGATIVE_REJECTION`, `COUNTERFACTUAL_ROBUSTNESS` – robustness aspects.

Code sketch:

```python
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
```

### 4.2 Documents and Responses

We keep these neutral and minimal so they can wrap vector-store docs, DB rows, etc.

```python
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
```

### 4.3 EvaluationSample (Dataset Row)

Represents one row of the evaluation dataset.

```python
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
```

Notes:

* `relevant_docs`: Required by retrieval relevance metrics.
* `candidate_docs`: Required by some ranking accuracy metrics.
* `reference_answer`: Required by correctness metrics (ROUGE, BLEU, etc.).
* `labels` can be used for robustness scenarios (e.g., `{"adversarial": true}` or `{"scenario": "counterfactual"}`).

### 4.4 SystemOutputs (Evaluable Outputs)

Outputs of the user’s RAG system for one sample.

```python
@dataclass
class SystemOutputs:
    retrieved: List[RetrievedDocument]
    response: Response

    timings: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
```

Notes:

* `timings` may include keys like `"end_to_end"`, `"retrieval"`, `"generation"`.
* `extra` can store traces, model configs, etc., which may later be used by advanced metrics.

### 4.5 MetricResult

```python
@dataclass
class MetricResult:
    name: str
    target: TargetCategory
    value: float
    details: Dict[str, Any] = field(default_factory=dict)
```

## 5. RAG System Interfaces (`core/rag_interface.py`)

### 5.1 Primary Interface: `RAGSystem`

We define an abstract base class that users implement for their RAG stack.

```python
from abc import ABC, abstractmethod
from typing import Protocol

from .types import EvaluationSample, SystemOutputs


class RAGSystem(ABC):
    @abstractmethod
    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        """Run the full RAG pipeline on a given sample.

        Responsibilities:
        - Perform retrieval using `sample.query`.
        - Perform generation using the retrieved docs (and possibly `sample` fields).
        - Return retrieved docs and final response wrapped in `SystemOutputs`.
        """
        raise NotImplementedError


# Optional protocol type for duck-typing
class RAGSystemProtocol(Protocol):
    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs: ...
```

### 5.2 Optional Split Interfaces: Retriever / Generator

Some users will have separate retriever and generator components. We provide thin abstractions and a default adapter.

```python
from typing import List

from .types import RetrievedDocument, Document, Response


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, *, top_k: int = 5) -> List[RetrievedDocument]:
        ...


class Generator(ABC):
    @abstractmethod
    def generate(self, query: str, context_docs: List[Document]) -> Response:
        ...


class SimpleRAGSystem(RAGSystem):
    def __init__(self, retriever: Retriever, generator: Generator):
        self._retriever = retriever
        self._generator = generator

    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        retrieved = self._retriever.retrieve(sample.query, top_k=top_k)
        docs = [r.doc for r in retrieved]
        response = self._generator.generate(sample.query, docs)
        return SystemOutputs(retrieved=retrieved, response=response)
```

## 6. Dataset Abstractions (`core/dataset.py`)

### 6.1 Base Class: `EvaluationDataset`

Represents a collection of `EvaluationSample` rows.

```python
from abc import ABC, abstractmethod
from typing import Iterator, List

from .types import EvaluationSample


class EvaluationDataset(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[EvaluationSample]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
```

### 6.2 In-Memory Implementation

```python
class InMemoryDataset(EvaluationDataset):
    def __init__(self, name: str, samples: List[EvaluationSample]):
        self._name = name
        self._samples = samples

    def __iter__(self) -> Iterator[EvaluationSample]:
        return iter(self._samples)

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._samples)
```

### 6.3 IO Helpers (`io/jsonl.py`)

We’ll support a simple JSONL format, one sample per line.

Example JSONL line:

```json
{
  "sample_id": "1",
  "query": "Who wrote 'Attention Is All You Need'?",
  "relevant_docs": [
    {"doc_id": "d1", "text": "... Vaswani et al ...", "metadata": {}}
  ],
  "reference_answer": {
    "text": "Ashish Vaswani et al.",
    "metadata": {}
  }
}
```

Loader design:

```python
import json
from pathlib import Path
from typing import Optional

from ..core.dataset import InMemoryDataset
from ..core.types import EvaluationSample, Document, Response


def load_jsonl_dataset(path: str | Path, name: Optional[str] = None) -> InMemoryDataset:
    path = Path(path)
    if name is None:
        name = path.stem

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Map dict → EvaluationSample
            relevant_docs = None
            if "relevant_docs" in raw:
                relevant_docs = [Document(**d) for d in raw["relevant_docs"]]

            candidate_docs = None
            if "candidate_docs" in raw:
                candidate_docs = [Document(**d) for d in raw["candidate_docs"]]

            reference_answer = None
            if "reference_answer" in raw:
                reference_answer = Response(**raw["reference_answer"])

            sample = EvaluationSample(
                sample_id=raw["sample_id"],
                query=raw["query"],
                relevant_docs=relevant_docs,
                candidate_docs=candidate_docs,
                reference_answer=reference_answer,
                labels=raw.get("labels", {}),
                metadata=raw.get("metadata", {}),
            )
            samples.append(sample)

    return InMemoryDataset(name=name, samples=samples)
```

## 7. Metrics Abstraction (`core/metrics_base.py`)

Metrics connect Auepora’s **Target space** and **Dataset space**: each metric declares which target it addresses and what dataset fields it requires.

### 7.1 Base Metric Interface

```python
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
```

### 7.2 Retrieval Metrics (`metrics/retrieval_basic.py`)

#### 7.2.1 Recall@k

```python
from typing import List, Set

from ..core.metrics_base import Metric
from ..core.types import EvaluationSample, SystemOutputs, MetricResult, TargetCategory


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
            if not sample.relevant_docs:
                continue

            gold_ids: Set[str] = {d.doc_id for d in sample.relevant_docs}
            if not gold_ids:
                continue

            topk = out.retrieved[: self.k]
            retrieved_ids: Set[str] = {r.doc.doc_id for r in topk}

            hit = len(gold_ids & retrieved_ids) / len(gold_ids)
            scores.append(hit)

        value = sum(scores) / len(scores) if scores else 0.0
        return MetricResult(
            name=self.name,
            target=self.target,
            value=value,
            details={"num_samples": len(scores)},
        )
```

We can later add MRR, MAP, nDCG, etc., using the same pattern.

### 7.3 Generation Metrics (`metrics/generation_text.py`)

#### 7.3.1 ROUGE-L for Answers

We’ll use `rouge-score` (installable via `pip install rouge-score`). The metric measures similarity between generated response and reference answer.

```python
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
```

### 7.4 Additional Metrics (`metrics/additional.py`)

#### 7.4.1 Mean Latency

Requires the user to populate `SystemOutputs.timings`.

```python
from typing import List

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
```

We can later extend this file with metrics for robustness (e.g., compare scores across adversarial vs. baseline subsets of the dataset using `EvaluationSample.labels`).

## 8. EvaluationPlan: Wiring Targets–Datasets–Metrics (`core/plan.py`)

The **EvaluationPlan** is the concrete realization of Auepora’s configuration stage: it specifies **which metrics to run**, and ensures the dataset provides the required annotations.

### 8.1 Data Model

```python
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .types import TargetCategory, EvaluationSample, MetricResult
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
                if not any(getattr(s, field_name) is not None for s in samples):
                    raise ValueError(
                        f"Dataset missing required field '{field_name}' "
                        f"for metric '{metric.name}'."
                    )
```

Notes:

* We don’t enforce that **every** sample has all fields; some metrics can tolerate partial coverage (the metric implementation can skip samples without required ground truth).
* `grouped_by_target` is useful for reporting grouped by target category.

## 9. Orchestrator: AueporaEvaluator (`core/runner.py`)

This is the top-level entry point that engineers and users interact with.

### 9.1 Responsibilities

* Iterate over the dataset.
* For each sample, call the user’s `RAGSystem.run` to produce `SystemOutputs`.
* Validate dataset fields against metrics.
* Run all metrics and collect `MetricResult` objects.
* Optionally, return per-sample outputs for debugging.

### 9.2 Implementation

```python
from typing import List

from .dataset import EvaluationDataset
from .rag_interface import RAGSystem
from .plan import EvaluationPlan
from .types import EvaluationSample, SystemOutputs, MetricResult


class AueporaEvaluator:
    def __init__(self, system: RAGSystem, plan: EvaluationPlan, *, top_k: int = 5):
        self.system = system
        self.plan = plan
        self.top_k = top_k

    def evaluate(self, dataset: EvaluationDataset) -> List[MetricResult]:
        samples: List[EvaluationSample] = []
        outputs: List[SystemOutputs] = []

        for sample in dataset:
            samples.append(sample)
            out = self.system.run(sample, top_k=self.top_k)
            outputs.append(out)

        # Auepora Dataset step – validate annotations vs. planned Metrics
        self.plan.validate_dataset(samples)

        # Auepora Metric step – compute results
        results: List[MetricResult] = []
        for metric in self.plan.metrics:
            res = metric.compute(samples, outputs)
            results.append(res)

        return results
```

Possible future extensions (but not required in v1):

* Parallelization (ThreadPool / multiprocessing) around `system.run`.
* Callbacks/hooks for logging and tracing.
* Caching of `SystemOutputs`.

## 10. Reporting Helpers (`reporting/tables.py`)

We want simple utilities to convert metric results to:

* A `dict` suitable for JSON serialization.
* A markdown table for CLI printing or docs.

```python
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
```

## 11. Example Usage (End-to-End)

This example shows how an engineer plugs in their system and runs the Auepora-style evaluation.

```python
from auepora_eval.core.types import EvaluationSample, Document, Response
from auepora_eval.core.dataset import InMemoryDataset
from auepora_eval.core.rag_interface import RAGSystem, SystemOutputs
from auepora_eval.core.plan import EvaluationPlan
from auepora_eval.core.runner import AueporaEvaluator
from auepora_eval.metrics.retrieval_basic import RecallAtK
from auepora_eval.metrics.generation_text import RougeLAnswer


class MyRAGSystem(RAGSystem):
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        retrieved = self.retriever.retrieve(sample.query, top_k=top_k)
        context = "\n\n".join(r.doc.text for r in retrieved)
        prompt = f"Question: {sample.query}\n\nContext:\n{context}\n\nAnswer:"
        answer_text = self.llm(prompt)

        resp = Response(text=answer_text)
        return SystemOutputs(retrieved=retrieved, response=resp)


# Build dataset
samples = [
    EvaluationSample(
        sample_id="1",
        query="Who wrote the paper 'Attention Is All You Need'?",
        relevant_docs=[Document(doc_id="d1", text="... Vaswani et al ...")],
        reference_answer=Response(text="Ashish Vaswani et al."),
    ),
]

dataset = InMemoryDataset(name="toy", samples=samples)

# Build EvaluationPlan
plan = EvaluationPlan(metrics=[RecallAtK(k=5), RougeLAnswer()])

# Run evaluation
evaluator = AueporaEvaluator(system=MyRAGSystem(retriever, llm), plan=plan, top_k=5)
results = evaluator.evaluate(dataset)

for r in results:
    print(r.name, r.value, r.details)
```

## 12. Engineering Tasks and Milestones

**Milestone 1 – Core Skeleton**

* [ ] Implement `core/types.py` with all dataclasses and enums.
* [ ] Implement `core/rag_interface.py` with `RAGSystem`, `Retriever`, `Generator`, `SimpleRAGSystem`.
* [ ] Implement `core/dataset.py` with `EvaluationDataset` and `InMemoryDataset`.

**Milestone 2 – Metrics and Plan**

* [ ] Implement `core/metrics_base.py`.
* [ ] Implement basic metrics:

  * [ ] `RecallAtK` in `metrics/retrieval_basic.py`.
  * [ ] `RougeLAnswer` in `metrics/generation_text.py`.
  * [ ] `MeanLatency` in `metrics/additional.py`.
* [ ] Implement `core/plan.py`.

**Milestone 3 – Runner and IO**

* [ ] Implement `core/runner.py` (`AueporaEvaluator`).
* [ ] Implement `io/jsonl.py` with `load_jsonl_dataset`.
* [ ] Add `reporting/tables.py` with simple helpers.

**Milestone 4 – Examples and Docs**

* [ ] Add `examples/simple_rag.py` showing a toy retriever + generator.
* [ ] Add `examples/example_plan.py` building an EvaluationPlan covering retrieval + generation.
* [ ] Document how to map Auepora’s target types to `TargetCategory` enum in README.

This spec should give the engineer everything needed to build a first working version of the Auepora-based RAG evaluation library. The key is that all code concepts (Targets, Dataset, Metrics, EvaluationPlan, Evaluator) mirror the decomposition in the paper, making it straightforward to extend with more advanced targets and metrics later.
