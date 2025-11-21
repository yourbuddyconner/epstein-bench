# Auepora Task Generation Module – Engineering Spec

## 1. Purpose and Scope

This document specifies the design and implementation details for a **task generation module** that turns a raw corpus into Auepora-style evaluation tasks.

The module must:

1. Generate `EvaluationSample` objects (as defined in the main Auepora library) from:

   * A generic in-memory corpus (`List[Document]`).
   * A **Hugging Face dataset** (via `datasets.Dataset` / `datasets.DatasetDict`).
2. Support multiple task types:

   * Single-hop factoid / definitional Q&A.
   * Multi-hop / compositional questions that require multiple documents.
   * Robustness variants (paraphrased / noisy queries, unanswerable queries, etc.).
3. Be pluggable with arbitrary LLM clients and retrievers.
4. Produce outputs that can be fed directly into `InMemoryDataset` (or any `EvaluationDataset` implementation) for evaluation with `AueporaEvaluator`.

This module operationalizes the **Dataset side** of the Auepora framework for synthetic / semi-synthetic benchmarks: given a corpus, it builds the annotated evaluation dataset required by the Metrics and Targets we already defined.

## 2. Relationship to the Core Library

### 2.1 Dependencies

The task generation module will live alongside the core Auepora evaluation library and depend on:

* `core.types`:

  * `Document`
  * `EvaluationSample`
  * `Response`
* `core.dataset`:

  * `InMemoryDataset` (or an interface to construct other `EvaluationDataset` implementations).

It must **not** depend on metrics or on the evaluator; it only generates datasets.

### 2.2 Package Layout

Add a new subpackage:

```text
auepora_eval/
  taskgen/
    __init__.py
    config.py        # Config objects for task generation
    hf_adapter.py    # Hugging Face dataset → Document[] adapter
    preprocess.py    # Chunking, embeddings, NER, index building
    anchors.py       # Anchor extraction from chunks
    single_hop.py    # Single-hop QA task generation
    multi_hop.py     # Multi-hop task generation
    robustness.py    # Robustness variants (paraphrases, noisy, negative)
    filters.py       # Quality filters
    generator.py     # High-level TaskGenerator orchestrator
```

## 3. Data Contracts

### 3.1 Core Types (reused)

We reuse the `Document`, `EvaluationSample`, and `Response` classes from `core.types`.

* `Document` represents a chunk-level unit in the corpus.
* `EvaluationSample` represents one evaluation task.
* `Response` is the reference answer object.

### 3.2 Task Types (labels)

Different task generators will label samples using `EvaluationSample.labels`, e.g.:

* `{"type": "single_hop_factoid"}`
* `{"type": "multi_hop", "hops": 2}`
* `{"scenario": "paraphrase", "variant_of": "<sample_id>"}`
* `{"scenario": "unanswerable"}`

These labels are not enforced by the core library but should be consistently used in this module.

## 4. Hugging Face Dataset Support

### 4.1 Requirements

The module must:

1. Accept a Hugging Face dataset (or dataset name + config) as input.
2. Support basic configuration of which columns map to:

   * Document ID
   * Text
   * Optional metadata
3. Convert that dataset into a list of `Document` objects suitable for downstream chunking and task generation.

### 4.2 Design: `HFCorpusConfig`

Create a config object describing how to map HF dataset rows to `Document`.

```python
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any


@dataclass
class HFCorpusConfig:
    # Either pass an already loaded dataset, or dataset loading params
    dataset_name: Optional[str] = None          # e.g. "wikipedia"
    dataset_config_name: Optional[str] = None   # e.g. "20220301.en"
    split: str = "train"

    # Column mapping
    text_column: str = "text"           # column with main text
    id_column: Optional[str] = None      # if None, auto-generate IDs

    # Optional: function to build metadata from a row
    metadata_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    # If the user passes an existing Dataset object, we skip load_dataset
    dataset_obj: Any | None = None  # type: ignore
```

Notes:

* The engineer should use `Any` type for `dataset_obj` to avoid hard dependency in type hints; at runtime it will be a `datasets.Dataset` or `DatasetDict`.
* `metadata_fn` receives the raw row dict and returns a small metadata dict attached to each `Document`.

### 4.3 Implementation: `HFCorpusAdapter`

`HFCorpusAdapter` encapsulates Hugging Face dataset loading and conversion to `Document` list.

```python
from typing import List

from datasets import load_dataset, Dataset, DatasetDict

from ..core.types import Document
from .config import HFCorpusConfig


class HFCorpusAdapter:
    def __init__(self, config: HFCorpusConfig):
        self.config = config
        self._dataset: Dataset | None = None

    def load(self) -> Dataset:
        if self.config.dataset_obj is not None:
            ds = self.config.dataset_obj
            # If DatasetDict, pick split
            if isinstance(ds, DatasetDict):
                self._dataset = ds[self.config.split]
            else:
                self._dataset = ds
        else:
            ds = load_dataset(
                self.config.dataset_name,
                self.config.dataset_config_name,
                split=self.config.split,
            )
            self._dataset = ds
        return self._dataset

    def to_documents(self) -> List[Document]:
        if self._dataset is None:
            self.load()
        assert self._dataset is not None

        docs: List[Document] = []
        for idx, row in enumerate(self._dataset):
            doc_id = (
                str(row[self.config.id_column])
                if self.config.id_column is not None
                else f"hf_doc_{idx}"
            )
            text = row[self.config.text_column]
            metadata = (
                self.config.metadata_fn(row)
                if self.config.metadata_fn is not None
                else {}
            )
            docs.append(Document(doc_id=doc_id, text=text, metadata=metadata))

        return docs
```

The rest of the task generation pipeline consumes `List[Document]` regardless of whether those came from HF or elsewhere.

## 5. Preprocessing and Anchor Extraction

### 5.1 Chunking and Embeddings (`preprocess.py`)

We need to chunk long documents and optionally build embeddings.

Design:

```python
from dataclasses import dataclass
from typing import List, Callable, Any

from ..core.types import Document


@dataclass
class ChunkingConfig:
    max_tokens: int = 256
    overlap_tokens: int = 32


@dataclass
class EmbeddingConfig:
    embed_fn: Callable[[List[str]], List[List[float]]]
    batch_size: int = 32


@dataclass
class PreprocessResult:
    chunks: List[Document]
    embeddings: List[List[float]]  # same length as chunks
```

Implementation sketch:

* `chunk_documents(docs: List[Document], cfg: ChunkingConfig) -> List[Document]`

  * Split `doc.text` into token chunks with overlap; maintain `metadata` and add `{"parent_doc_id": doc.doc_id, "chunk_index": i}`.
* `embed_chunks(chunks: List[Document], cfg: EmbeddingConfig) -> List[List[float]]`

  * Call `cfg.embed_fn` in batches.

We will also need a simple index abstraction (e.g. thin wrapper around FAISS or user-provided retriever) for later steps.

### 5.2 Anchors (`anchors.py`)

An **Anchor** is an interesting sentence/span within a chunk that we can turn into QA.

```python
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Anchor:
    doc_id: str
    sentence: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
```

`extract_anchors(chunks: List[Document], max_anchors_per_chunk: int = 3) -> List[Anchor]`:

* For each chunk:

  * Split into sentences.
  * Score sentences with simple heuristics:

    * Length within a range.
    * Contains at least one capitalized token or number.
  * Keep top N per chunk.
* Fill `metadata` with `{ "chunk_doc_id": chunk.doc_id }` and any extra info (entities, section, etc.).

## 6. LLM + Retriever Interfaces for Task Generation

### 6.1 LLM Client Protocol

We want to stay agnostic to specific LLMs. Define a minimal protocol in `taskgen/config.py`:

```python
from typing import Protocol, List


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...

    # Optionally later: support chat-style, JSON outputs, etc.
```

The engineer will implement prompt templates inside each generator module using this interface.

### 6.2 Retriever Interface

For multi-hop and negative sampling we may need a retriever on the **corpus chunks**.

You can reuse the `Retriever` interface defined in `core.rag_interface`, or define a simple internal interface:

```python
from typing import List, Tuple

from ..core.types import Document


class CorpusRetriever(Protocol):
    def search(self, query: str, k: int = 10) -> List[Document]:
        ...
```

Implementation detail: For v1, an engineer can build a trivial embedding-based retriever over `PreprocessResult.embeddings` using nearest neighbors.

## 7. Single-Hop Task Generation (`single_hop.py`)

### 7.1 High-Level Algorithm

For each `Anchor`:

1. Select an answer span (substring of `anchor.sentence`).
2. Ask the LLM to generate a question whose answer is that span and is answerable from the sentence.
3. Find relevant documents (chunks) containing this answer span.
4. Build `EvaluationSample` with:

   * `query` = generated question.
   * `relevant_docs` = list of `Document` objects with the answer span.
   * `reference_answer` = `Response` built from the span / sentence.
   * `labels` = `{ "type": "single_hop_factoid" }`.

### 7.2 Config

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class SingleHopConfig:
    max_tasks: Optional[int] = None
    max_tasks_per_anchor: int = 1
    min_answer_length: int = 3
    max_answer_length: int = 64
```

### 7.3 Implementation Sketch

```python
from typing import List

from ..core.types import Document, EvaluationSample, Response
from .anchors import Anchor
from .config import LLMClient, SingleHopConfig


def choose_answer_span(anchor: Anchor) -> str:
    # Simple baseline: choose a named entity or noun phrase via heuristic or LLM.
    # v1 can use a cheap heuristic (first capitalized phrase) and be improved later.
    ...


def find_relevant_docs(answer_span: str, chunks: List[Document]) -> List[Document]:
    # Baseline: simple substring match over chunk.text.
    # Optimization: restrict to chunks whose metadata["parent_doc_id"] matches anchor's doc.
    ...


def generate_single_hop_tasks(
    anchors: List[Anchor],
    chunks: List[Document],
    llm: LLMClient,
    cfg: SingleHopConfig,
) -> List[EvaluationSample]:
    samples: List[EvaluationSample] = []

    for anchor in anchors:
        if cfg.max_tasks is not None and len(samples) >= cfg.max_tasks:
            break

        answer_span = choose_answer_span(anchor)
        if not (cfg.min_answer_length <= len(answer_span.split()) <= cfg.max_answer_length):
            continue

        # 1) Generate question
        prompt = build_single_hop_prompt(anchor.sentence, answer_span)
        question = llm.generate(prompt).strip()

        # 2) Find relevant docs
        rel_docs = find_relevant_docs(answer_span, chunks)
        if not rel_docs:
            continue

        # 3) Build reference answer
        ref_answer = Response(text=answer_span)

        sample = EvaluationSample(
            sample_id=str(uuid4()),
            query=question,
            relevant_docs=rel_docs,
            reference_answer=ref_answer,
            labels={"type": "single_hop_factoid"},
            metadata={"source_anchor_doc_id": anchor.doc_id},
        )
        samples.append(sample)

    return samples
```

`build_single_hop_prompt` is a helper that constructs a robust prompt instructing the LLM:

* to produce a single question;
* whose answer is exactly the provided answer span;
* that is answerable from the sentence.

## 8. Multi-Hop Task Generation (`multi_hop.py`)

### 8.1 Graph-Based Entity Linking

We want to create questions that require combining information from 2+ documents.

Implementation strategy:

1. For each chunk, precompute a set of entities (use a simple NER model or an LLM-based tagger; store in `Document.metadata["entities"]`).
2. Build an undirected graph where nodes are chunk IDs and edges connect chunks that share at least one entity.
3. Sample pairs (or small subgraphs) from that graph as candidates for multi-hop tasks.

### 8.2 Config

```python
from dataclasses import dataclass


@dataclass
class MultiHopConfig:
    max_tasks: int = 100
    max_hops: int = 2  # v1 supports 2-hop questions
    min_shared_entities: int = 1
```

### 8.3 Implementation Sketch

```python
from typing import List, Tuple

from ..core.types import Document, EvaluationSample, Response
from .anchors import Anchor
from .config import LLMClient, MultiHopConfig


def build_entity_graph(chunks: List[Document]) -> Dict[str, List[str]]:
    # adjacency list: doc_id -> list of neighboring doc_ids
    ...


def sample_doc_pairs(graph: Dict[str, List[str]], max_pairs: int) -> List[Tuple[str, str]]:
    ...


def generate_multi_hop_tasks(
    chunks: List[Document],
    anchors_by_doc: Dict[str, List[Anchor]],
    llm: LLMClient,
    cfg: MultiHopConfig,
) -> List[EvaluationSample]:
    graph = build_entity_graph(chunks)
    pairs = sample_doc_pairs(graph, cfg.max_tasks * 2)  # oversample, we may drop some

    samples: List[EvaluationSample] = []

    for doc_id1, doc_id2 in pairs:
        if len(samples) >= cfg.max_tasks:
            break

        anchors1 = anchors_by_doc.get(doc_id1, [])
        anchors2 = anchors_by_doc.get(doc_id2, [])
        if not anchors1 or not anchors2:
            continue

        a1 = random.choice(anchors1)
        a2 = random.choice(anchors2)

        # Ask LLM: combine facts from a1.sentence and a2.sentence
        prompt = build_multi_hop_prompt(a1.sentence, a2.sentence)
        out = llm.generate(prompt)
        parsed = parse_multi_hop_output(out)  # expects question + answer
        if parsed is None:
            continue

        question, answer_text = parsed

        # Build sample
        relevant_docs = [
            _lookup_chunk(chunks, doc_id1),
            _lookup_chunk(chunks, doc_id2),
        ]

        sample = EvaluationSample(
            sample_id=str(uuid4()),
            query=question,
            relevant_docs=relevant_docs,
            reference_answer=Response(text=answer_text),
            labels={"type": "multi_hop", "hops": 2},
        )
        samples.append(sample)

    return samples
```

`build_multi_hop_prompt` instructs the LLM to:

* read two facts;
* construct a question whose answer requires combining them;
* output both question and answer in a structured format.

## 9. Robustness Task Generation (`robustness.py`)

We want to generate variants of base tasks to evaluate robustness targets like:

* `NOISE_ROBUSTNESS`
* `NEGATIVE_REJECTION`

### 9.1 Paraphrase / Noise Variants

Given a list of base `EvaluationSample`s, create paraphrased or noisy variants.

Config:

```python
from dataclasses import dataclass


@dataclass
class RobustnessConfig:
    num_paraphrases: int = 3
    add_typos: bool = True
    typo_rate: float = 0.05
```

Implementation sketch:

```python
from typing import List

from ..core.types import EvaluationSample
from .config import LLMClient, RobustnessConfig


def generate_paraphrase_variants(
    base_samples: List[EvaluationSample],
    llm: LLMClient,
    cfg: RobustnessConfig,
) -> List[EvaluationSample]:
    variants: List[EvaluationSample] = []

    for base in base_samples:
        for i in range(cfg.num_paraphrases):
            prompt = build_paraphrase_prompt(base.query)
            paraphrased = llm.generate(prompt).strip()
            noisy = add_typos(paraphrased, cfg.typo_rate) if cfg.add_typos else paraphrased

            s = EvaluationSample(
                sample_id=str(uuid4()),
                query=noisy,
                relevant_docs=base.relevant_docs,
                candidate_docs=base.candidate_docs,
                reference_answer=base.reference_answer,
                labels={
                    **base.labels,
                    "scenario": "paraphrase",
                    "variant_of": base.sample_id,
                },
                metadata={**base.metadata},
            )
            variants.append(s)

    return variants
```

### 9.2 Unanswerable / Negative Queries

To support `NEGATIVE_REJECTION` targets, we generate questions for which the corpus does **not** contain an answer.

High-level algorithm:

1. Sample entities from the corpus (from metadata or simple heuristics).
2. Perturb them (e.g., change year, location, or name).
3. Ask LLM to create plausible questions about the perturbed facts.
4. Ensure there is no exact or near match for the perturbed fact in the corpus.
5. Create `EvaluationSample` with:

   * `relevant_docs = []` or a small set of docs that might mislead.
   * `reference_answer` = Response(text="There is not enough information in the corpus to answer this question.").
   * `labels["scenario"] = "unanswerable"`.

Implementation details can be encapsulated inside `generate_unanswerable_tasks`.

## 10. Quality Filters (`filters.py`)

We need filters to discard low-quality tasks.

Define a simple interface:

```python
from typing import Protocol

from ..core.types import EvaluationSample


class SampleFilter(Protocol):
    def keep(self, sample: EvaluationSample) -> bool:
        ...
```

Example filters:

1. **Answer presence filter**:

   * For all `relevant_docs`, check if `reference_answer.text` appears as substring in at least one doc.
2. **Min/Max question length** filter.
3. **LLM consistency filter** (optional, can be added later):

   * Ask LLM to answer the question from the provided `relevant_docs` only and score its similarity to `reference_answer`.

`apply_filters(samples: List[EvaluationSample], filters: List[SampleFilter]) -> List[EvaluationSample]` should be implemented to run all filters and keep only the samples where all filters return `True`.

## 11. Orchestrator: `TaskGenerator` (`generator.py`)

### 11.1 Config

Create a main configuration object that ties together sub-configs.

```python
from dataclasses import dataclass
from typing import Optional

from .config import HFCorpusConfig, SingleHopConfig, MultiHopConfig, RobustnessConfig
from .preprocess import ChunkingConfig, EmbeddingConfig


@dataclass
class TaskGeneratorConfig:
    # Corpus source
    hf_corpus: Optional[HFCorpusConfig] = None

    # Preprocess
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: Optional[EmbeddingConfig] = None  # optional if not using embedding-based features

    # Task types
    single_hop: Optional[SingleHopConfig] = SingleHopConfig()
    multi_hop: Optional[MultiHopConfig] = None
    robustness: Optional[RobustnessConfig] = None
```

### 11.2 Class Definition

```python
from typing import List, Optional

from ..core.types import Document, EvaluationSample
from ..core.dataset import InMemoryDataset

from .config import TaskGeneratorConfig, LLMClient
from .hf_adapter import HFCorpusAdapter
from .preprocess import chunk_documents, embed_chunks
from .anchors import extract_anchors
from .single_hop import generate_single_hop_tasks
from .multi_hop import generate_multi_hop_tasks
from .robustness import generate_paraphrase_variants, generate_unanswerable_tasks
from .filters import apply_filters, SampleFilter


class TaskGenerator:
    def __init__(
        self,
        llm: LLMClient,
        config: TaskGeneratorConfig,
        sample_filters: Optional[List[SampleFilter]] = None,
    ):
        self.llm = llm
        self.config = config
        self.sample_filters = sample_filters or []

    def build_corpus(self) -> List[Document]:
        if self.config.hf_corpus is not None:
            adapter = HFCorpusAdapter(self.config.hf_corpus)
            return adapter.to_documents()
        else:
            raise ValueError("TaskGenerator requires an HFCorpusConfig or another corpus source.")

    def generate_samples(self) -> List[EvaluationSample]:
        # 1) Build corpus
        docs = self.build_corpus()

        # 2) Preprocess: chunk + embeddings
        chunks = chunk_documents(docs, self.config.chunking)
        if self.config.embedding is not None:
            embeddings = embed_chunks(chunks, self.config.embedding)
            # embeddings can later be used for hard-negative mining, etc.

        # 3) Anchors
        anchors = extract_anchors(chunks)

        # 4) Single-hop tasks
        all_samples: List[EvaluationSample] = []
        if self.config.single_hop is not None:
            single_hop_samples = generate_single_hop_tasks(
                anchors=anchors,
                chunks=chunks,
                llm=self.llm,
                cfg=self.config.single_hop,
            )
            all_samples.extend(single_hop_samples)

        # 5) Multi-hop tasks (optional)
        if self.config.multi_hop is not None:
            anchors_by_doc = group_anchors_by_doc(anchors)
            multi_hop_samples = generate_multi_hop_tasks(
                chunks=chunks,
                anchors_by_doc=anchors_by_doc,
                llm=self.llm,
                cfg=self.config.multi_hop,
            )
            all_samples.extend(multi_hop_samples)

        # 6) Robustness variants (optional)
        if self.config.robustness is not None:
            paraphrases = generate_paraphrase_variants(
                base_samples=all_samples,
                llm=self.llm,
                cfg=self.config.robustness,
            )
            all_samples.extend(paraphrases)

            negatives = generate_unanswerable_tasks(
                chunks=chunks,
                llm=self.llm,
                cfg=self.config.robustness,
            )
            all_samples.extend(negatives)

        # 7) Filters
        if self.sample_filters:
            all_samples = apply_filters(all_samples, self.sample_filters)

        return all_samples

    def generate_dataset(self, name: str = "generated") -> InMemoryDataset:
        samples = self.generate_samples()
        return InMemoryDataset(name=name, samples=samples)
```

`group_anchors_by_doc` is a small helper mapping `doc_id -> List[Anchor]`.

## 12. Example Usage with a Hugging Face Dataset

This example shows how an engineer wires everything up from HF dataset to Auepora evaluation.

```python
from datasets import load_dataset

from auepora_eval.core.runner import AueporaEvaluator
from auepora_eval.core.plan import EvaluationPlan
from auepora_eval.metrics.retrieval_basic import RecallAtK
from auepora_eval.metrics.generation_text import RougeLAnswer

from auepora_eval.taskgen.config import HFCorpusConfig, TaskGeneratorConfig, ChunkingConfig, SingleHopConfig
from auepora_eval.taskgen.generator import TaskGenerator


class SimpleLLM:
    def generate(self, prompt: str) -> str:
        # wrap your preferred LLM API here
        ...


# 1) Configure HF corpus
hf_cfg = HFCorpusConfig(
    dataset_name="wikipedia",
    dataset_config_name="20220301.en",
    split="train",
    text_column="text",
    id_column="id",
)

# 2) Configure task generation
config = TaskGeneratorConfig(
    hf_corpus=hf_cfg,
    chunking=ChunkingConfig(max_tokens=256, overlap_tokens=32),
    single_hop=SingleHopConfig(max_tasks=1000),
)

llm = SimpleLLM()

# 3) Generate dataset
generator = TaskGenerator(llm=llm, config=config)

eval_dataset = generator.generate_dataset(name="wikipedia_single_hop")

# 4) Build evaluation plan for Auepora
plan = EvaluationPlan(metrics=[RecallAtK(k=5), RougeLAnswer()])

# 5) Plug in a RAG system and evaluate
rag_system = MyRAGSystem(retriever, llm_for_rag)

evaluator = AueporaEvaluator(system=rag_system, plan=plan, top_k=5)
results = evaluator.evaluate(eval_dataset)

for r in results:
    print(r.name, r.value, r.details)
```

## 13. Engineering Tasks and Milestones

**Milestone 1 – HF Adapter and Preprocessing**

* [ ] Implement `HFCorpusConfig` and `HFCorpusAdapter`.
* [ ] Implement `ChunkingConfig`, `EmbeddingConfig`, `chunk_documents`, `embed_chunks`.
* [ ] Implement `Anchor` and `extract_anchors`.

**Milestone 2 – Single-Hop Task Generation**

* [ ] Implement `SingleHopConfig`.
* [ ] Implement `choose_answer_span`, `find_relevant_docs`, `build_single_hop_prompt`.
* [ ] Implement `generate_single_hop_tasks`.

**Milestone 3 – Multi-Hop and Robustness**

* [ ] Implement entity extraction and `build_entity_graph`.
* [ ] Implement `MultiHopConfig`, `generate_multi_hop_tasks`.
* [ ] Implement `RobustnessConfig`, paraphrase and negative task generators.

**Milestone 4 – Filters and Orchestrator**

* [ ] Implement `SampleFilter` interface and basic filters.
* [ ] Implement `TaskGeneratorConfig` and `TaskGenerator`.
* [ ] Add examples using a Hugging Face dataset in `examples/`.

This spec should give the engineer a concrete plan for implementing a **task generation pipeline** that takes any Hugging Face dataset as the corpus, transforms it into Auepora-compatible `EvaluationSample`s, and outputs datasets ready to be evaluated by the main Auepora evaluation library.
