# Epstein Bench: The Files Don’t Lie, But Your RAG Might

<p align="center">
  <a href="https://github.com/tensonaut/EPSTEIN_FILES_20K"><img src="https://img.shields.io/badge/Epstein%20Files-Public-red?style=flat-square" alt="Epstein Files"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.11+-yellow?style=flat-square" alt="Python">
  <a href="https://arxiv.org/abs/2405.07437"><img src="https://img.shields.io/badge/RAG-Certified-green?style=flat-square" alt="RAG Certified"></a>
  <img src="https://img.shields.io/badge/funded%20by%20yc-not%20yet-orange?style=flat-square" alt="YC Status">
</p>

<p align="center">
  <img src="mascot.png" alt="Epstein Bench Mascot" width="220" />
</p>

Epstein Bench is a RAG benchmark built upon the **Epstein Files**—a collection of publicly released documents regarding the Jeffrey Epstein associates. Similar to how the **Enron Email Dataset** became a standard for network analysis and NLP in the early 2000s, this corpus provides a highly complex, noisy, and entity-rich environment for stress-testing modern Retrieval-Augmented Generation systems.

This project implements the **Auepora Evaluation Framework** ([arXiv:2405.07437](https://arxiv.org/abs/2405.07437)), a rigorous methodology for decomposing RAG performance into specific targets, datasets, and metrics.

---

## Table of Contents

- [Epstein Bench: The Files Don’t Lie, But Your RAG Might](#epstein-bench-the-files-dont-lie-but-your-rag-might)
  - [Table of Contents](#table-of-contents)
  - [Philosophy \& Methodology](#philosophy--methodology)
    - [1. The Framework: Auepora](#1-the-framework-auepora)
    - [2. The Benchmark: Epstein Bench](#2-the-benchmark-epstein-bench)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Core Concepts](#core-concepts)
    - [Targets, Datasets, and Metrics](#targets-datasets-and-metrics)
    - [The RAG Interface](#the-rag-interface)
  - [Generating Evaluation Data](#generating-evaluation-data)
  - [Implementing Your Own RAG System](#implementing-your-own-rag-system)
  - [Project Structure](#project-structure)
  - [Citation](#citation)

---

## Philosophy & Methodology

This project distinguishes between the **evaluation framework** (The Science) and the **benchmark data** (The Challenge).

### 1. The Framework: Auepora
Based on *"Evaluation of Retrieval-Augmented Generation: A Survey"* by Yu et al. (2024), the Auepora framework provides the taxonomy for our evaluation. It decouples RAG performance into:
- **Target Space:** Explicit definitions of *what* we are measuring (e.g., `RETRIEVAL_RELEVANCE`, `GENERATION_FAITHFULNESS`).
- **Dataset Space:** The required annotations (e.g., ground-truth documents vs. reference answers).
- **Metric Space:** The mathematical scoring functions (e.g., Recall@K, ROUGE-L, LLM Judges).

### 2. The Benchmark: Epstein Bench
While the framework can be applied to any data, **Epstein Bench** specifically targets the **Epstein Files**. This corpus was chosen because it mirrors real-world enterprise challenges better than sanitized academic datasets (like SQuAD or MS MARCO):
- **Extreme Noise:** Contains scanned PDFs, handwritten notes, legal depositions, and messy OCR.
- **Complex Graph:** Requires multi-hop reasoning across thousands of entities (people, organizations, locations) with hidden connections.
- **Needle-in-a-Haystack:** Relevant information is often buried in hundreds of pages of irrelevant legalese.

Just as the Enron corpus became the gold standard for email and network analysis, Epstein Bench aims to be the stress-test for "in-the-wild" RAG systems.

---

## Installation

This project uses Anaconda for dependency management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/epstein-bench.git
    cd epstein-bench
    ```

2.  **Create and activate the environment:**
    ```bash
    conda create -n epstein-bench python=3.11
    conda activate epstein-bench
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Copy the example environment file and add your API keys (needed for LLM-based metrics and generation).
    ```bash
    cp env.example .env
    # Edit .env and add OPENAI_API_KEY, etc.
    ```

---

## Quick Start

To see the framework in action, you can run the baseline RAG example. This script creates a simple TF-IDF retriever and evaluates it against a sample dataset.

```bash
# Run the baseline evaluation example
python -m src.auepora_eval.examples.baseline_rag --dataset my_bench.jsonl --top_k 5 --use_llm
```

*Note: If you don't have `my_bench.jsonl`, you can generate one using the task generator (see below).*

---

## Core Concepts

### Targets, Datasets, and Metrics

An **Evaluation Plan** binds these three concepts together.

| Target Category | Description | Common Metrics | Required Data |
|----------------|-------------|----------------|---------------|
| **Retrieval Relevance** | Did we find the right docs? | Recall@K, Precision@K | `relevant_docs` |
| **Retrieval Accuracy** | Is the ranking correct? | MRR, NDCG | `relevant_docs` (ordered) |
| **Gen Correctness** | Is the answer right? | ROUGE, BLEU, BERTScore | `reference_answer` |
| **Gen Faithfulness** | Is it grounded in context? | LLM Judge (Faithfulness) | `relevant_docs` + `response` |
| **Robustness** | Can it handle noise? | Performance degradation on noisy queries | `labels["variant_of"]` |

### The RAG Interface

To evaluate your system, you simply wrap it in the `RAGSystem` interface.

```python
from src.auepora_eval.core.rag_interface import RAGSystem, SystemOutputs
from src.auepora_eval.core.types import EvaluationSample, Response

class MyRAG(RAGSystem):
    def run(self, sample: EvaluationSample, *, top_k: int = 5) -> SystemOutputs:
        # 1. Your retrieval logic
        docs = my_retriever.search(sample.query, k=top_k)
        
        # 2. Your generation logic
        answer = my_llm.generate(sample.query, context=docs)
        
        # 3. Return standard output
        return SystemOutputs(
            retrieved=docs, 
            response=Response(text=answer)
        )
```

---

## Generating Evaluation Data

One of the hardest parts of RAG evaluation is getting good test data. Epstein Bench includes a **Task Generator** that creates synthetic evaluation samples from any Hugging Face dataset.

It generates:
- **Single-hop questions:** Factoid questions answerable from a single chunk.
- **Multi-hop questions:** Complex questions requiring synthesis of multiple chunks.
- **Robustness variants:** Paraphrased questions, typos, and unanswerable queries.

**Example Usage:**

```python
# See src/auepora_eval/examples/generate_tasks.py for full code
from src.auepora_eval.taskgen.generator import TaskGenerator, TaskGeneratorConfig

# Configure to use Wikipedia
config = TaskGeneratorConfig(
    hf_corpus=HFCorpusConfig(dataset_name="wikipedia", split="train"),
    single_hop=SingleHopConfig(max_tasks=50)
)

# Run generator
generator = TaskGenerator(llm=my_llm_client, config=config)
dataset = generator.generate_dataset("my_wiki_bench")

# Save to disk
save_jsonl_dataset(dataset, "my_bench.jsonl")
```

---

## Implementing Your Own RAG System

To evaluate your own custom RAG pipeline (e.g., using LangChain, LlamaIndex, or custom code):

1.  **Inherit from `RAGSystem`**: Create a class that implements the `run` method.
2.  **Map Inputs**: The `run` method receives an `EvaluationSample` containing the query.
3.  **Map Outputs**: Return a `SystemOutputs` object containing your retrieved documents (as `RetrievedDocument` objects) and the final generated text (as a `Response` object).
4.  **Run Evaluator**: Pass your system instance to `AueporaEvaluator`.

**Code Skeleton:**

```python
from src.auepora_eval.core.types import RetrievedDocument, Document
from src.auepora_eval.core.runner import AueporaEvaluator
from src.epstein_bench.plan import build_metric_plan

# ... implementation of YourCustomRAG ...

# Define what you want to measure
plan = build_metric_plan(
    top_k=5, 
    enable_bertscore=True, 
    enable_llm_metrics=True
)

# Run
evaluator = AueporaEvaluator(system=YourCustomRAG(), plan=plan)
results = evaluator.evaluate(dataset)
```

---

## Project Structure

```text
src/
├── auepora_eval/           # Core Evaluation Library
│   ├── core/               # Types, Interfaces, Runner
│   ├── metrics/            # Metric implementations (Recall, ROUGE, etc.)
│   ├── taskgen/            # Synthetic data generation pipeline
│   ├── io/                 # Data loading/saving
│   └── examples/           # Example scripts
└── epstein_bench/          # Project-specific configurations & runners
    ├── plan.py             # Metric plan builders
    └── llm_critic.py       # LLM Judge implementations
```

**Key Files:**
- `docs/spec.md`: Detailed engineering specification of the library.
- `docs/metrics.md`: Mathematical definitions of all supported metrics.
- `docs/task-generation.md`: Logic behind the synthetic data generator.

---

## Citation

If you use Epstein Bench or the Auepora-based evaluation components in your research, please cite:

- The Auepora evaluation framework survey: [arXiv:2405.07437](https://arxiv.org/abs/2405.07437)

```bibtex
@article{yu2024evaluation,
  title   = {Evaluation of Retrieval-Augmented Generation: A Survey},
  author  = {Hao Yu and Aoran Gan and Kai Zhang and Shiwei Tong and Qi Liu and Zhaofeng Liu},
  journal = {arXiv preprint arXiv:2405.07437},
  year    = {2024},
  url     = {https://arxiv.org/abs/2405.07437}
}
```

- This repository:

```bibtex
@software{epstein_bench_2025,
  title  = {Epstein Bench: The Files Don’t Lie, But Your RAG Might},
  author = {Conner Swann},
  year   = {2025},
  url    = {https://github.com/yourbuddyconner/epstein-bench}
}
```