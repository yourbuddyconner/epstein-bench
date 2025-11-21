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
  - [Documentation](#documentation)
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

Just as the Enron corpus became the gold standard for email and network analysis, Epstein Bench aims to be the stress-test for "in-the-wild" RAG systems. It forces systems to grapple with the messiness of real legal discovery rather than perfectly formatted Wikipedia articles.

---

## Installation

This project uses Anaconda for dependency management.

1.  **Clone and Setup:**
    ```bash
    git clone https://github.com/yourusername/epstein-bench.git
    cd epstein-bench
    conda create -n epstein-bench python=3.11
    conda activate epstein-bench
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    ```bash
    cp env.example .env
    # Edit .env and add OPENAI_API_KEY, etc.
    ```

---

## Quick Start

To see the framework in action, run the baseline RAG example:

```bash
python -m src.auepora_eval.examples.baseline_rag --dataset epstein_bench_tasks.jsonl --top_k 5 --use_llm
```

*Note: If you don't have `epstein_bench_tasks.jsonl`, see the [Usage Guide](docs/usage.md) to generate one.*

---

## Core Concepts

An **Evaluation Plan** binds Targets, Datasets, and Metrics together.

| Target Category | Description | Common Metrics |
|----------------|-------------|----------------|
| **Retrieval Relevance** | Did we find the right docs? | Recall@K, Precision@K |
| **Retrieval Accuracy** | Is the ranking correct? | MRR, NDCG |
| **Gen Correctness** | Is the answer right? | ROUGE, BLEU, BERTScore |
| **Gen Faithfulness** | Is it grounded in context? | LLM Judge (Faithfulness) |
| **Robustness** | Can it handle noise? | Performance degradation metrics |

---

## Documentation

For detailed instructions on how to use the library, implement your own RAG system, or generate synthetic data, please refer to the **[Usage & Implementation Guide](docs/usage.md)**.

Key documentation files:
- [Usage & Implementation Guide](docs/usage.md)
- [Engineering Spec](docs/spec.md)
- [Metrics Definitions](docs/metrics.md)
- [Task Generation Logic](docs/task-generation.md)

---

## Project Structure

```text
src/
├── auepora_eval/           # Core Evaluation Library
│   ├── core/               # Types, Interfaces, Runner
│   ├── metrics/            # Metric implementations
│   ├── taskgen/            # Synthetic data generation
│   ├── io/                 # Data loading/saving
│   └── examples/           # Example scripts
└── epstein_bench/          # Project-specific configurations
    ├── plan.py             # Metric plan builders
    └── llm_critic.py       # LLM Judge implementations
```

---

## Citation

If you use Epstein Bench or the Auepora-based evaluation components, please cite:

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
