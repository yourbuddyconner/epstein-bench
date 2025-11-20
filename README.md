<div align="center">
  <img src="mascot.png" alt="Epstein Bench Mascot" width="300">
</div>

# Epstein Bench

A specialized RAG benchmark for legal and financial discovery, focusing on the Jeffrey Epstein document release.
It implements the **TRACe** evaluation framework (Reasoning, Adherence, Completeness, Relevance).

## Overview

This project provides tools to:
1. **Generate** high-quality, reasoning-heavy RAG questions from raw documents (`scripts/trace_generator.py`)
2. **Evaluate** RAG systems using LLM-as-a-judge metrics (`scripts/trace_evaluator.py`)
3. **Process** raw PDF/text documents into a clean dataset

## TRACe Benchmark Methodology

Unlike simple "Needle-in-a-Haystack" benchmarks, this dataset focuses on:
*   **Reasoning:** Questions require synthesizing information, not just keyword lookup.
*   **Adherence:** Evaluates if the model's answer is grounded in the retrieved context (hallucination check).
*   **Hard Negatives:** Includes distractors that are semantically similar but irrelevant.
*   **OCR Filtering:** Automatically filters out low-quality/garbled text segments during generation.

## Setup

1.  Create environment:
```bash
    conda create -n epstein-bench python=3.11
conda activate epstein-bench
pip install -r requirements.txt
```

2.  Set up OpenAI API key (for generation/evaluation):
```bash
    cp env.example .env
    # Edit .env with your OPENAI_API_KEY
```

## Usage

### 1. Generate Benchmark
Create a new TRACe benchmark from processed documents:
```bash
python scripts/trace_generator.py
```
This creates `benchmarks/epstein_trace_v1.json`.

### 2. Evaluate a RAG System
Run the evaluator on your RAG pipeline:
```bash
python scripts/trace_evaluator.py
```
(See `scripts/trace_evaluator.py` for how to plug in your own `retrieve_and_answer` function).

## File Structure

- `benchmarks/`: Generated benchmark JSON files
- `data/`: Processed document storage
- `scripts/`:
    - `trace_generator.py`: Generates questions/answers/distractors
    - `trace_evaluator.py`: LLM-based evaluation script
    - `document_selector.py`: Selects diverse documents for benchmarking
- `src/`: Core utilities

## License

MIT
