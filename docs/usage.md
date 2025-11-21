# Usage & Implementation Guide

This guide covers how to generate evaluation data and implement your own RAG system using the Auepora framework.

## The RAG Interface

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
save_jsonl_dataset(dataset, "epstein_bench_tasks.jsonl")
```

