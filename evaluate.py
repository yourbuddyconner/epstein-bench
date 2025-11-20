#!/usr/bin/env python3
"""
Simple evaluation script for testing RAG systems on Epstein Bench.
This is the main entry point for users to evaluate their systems.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from nih_evaluator import NIHEvaluator
from baseline_rag import BaselineTFIDFRAG, EmbeddingRAG


def evaluate_baseline(benchmark_path: str = "./benchmarks/epstein_nih_v1.json"):
    """Run evaluation with the baseline TF-IDF RAG system."""
    
    # Check if benchmark exists
    if not Path(benchmark_path).exists():
        print(f"Error: Benchmark file not found at {benchmark_path}")
        print("Please download the canonical benchmark from the repository.")
        return
    
    # Load sample documents for baseline (in production, these would be pre-indexed)
    docs_file = "./benchmarks/epstein_documents_sample.json"
    if not Path(docs_file).exists():
        print(f"Error: Sample documents not found at {docs_file}")
        print("Please download the sample documents from the repository.")
        return
    
    print("="*60)
    print("EPSTEIN BENCH - Baseline Evaluation")
    print("="*60)
    
    # Initialize baseline RAG
    print("\nInitializing TF-IDF baseline RAG system...")
    baseline_rag = BaselineTFIDFRAG(docs_file, top_k=10)
    
    # Run evaluation
    evaluator = NIHEvaluator(benchmark_path)
    metrics = evaluator.evaluate_rag_system(baseline_rag, "tfidf_baseline", verbose=False)
    
    return metrics


def evaluate_custom(rag_system, system_name: str, 
                   benchmark_path: str = "./benchmarks/epstein_nih_v1.json"):
    """
    Evaluate a custom RAG system.
    
    Args:
        rag_system: Your RAG system implementing retrieve_and_answer(question) method
        system_name: Name for your system
        benchmark_path: Path to the benchmark file
    """
    
    # Check if benchmark exists
    if not Path(benchmark_path).exists():
        print(f"Error: Benchmark file not found at {benchmark_path}")
        print("Please download the canonical benchmark from the repository.")
        return
    
    print("="*60)
    print(f"EPSTEIN BENCH - Evaluating {system_name}")
    print("="*60)
    
    # Run evaluation
    evaluator = NIHEvaluator(benchmark_path)
    metrics = evaluator.evaluate_rag_system(rag_system, system_name, verbose=False)
    
    return metrics


class ExampleRAGSystem:
    """
    Example template for implementing your own RAG system.
    Your system must implement the retrieve_and_answer method.
    """
    
    def __init__(self):
        # Initialize your retrieval index, models, etc.
        pass
    
    def retrieve_and_answer(self, question: str) -> Tuple[str, List[str], Dict]:
        """
        Retrieve relevant documents and generate an answer.
        
        Args:
            question: The NIH question to answer
            
        Returns:
            - answer: The generated answer string
            - retrieved_doc_ids: List of retrieved document IDs (e.g., ["epstein_000001", ...])
            - timing: Dict with 'retrieval_ms' and 'generation_ms' keys
        """
        # Your implementation here
        answer = "Your extracted/generated answer here"
        retrieved_docs = ["epstein_000001", "epstein_000002"]  # Your retrieved doc IDs
        timing = {
            "retrieval_ms": 100.0,  # Time spent on retrieval in milliseconds
            "generation_ms": 50.0   # Time spent on answer generation in milliseconds
        }
        return answer, retrieved_docs, timing


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG systems on Epstein Bench")
    parser.add_argument(
        "--system",
        choices=["baseline", "example"],
        default="baseline",
        help="Which system to evaluate"
    )
    parser.add_argument(
        "--benchmark",
        default="./benchmarks/epstein_nih_v1.json",
        help="Path to benchmark file"
    )
    
    args = parser.parse_args()
    
    if args.system == "baseline":
        evaluate_baseline(args.benchmark)
    elif args.system == "example":
        # Example of evaluating a custom system
        example_rag = ExampleRAGSystem()
        evaluate_custom(example_rag, "example_system", args.benchmark)
    
    print("\n✅ Evaluation complete! Check ./results/ for detailed outputs.")
