from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable

from ..auepora_eval.core.rag_interface import RAGSystem
from ..auepora_eval.core.runner import AueporaEvaluator
from ..auepora_eval.io.jsonl import load_jsonl_dataset
from ..config import config
from .plan import build_metric_plan
from .llm_critic import OpenAIChatCritic

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("benchmark_runner")


def _load_rag_system(factory_path: str) -> RAGSystem:
    if ":" not in factory_path:
        raise ValueError("RAG factory must be specified as 'module_path:callable'.")
    module_name, attr_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)
    factory: Callable[[], RAGSystem] = getattr(module, attr_name)
    system = factory()
    if not isinstance(system, RAGSystem) and not hasattr(system, "run"):
        raise TypeError("Factory must return an object implementing RAGSystem.run(sample, top_k=...).")
    return system


def main():
    parser = argparse.ArgumentParser(description="Generic Benchmark Runner for Auepora RAG systems")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset (see docs/spec.md)")
    parser.add_argument("--rag", required=True, help="Import path to RAG factory, e.g. 'mypkg.my_rag:build_rag_system'")
    parser.add_argument("--top_k", type=int, default=5, help="Retriever depth for metrics & system")
    parser.add_argument("--no-bertscore", action="store_false", dest="enable_bertscore", help="Disable BERTScore metrics")
    parser.add_argument("--local-bertscore", action="store_false", dest="use_openai_bertscore", help="Use local BERT model for BERTScore instead of OpenAI (slower, no API cost)")
    parser.add_argument("--no-llm-metrics", action="store_false", dest="enable_llm_metrics", help="Disable LLM-judged metrics")
    parser.add_argument("--output-dir", default="docs/benchmark_data", help="Directory to save benchmark results JSON")
    parser.add_argument("--system-name", help="Name of the system for the leaderboard (defaults to rag factory path)")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of parallel workers for RAG system evaluation")
    parser.set_defaults(enable_bertscore=True, enable_llm_metrics=True, use_openai_bertscore=True)
    args = parser.parse_args()

    logger.info("Loading dataset from %s", args.dataset)
    dataset = load_jsonl_dataset(args.dataset)
    logger.info("Loaded %d samples.", len(dataset))

    logger.info("Loading RAG system from %s", args.rag)
    rag_system = _load_rag_system(args.rag)

    llm_metric_critic = None
    if args.enable_llm_metrics:
        if not config.openai_api_key:
            raise RuntimeError("LLM metrics requested but OPENAI_API_KEY is not set.")
        llm_metric_critic = OpenAIChatCritic()

    plan = build_metric_plan(
        top_k=args.top_k,
        enable_bertscore=args.enable_bertscore,
        use_openai_bertscore=args.use_openai_bertscore,
        enable_llm_metrics=args.enable_llm_metrics,
        llm_critic=llm_metric_critic,
    )

    evaluator = AueporaEvaluator(system=rag_system, plan=plan, top_k=args.top_k, max_workers=args.concurrency)
    results = evaluator.evaluate(dataset)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    for result in results:
        print(f"{result.name:<20} | {result.value:.4f} | Details: {result.details}")
    print("=" * 40 + "\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    system_name = args.system_name or args.rag
    timestamp = datetime.now(timezone.utc).isoformat()

    # Simple flattening for JSON
    metrics_dict = {}
    for result in results:
        metrics_dict[result.name] = {
            "value": result.value,
            "target": result.target.name,
            "details": result.details
        }

    output_data = {
        "system_name": system_name,
        "timestamp": timestamp,
        "dataset": args.dataset,
        "metrics": metrics_dict,
        "config": {
            "top_k": args.top_k,
            "bertscore": args.enable_bertscore,
            "llm_metrics": args.enable_llm_metrics
        }
    }

    # Create filename based on system name safe string and timestamp
    safe_name = "".join(c if c.isalnum() else "_" for c in system_name)
    filename = f"{safe_name}_{int(datetime.now().timestamp())}.json"
    filepath = os.path.join(args.output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved benchmark results to {filepath}")


if __name__ == "__main__":
    main()
