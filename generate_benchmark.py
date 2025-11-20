#!/usr/bin/env python3
"""
Main script to generate the Epstein Bench TRACe benchmark.
This script coordinates the entire benchmark generation pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent / "scripts"))

from scripts.data_loader import EpsteinDataLoader
from scripts.document_selector import InterestingDocumentSelector
from scripts.trace_generator import TraceBenchmarkGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate Epstein Bench TRACe Benchmark")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of TRACe questions to generate (default: 100)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force re-download of the dataset even if cached",
    )
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="epstein_trace_v1",
        help="Name for the benchmark output file",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        default=True,
        help="Use OpenAI to generate high-quality answers (default: True)",
    )
    parser.add_argument(
        "--no-openai",
        dest="use_openai",
        action="store_false",
        help="Disable OpenAI and use extractive answers instead",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data",
        help="Directory for dataset + tagging caches (default: ./data)",
    )
    parser.add_argument(
        "--enable-ner-selection",
        dest="use_ner_selection",
        action="store_true",
        default=True,
        help="Use the tagging/NER pipeline to preselect interesting documents (default: enabled)",
    )
    parser.add_argument(
        "--disable-ner-selection",
        dest="use_ner_selection",
        action="store_false",
        help="Skip the tagging/NER preselection phase entirely",
    )
    parser.add_argument(
        "--ner-top-k",
        type=int,
        default=2500,
        help="Maximum number of documents to keep after NER filtering (default: 2500)",
    )
    parser.add_argument(
        "--ner-min-score",
        type=float,
        default=2.5,
        help="Minimum interestingness score required to keep a document (default: 2.5)",
    )
    parser.add_argument(
        "--ner-candidate-pool",
        type=int,
        default=6000,
        help="Only the top-N longest documents are sent through NER selection (default: 6000)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of documents processed for testing (applies to NER selection)",
    )
    parser.add_argument(
        "--ner-max-chars",
        type=int,
        default=4000,
        help="Maximum characters per document sent to the tagging backend (default: 4000)",
    )
    parser.add_argument(
        "--ner-min-confidence",
        type=float,
        default=0.55,
        help="Minimum entity confidence to count toward the interestingness score (default: 0.55)",
    )
    parser.add_argument(
        "--ner-min-entities",
        type=int,
        default=3,
        help="Minimum number of high-confidence entities required before scoring a document (default: 3)",
    )
    parser.add_argument(
        "--generation-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use for answer generation (default: gpt-4o)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EPSTEIN BENCH - TRACe Benchmark Generator")
    print("=" * 60)

    # Step 1: Load and process the dataset
    print("\n[Step 1/3] Loading Epstein Files dataset...")
    loader = EpsteinDataLoader(cache_dir=args.cache_dir)
    loader.load_dataset(force_reload=args.force_reload)
    documents = loader.process_documents()
    loader.save_processed_data()

    selected_documents = documents
    doc_annotations = {}
    selection_metadata = {
        "ner_selection_enabled": False,
        "retained_documents": len(documents),
        "total_documents": len(documents),
    }

    if args.use_ner_selection:
        print("\n[Step 1b/3] Ranking documents with tagging/NER pipeline...")
        
        candidate_pool_size = args.ner_candidate_pool
        if args.limit:
             candidate_pool_size = min(candidate_pool_size, args.limit)
             print(f"Testing mode: Limiting candidate pool to {candidate_pool_size} documents")

        selector = InterestingDocumentSelector(
            cache_dir=args.cache_dir,
            max_chars=args.ner_max_chars,
            min_confidence=args.ner_min_confidence,
            min_entities=args.ner_min_entities,
        )
        selected_documents = selector.select_documents(
            documents,
            top_k=args.ner_top_k,
            min_score=args.ner_min_score,
            candidate_pool_size=candidate_pool_size,
        )
        doc_annotations = selector.annotations
        selection_metadata = selector.selection_metadata

        if not selected_documents:
            print("⚠️  NER selection yielded zero documents. Falling back to full corpus.")
            selected_documents = documents
    else:
        print("\n[Step 1b/3] NER-based document selection disabled.")

    # Step 2: Generate TRACe questions
    print(f"\n[Step 2/3] Generating {args.num_questions} TRACe questions...")
    if args.use_openai:
        import os

        if not os.getenv("OPENAI_API_KEY"):
            print("\n⚠️  Warning: OPENAI_API_KEY environment variable not set!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            print("Cannot generate reasoning questions without OpenAI.\n")
            return

    generator = TraceBenchmarkGenerator(
        selected_documents,
        output_dir="./benchmarks",
        use_openai=args.use_openai,
        model_name=args.generation_model,
    )
    questions = generator.generate_trace_questions(
        num_questions=args.num_questions,
        context_length=1500
    )

    # Step 3: Save benchmark
    print(f"\n[Step 3/3] Saving benchmark as '{args.benchmark_name}'...")
    generator.save_benchmark(name=args.benchmark_name)
    # stats_df = generator.generate_statistics() # Statistics removed from TRACe generator for now

    print("\n" + "=" * 60)
    print("✅ Benchmark generation complete!")
    print("=" * 60)
    print(f"\nBenchmark saved to: ./benchmarks/{args.benchmark_name}.json")
    print(f"Total questions generated: {len(questions)}")
    if selection_metadata.get("ner_selection_enabled"):
        print(
            f"Documents used for question generation: "
            f"{selection_metadata.get('retained_documents')} "
            f"(from {selection_metadata.get('total_documents')} total)"
        )
    print("\nNext steps:")
    print("1. Review the generated questions in the benchmark file")
    print("2. Implement your RAG system with the required interface")
    print("3. Run evaluation using trace_evaluator.py")
    print("\nExample evaluation code:")
    print("  from scripts.trace_evaluator import TraceEvaluator")
    print(f"  evaluator = TraceEvaluator('./benchmarks/{args.benchmark_name}.json')")
    print("  metrics = evaluator.evaluate_rag_system(your_rag_system)")


if __name__ == "__main__":
    main()
