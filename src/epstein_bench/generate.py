from typing import Dict, Any, Optional
import argparse
import logging
from pathlib import Path

from auepora_eval.taskgen.config import (
    TaskGeneratorConfig, HFCorpusConfig, ChunkingConfig, SingleHopConfig, MultiHopConfig, RobustnessConfig
)
from auepora_eval.taskgen.generator import TaskGenerator
from auepora_eval.taskgen.openai_client import OpenAIClient
from auepora_eval.io.jsonl import save_jsonl_dataset
from config import config

# Placeholder LLM Client for fallback or testing
class MockLLMClient:
    def generate(self, prompt: str) -> str:
        if "multi-hop" in prompt.lower():
            return "Question: What is the connection between Document A and Document B?\nAnswer: They both mention key events."
        if "write a question" in prompt.lower():
            return "What specific details are mentioned in this document?"
        if "paraphrase" in prompt.lower():
            return "Can you list the details found in this document?"
        return "Mock response"

def get_llm_client():
    if config.tagging_backend == "openai" and config.openai_api_key:
        print(f"Initializing OpenAI Client with model {config.tagging_model_name}...")
        return OpenAIClient(
            api_key=config.openai_api_key,
            model=config.tagging_model_name,
            temperature=config.tagging_temperature,
            max_tokens=config.tagging_max_output_tokens
        )
    else:
        print("Using Mock LLM Client (OpenAI key not set or backend not 'openai')...")
        return MockLLMClient()

def get_epstein_config(test_mode: bool = False, limit: Optional[int] = None, doc_limit: Optional[int] = None) -> TaskGeneratorConfig:
    """
    Returns the configuration for generating the Epstein Bench dataset
    using tensonaut/EPSTEIN_FILES_20K.
    
    Args:
        test_mode: If True, reduces dataset size and task counts for quick testing.
        limit: Optional limit on the number of generated tasks (single hop).
        doc_limit: Optional limit on the number of documents to load from the corpus.
    """
    
    # Determine document limit
    # Priority: CLI arg > Test mode default > None (all docs)
    final_doc_limit = doc_limit
    if final_doc_limit is None and test_mode:
        final_doc_limit = 100

    # 1. Configure the HF Corpus
    hf_cfg = HFCorpusConfig(
        dataset_name="tensonaut/EPSTEIN_FILES_20K",
        split="train",
        text_column="text",
        id_column="filename", 
        metadata_fn=lambda row: {"source_filename": row.get("filename", "")},
        limit=final_doc_limit
    )

    # 2. Configure Preprocessing
    chunk_cfg = ChunkingConfig(
        max_tokens=config.tagging_max_output_tokens,
        overlap_tokens=50
    )

    # 3. Configure Task Types
    # We want a mix of simple factoid questions and complex multi-hop reasoning.
    
    if limit is not None:
        sh_max = limit
        mh_max = max(1, int(limit * 0.5))
    else:
        sh_max = 5 if test_mode else 200
        mh_max = 2 if test_mode else 100
    
    single_hop_cfg = SingleHopConfig(
        max_tasks=sh_max,
        min_answer_length=3,
        max_answer_length=100
    )

    multi_hop_cfg = MultiHopConfig(
        max_tasks=mh_max,
        max_hops=2,
        min_shared_entities=1,
        use_llm_entities=False,
    )
    
    robustness_cfg = RobustnessConfig(
        num_paraphrases=1 if test_mode else 2,
        add_typos=True,
        typo_rate=0.02,
        num_unanswerable=2 if test_mode else 25,
    )

    return TaskGeneratorConfig(
        hf_corpus=hf_cfg,
        chunking=chunk_cfg,
        single_hop=single_hop_cfg,
        multi_hop=multi_hop_cfg,
        robustness=robustness_cfg
    )


def run_generation():
    parser = argparse.ArgumentParser(description="Epstein Bench Task Generation")
    parser.add_argument("--test", action="store_true", help="Run in test mode (limited docs/tasks)")
    parser.add_argument("--limit", type=int, help="Limit the number of generated tasks (overrides defaults)")
    parser.add_argument("--doc-limit", type=int, help="Limit the number of documents loaded from the corpus")
    parser.add_argument("--output", type=str, default="epstein_bench.jsonl", help="Path to save the generated dataset")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Initializing Epstein Bench Task Generation (Test Mode: {args.test}, Task Limit: {args.limit}, Doc Limit: {args.doc_limit})...")
    logger.info(f"Using backend: {config.tagging_backend}")
    
    epstein_config = get_epstein_config(test_mode=args.test, limit=args.limit, doc_limit=args.doc_limit)
    llm = get_llm_client()
    
    generator = TaskGenerator(llm=llm, config=epstein_config)
    
    logger.info(f"Loading corpus from {epstein_config.hf_corpus.dataset_name}...")
    
    try:
        dataset = generator.generate_dataset(name="epstein_bench_v1")
        
        logger.info(f"Successfully generated {len(dataset)} evaluation samples.")
        
        # Save to file
        output_path = Path(args.output)
        logger.info(f"Saving dataset to {output_path}...")
        save_jsonl_dataset(dataset, output_path)
        logger.info(f"Dataset saved to {output_path}")
        
        # Preview a few
        for i, sample in enumerate(dataset):
            if i >= 5: break
            print(f"\n--- Sample {i+1} ---")
            print(f"Type: {sample.labels.get('type', 'unknown')}")
            print(f"Query: {sample.query}")
            print(f"Answer: {sample.reference_answer.text if sample.reference_answer else 'N/A'}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error during generation: {e}")
        logger.error("Check your network connection or HF dataset availability.")

if __name__ == "__main__":
    run_generation()
