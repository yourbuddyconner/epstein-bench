from datasets import Dataset
from typing import List

from ..core.types import EvaluationSample
from ..taskgen.config import (
    TaskGeneratorConfig, HFCorpusConfig, ChunkingConfig, SingleHopConfig, MultiHopConfig
)
from ..taskgen.generator import TaskGenerator


class MockLLMClient:
    def generate(self, prompt: str) -> str:
        if "multi-hop" in prompt.lower():
            return "Question: What links Entity A and Entity B?\nAnswer: They are both mentioned here."
        if "write a question" in prompt.lower():
            return "What is the capital of France?"
        if "paraphrase" in prompt.lower():
            return "Can you tell me the capital of France?"
        return "Mock response"


def run_demo():
    # 1. Create dummy dataset
    data = {
        "id": ["1", "2"],
        "text": [
            "Paris is the capital of France. It has the Eiffel Tower.",
            "Berlin is the capital of Germany. It has the Brandenburg Gate."
        ],
        "meta": ["meta1", "meta2"]
    }
    ds = Dataset.from_dict(data)

    # 2. Config
    hf_cfg = HFCorpusConfig(
        dataset_obj=ds,
        text_column="text",
        id_column="id",
        metadata_fn=lambda x: {"orig_meta": x["meta"]}
    )
    
    gen_cfg = TaskGeneratorConfig(
        hf_corpus=hf_cfg,
        chunking=ChunkingConfig(max_tokens=50, overlap_tokens=0),
        single_hop=SingleHopConfig(max_tasks=5, min_answer_length=1),
        multi_hop=MultiHopConfig(max_tasks=2)
    )

    # 3. Generator
    llm = MockLLMClient()
    generator = TaskGenerator(llm=llm, config=gen_cfg)

    # 4. Generate
    dataset = generator.generate_dataset("dummy_gen")
    
    print(f"Generated {len(dataset)} samples.")
    for s in dataset:
        print(f"[{s.labels.get('type', 'unknown')}] Q: {s.query} | A: {s.reference_answer.text if s.reference_answer else 'None'}")


if __name__ == "__main__":
    run_demo()

