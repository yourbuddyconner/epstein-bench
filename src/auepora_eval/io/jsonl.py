import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import asdict

from ..core.dataset import InMemoryDataset, EvaluationDataset
from ..core.types import EvaluationSample, Document, Response


def load_jsonl_dataset(path: str | Path, name: Optional[str] = None) -> InMemoryDataset:
    path = Path(path)
    if name is None:
        name = path.stem

    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            # Map dict → EvaluationSample
            relevant_docs = None
            if "relevant_docs" in raw and raw["relevant_docs"]:
                relevant_docs = [Document(**d) for d in raw["relevant_docs"]]

            candidate_docs = None
            if "candidate_docs" in raw and raw["candidate_docs"]:
                candidate_docs = [Document(**d) for d in raw["candidate_docs"]]

            reference_answer = None
            if "reference_answer" in raw and raw["reference_answer"]:
                reference_answer = Response(**raw["reference_answer"])

            sample = EvaluationSample(
                sample_id=raw["sample_id"],
                query=raw["query"],
                relevant_docs=relevant_docs,
                candidate_docs=candidate_docs,
                reference_answer=reference_answer,
                labels=raw.get("labels", {}),
                metadata=raw.get("metadata", {}),
            )
            samples.append(sample)

    return InMemoryDataset(name=name, samples=samples)


def save_jsonl_dataset(dataset: EvaluationDataset, path: str | Path) -> None:
    """Save an EvaluationDataset to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            # Convert sample to dict, handling dataclasses recursively
            # We construct the dict manually to control formatting and avoid clutter
            
            data: dict[str, Any] = {
                "sample_id": sample.sample_id,
                "query": sample.query,
                "labels": sample.labels,
                "metadata": sample.metadata,
            }
            
            if sample.relevant_docs:
                data["relevant_docs"] = [asdict(d) for d in sample.relevant_docs]
            
            if sample.candidate_docs:
                data["candidate_docs"] = [asdict(d) for d in sample.candidate_docs]
                
            if sample.reference_answer:
                data["reference_answer"] = asdict(sample.reference_answer)
            
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
